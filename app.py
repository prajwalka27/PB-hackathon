"""Streamlit web app: Network Intrusion Detection (Normal vs Attack).

Run:  streamlit run app.py
"""
import json
import random
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import accuracy_score, classification_report

from preprocess import CATEGORICAL, COLUMNS, FEATURES, NUMERIC

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "nids_model.joblib"
META_PATH = ROOT / "model" / "metadata.json"
SAMPLE_PATH = ROOT / "sample_data" / "sample_traffic.csv"

st.set_page_config(page_title="NIDS: Normal vs Attack", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


model, meta = load_artifacts()
if model is None:
    st.error("Model not found. Run `python train.py` first, then restart the app.")
    st.stop()
spec = meta["feature_spec"]

# ---------------------------------------------------------------- presets ---
# Illustrative connection profiles for demos. Anything not listed uses the
# dataset median / most common value.
PRESETS = {
    "Typical (dataset median)": {},
    "Normal web browsing": dict(
        protocol_type="tcp", service="http", flag="SF".lower(), src_bytes=232,
        dst_bytes=8153, logged_in=1, count=5, srv_count=5, same_srv_rate=1.0,
        dst_host_count=30, dst_host_srv_count=255, dst_host_same_srv_rate=1.0),
    "DoS: SYN flood": dict(
        protocol_type="tcp", service="private", flag="s0", src_bytes=0, dst_bytes=0,
        logged_in=0, count=270, srv_count=20, serror_rate=1.0, srv_serror_rate=1.0,
        same_srv_rate=0.07, diff_srv_rate=0.06, dst_host_count=255,
        dst_host_srv_count=20, dst_host_same_srv_rate=0.08,
        dst_host_serror_rate=1.0, dst_host_srv_serror_rate=1.0),
    "Probe: port scan": dict(
        protocol_type="tcp", service="private", flag="rej", src_bytes=0, dst_bytes=0,
        logged_in=0, count=120, srv_count=8, rerror_rate=1.0, srv_rerror_rate=1.0,
        same_srv_rate=0.06, diff_srv_rate=0.08, dst_host_count=255,
        dst_host_srv_count=12, dst_host_same_srv_rate=0.05,
        dst_host_diff_srv_rate=0.07, dst_host_rerror_rate=1.0,
        dst_host_srv_rerror_rate=1.0),
}


# ------------------------------------------------- feature explanations ---
# (column name, plain-English name, what it means)
FEATURE_GROUPS = [
    ("🔌 Basic connection details",
     "What the connection looks like by itself.",
     [
         ("duration", "Connection duration (seconds)",
          "How long the connection lasted, in seconds."),
         ("protocol_type", "Protocol",
          "Network protocol used. tcp = reliable, connection-based; udp = fast, no handshake; "
          "icmp = control messages such as ping."),
         ("service", "Target service",
          "The network service being contacted, e.g. http = web, ftp = file transfer, "
          "smtp = email, private = an uncommon or unlisted port."),
         ("flag", "Connection status",
          "How the connection started and ended. sf = normal; s0 = connection attempt never "
          "answered (typical of SYN-flood attacks); rej = connection rejected (typical of port "
          "scans); rsto / rstr = connection reset; others are rarer error states."),
         ("src_bytes", "Bytes sent by source",
          "Data sent from the client (source) to the server. Very small or unusual values can "
          "signal scans or floods."),
         ("dst_bytes", "Bytes sent back by destination",
          "Data sent from the server back to the client."),
         ("land", "Same source and destination?",
          "Yes if the source and destination address and port are identical. This is a known "
          "attack trick (LAND attack); normal traffic is almost always No."),
         ("wrong_fragment", "Malformed fragments",
          "Number of 'wrong' (badly formed) data fragments in the connection."),
         ("urgent", "Urgent packets",
          "Number of packets marked 'urgent'. Rarely used in normal traffic."),
     ]),
    ("🔐 Login & system behaviour",
     "What happened on the target machine during the connection.",
     [
         ("hot", "Sensitive-area accesses",
          "Number of 'hot' indicators: actions such as entering system directories or "
          "creating/executing programs."),
         ("num_failed_logins", "Failed login attempts",
          "How many times a login failed. Many failures suggest password guessing."),
         ("logged_in", "Logged in successfully?",
          "Yes if the login succeeded."),
         ("num_compromised", "Compromised conditions",
          "Number of signs that the system may have been tampered with (e.g. 'file not found' "
          "errors on system files)."),
         ("root_shell", "Admin (root) shell obtained?",
          "Yes if the user got a root/administrator command shell."),
         ("su_attempted", "'su root' attempted?",
          "Whether the 'su root' command (switch to admin user) was tried."),
         ("num_root", "Admin (root) operations",
          "Number of operations performed with root/administrator rights."),
         ("num_file_creations", "Files created",
          "Number of file-creation operations during the connection."),
         ("num_shells", "Shell prompts opened",
          "Number of command-line shells opened."),
         ("num_access_files", "Access-control file operations",
          "Number of operations on sensitive access-control files (e.g. password files)."),
         ("num_outbound_cmds", "Outbound FTP commands",
          "Number of outbound commands in an FTP session (always 0 in this dataset)."),
         ("is_host_login", "Privileged 'host' login?",
          "Yes if the login belongs to the host list (special privileged accounts such as "
          "root or admin)."),
         ("is_guest_login", "Guest login?",
          "Yes if the user logged in as 'guest' or 'anonymous'."),
     ]),
    ("⏱️ Recent traffic (last 2 seconds)",
     "How many connections hit the same host or service in the past 2 seconds. "
     "Floods and scans show up here.",
     [
         ("count", "Connections to same host",
          "Number of connections to the same destination host in the last 2 seconds. "
          "Very high values suggest a flood."),
         ("srv_count", "Connections to same service",
          "Number of connections to the same service (port) in the last 2 seconds."),
         ("serror_rate", "Half-open connection rate (host)",
          "Share (0 to 1) of those connections with SYN errors (started but never completed). "
          "Close to 1 is typical of a SYN flood."),
         ("srv_serror_rate", "Half-open connection rate (service)",
          "Same as above, but counted over connections to the same service."),
         ("rerror_rate", "Rejected connection rate (host)",
          "Share (0 to 1) of connections that were rejected. Close to 1 is typical of port scans."),
         ("srv_rerror_rate", "Rejected connection rate (service)",
          "Same as above, but counted over connections to the same service."),
         ("same_srv_rate", "Same-service rate",
          "Share (0 to 1) of connections going to the same service. Normal browsing is usually high."),
         ("diff_srv_rate", "Different-service rate",
          "Share (0 to 1) of connections going to different services. High values suggest "
          "a port scan."),
         ("srv_diff_host_rate", "Same-service, different-host rate",
          "Share (0 to 1) of same-service connections that go to different hosts."),
     ]),
    ("🖥️ Destination host history (last 100 connections)",
     "Longer-term view of traffic aimed at the same destination host.",
     [
         ("dst_host_count", "Connections to this host",
          "Number of connections to the same destination host among the last 100 (max 255)."),
         ("dst_host_srv_count", "Connections to this host's service",
          "Number of connections to the same host and service among the last 100."),
         ("dst_host_same_srv_rate", "Same-service rate (host history)",
          "Share (0 to 1) of connections to this host that use the same service."),
         ("dst_host_diff_srv_rate", "Different-service rate (host history)",
          "Share (0 to 1) of connections to this host that use different services. High = scan."),
         ("dst_host_same_src_port_rate", "Same source-port rate",
          "Share (0 to 1) of connections to this host coming from the same source port."),
         ("dst_host_srv_diff_host_rate", "Same service, different sources",
          "Share (0 to 1) of connections to this service that come from different hosts."),
         ("dst_host_serror_rate", "Half-open rate (host history)",
          "Share (0 to 1) of connections to this host with SYN errors."),
         ("dst_host_srv_serror_rate", "Half-open rate (host + service history)",
          "Share (0 to 1) of connections to this host and service with SYN errors."),
         ("dst_host_rerror_rate", "Rejected rate (host history)",
          "Share (0 to 1) of connections to this host that were rejected."),
         ("dst_host_srv_rerror_rate", "Rejected rate (host + service history)",
          "Share (0 to 1) of connections to this host and service that were rejected."),
     ]),
]

FEATURE_INFO = {name: (friendly, meaning, group)
                for group, _, items in FEATURE_GROUPS for name, friendly, meaning in items}
YES_NO_FEATURES = {"land", "logged_in", "root_shell", "is_host_login", "is_guest_login"}


def label_for(f):
    friendly = FEATURE_INFO.get(f, (f,))[0]
    return f"{friendly} ({f})" if friendly != f else f


def help_for(f):
    return FEATURE_INFO[f][1] if f in FEATURE_INFO else None


def default_value(f):
    s = spec[f]
    if s["kind"] == "cat":
        return s["default"]
    if s["kind"] in ("binary", "count"):
        return int(s["default"])
    return float(s["default"])


def cast(f, v):
    s = spec[f]
    if s["kind"] == "cat":
        return v if v in s["options"] else s["default"]
    if s["kind"] in ("binary", "count"):
        return int(v)
    return float(v)


def apply_preset(name):
    values = {f: default_value(f) for f in FEATURES}
    values.update({f: cast(f, v) for f, v in PRESETS[name].items()})
    for f, v in values.items():
        st.session_state[f"in_{f}"] = v
    st.session_state["sel_network"] = NET_NONE
    st.session_state["loaded_truth"] = None
    st.session_state["loaded_from"] = None


for f in FEATURES:  # initialise widget state once
    st.session_state.setdefault(f"in_{f}", default_value(f))


def render_input(f):
    s, key, label, tip = spec[f], f"in_{f}", label_for(f), help_for(f)
    if s["kind"] == "cat":
        st.selectbox(label, s["options"], key=key, help=tip)
    elif s["kind"] == "binary":
        if f in YES_NO_FEATURES:
            st.selectbox(label, [0, 1], key=key, help=tip,
                         format_func=lambda v: "Yes" if v == 1 else "No")
        else:
            st.selectbox(label, [0, 1], key=key, help=tip)
    elif s["kind"] == "rate":
        st.number_input(label, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                        key=key, help=tip)
    else:
        st.number_input(label, min_value=0, step=1, key=key, help=tip)


# -------------------------------------------------------------- inference ---
def run_model(df, threshold):
    proba = model.predict_proba(df[FEATURES])[:, 1]  # P(attack)
    return proba, (proba >= threshold).astype(int)


def threat_level(p, threshold):
    """Turn one attack probability into a human-friendly threat level."""
    if p >= 0.90:
        return "🔴 Critical"
    if p >= threshold:
        return "🟠 High"
    if p >= 0.30:
        return "🟡 Medium (suspicious, below alert threshold)"
    return "🟢 Low"


def show_verdict(p, threshold):
    if p >= threshold:
        st.error(f"🚨 **ATTACK DETECTED**: attack probability {p:.1%}")
    else:
        st.success(f"✅ **NORMAL TRAFFIC**: attack probability {p:.1%}")
    st.progress(float(min(max(p, 0.0), 1.0)))
    st.caption(f"Threat level for this connection: **{threat_level(p, threshold)}**")


# ---------------------------------------------- network-level comparison ---
DEMO_NETWORKS = {  # name -> share of attack traffic mixed into the demo log
    "🏠 Home Wi-Fi": 0.03,
    "🏢 Office network": 0.12,
    "🎓 College campus": 0.30,
    "☕ Public Wi-Fi": 0.50,
    "🌐 Web server under DoS attack": 0.85,
}


def network_risk(share):
    """Risk level of a whole network from the share of flagged connections."""
    if share >= 0.60:
        return "🔴 Critical"
    if share >= 0.30:
        return "🟠 High"
    if share >= 0.10:
        return "🟡 Medium"
    return "🟢 Low"


@st.cache_data
def build_demo_networks(path_str, n=150):
    """Build demo network logs by re-sampling real NSL-KDD rows with different attack mixes."""
    raw = pd.read_csv(path_str)
    is_att = raw["label"].astype(str).str.strip().str.lower() != "normal"
    attacks, normals = raw[is_att], raw[~is_att]
    nets = {}
    for i, (name, share) in enumerate(DEMO_NETWORKS.items()):
        n_att = int(round(n * share))
        parts = []
        if n_att and len(attacks):
            parts.append(attacks.sample(n_att, replace=n_att > len(attacks), random_state=i))
        if n - n_att and len(normals):
            parts.append(normals.sample(n - n_att, replace=(n - n_att) > len(normals),
                                        random_state=100 + i))
        if parts:
            nets[name] = pd.concat(parts).sample(frac=1, random_state=7).reset_index(drop=True)
    return nets


def score_network(df, threshold):
    proba, pred = run_model(df, threshold)
    n = len(df)
    return {"proba": proba, "pred": pred, "n": n, "n_att": int(pred.sum()),
            "share": float(pred.mean()) if n else 0.0,
            "avg": float(proba.mean()) if n else 0.0}


def load_uploaded(file):
    """Accept CSVs with a header, or raw NSL-KDD style files without one."""
    df = pd.read_csv(file)
    df.columns = df.columns.astype(str).str.strip()
    if set(FEATURES).issubset(df.columns):
        return df
    file.seek(0)
    raw = pd.read_csv(file, header=None)
    if len(FEATURES) <= raw.shape[1] <= len(COLUMNS):
        raw.columns = COLUMNS[: raw.shape[1]]
        return raw
    missing = [c for c in FEATURES if c not in df.columns]
    raise ValueError(f"CSV is missing {len(missing)} required columns, e.g. {missing[:6]}")


def clean_uploaded(df):
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce")  # NaNs imputed by model
    return df


NET_NONE = "— Custom (no network) —"


def safe_value(f, v):
    """Convert a value from a data row into something the form widget accepts."""
    s = spec[f]
    if pd.isna(v):
        return default_value(f)
    if s["kind"] == "cat":
        v = str(v).strip().lower()
        return v if v in s["options"] else s["default"]
    if s["kind"] == "binary":
        return 1 if float(v) > 0 else 0
    if s["kind"] == "rate":
        return float(min(max(float(v), 0.0), 1.0))
    return int(max(float(v), 0))


def load_random_connection():
    """Fill the manual form with a real connection from the selected network."""
    name = st.session_state.get("sel_network", NET_NONE)
    if name == NET_NONE or not SAMPLE_PATH.exists():
        return
    nets = build_demo_networks(str(SAMPLE_PATH))
    if name not in nets:
        return
    row = nets[name].iloc[random.randrange(len(nets[name]))]
    for f in FEATURES:
        st.session_state[f"in_{f}"] = safe_value(f, row[f])
    lab = str(row.get("label", "")).strip().lower()
    st.session_state["loaded_truth"] = lab or None
    st.session_state["loaded_from"] = name


# --------------------------------------------------- visitor device info ---
def parse_user_agent(ua):
    """Rough OS / browser / device-type guess from the browser's User-Agent text."""
    u = (ua or "").lower()
    if not u:
        return "Unknown", "Unknown", "Unknown"
    if "iphone" in u:
        os_name, device = "iOS", "Phone"
    elif "ipad" in u:
        os_name, device = "iPadOS", "Tablet"
    elif "android" in u:
        os_name = "Android"
        device = "Phone" if "mobile" in u else "Tablet"
    elif "windows" in u:
        os_name, device = "Windows", "Computer"
    elif "cros" in u:
        os_name, device = "ChromeOS", "Computer"
    elif "mac os" in u or "macintosh" in u:
        os_name, device = "macOS", "Computer"
    elif "linux" in u:
        os_name, device = "Linux", "Computer"
    else:
        os_name, device = "Unknown", "Unknown"
    if "edg/" in u or "edga/" in u or "edgios/" in u:
        browser = "Microsoft Edge"
    elif "opr/" in u or "opera" in u:
        browser = "Opera"
    elif "samsungbrowser" in u:
        browser = "Samsung Internet"
    elif "firefox/" in u or "fxios" in u:
        browser = "Firefox"
    elif "chrome/" in u or "crios" in u:
        browser = "Chrome"
    elif "safari/" in u:
        browser = "Safari"
    else:
        browser = "Unknown"
    return device, os_name, browser


def get_request_info():
    """User-Agent and public IP as seen by the server (empty if unavailable)."""
    try:
        headers = st.context.headers
        ua = headers.get("User-Agent", "")
        fwd = headers.get("X-Forwarded-For", "")
        return ua, (fwd.split(",")[0].strip() if fwd else "")
    except Exception:
        return "", ""


CLIENT_INFO_HTML = """
<div style="font-family:system-ui,sans-serif;background:#262730;color:#eaeaea;
            border-radius:10px;padding:14px 18px;font-size:14px;line-height:1.7">
  <table id="t" style="width:100%;border-collapse:collapse"></table>
</div>
<script>
  const c = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  const rows = [
    ["Screen", screen.width + " × " + screen.height + " px (pixel ratio " + (window.devicePixelRatio || 1) + ")"],
    ["Touch screen", navigator.maxTouchPoints > 0 ? "Yes (" + navigator.maxTouchPoints + " touch points)" : "No"],
    ["Platform", (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "Unknown"],
    ["Language", navigator.language || "Unknown"],
    ["Time zone", Intl.DateTimeFormat().resolvedOptions().timeZone || "Unknown"],
    ["Online", navigator.onLine ? "Yes" : "No"],
    ["Connection quality", c ? ((c.effectiveType || "?").toUpperCase() + " · about " + c.downlink + " Mbps · latency " + c.rtt + " ms" + (c.saveData ? " · data saver ON" : "")) : "Not reported by this browser"],
    ["CPU cores", navigator.hardwareConcurrency || "Not reported"],
    ["Device memory", navigator.deviceMemory ? "about " + navigator.deviceMemory + " GB" : "Not reported"],
    ["Cookies enabled", navigator.cookieEnabled ? "Yes" : "No"]
  ];
  document.getElementById("t").innerHTML = rows.map(function (r) {
    return "<tr><td style='padding:3px 12px 3px 0;color:#9aa0aa;white-space:nowrap;vertical-align:top'>" + r[0] +
           "</td><td style='padding:3px 0'>" + r[1] + "</td></tr>";
  }).join("");
</script>
"""


LIVE_HTML = """
<style>
 body{margin:0}
 .wrap{font-family:system-ui,sans-serif;background:#0e1117;color:#eaeaea;padding:16px;border-radius:12px}
 .sub{color:#9aa0aa;margin-bottom:12px}
 .err{display:none;background:#3a1d1d;border-radius:8px;padding:12px 14px;margin-bottom:12px;line-height:1.6}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:16px}
 .card{background:#1b1f2a;border-radius:10px;padding:12px 14px}
 .k{color:#9aa0aa;font-size:12px}.v{font-size:20px;font-weight:600;margin-top:4px;word-break:break-word}
 .badge{display:inline-block;padding:3px 14px;border-radius:20px;font-weight:700;color:#000}
 .green{background:#3ddc84}.yellow{background:#ffd54a}.orange{background:#ff9f43}
 .red{background:#ff5252}.gray{background:#8b93a1}
 table{width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:18px}
 th,td{text-align:left;padding:5px 8px;border-bottom:1px solid #2a2f3a}th{color:#9aa0aa;font-weight:500}
 h3{margin:6px 0 8px;font-size:16px}
</style>
<div class="wrap">
 <div class="sub" id="sub">Connecting to the agent on this device...</div>
 <div class="err" id="err"></div>
 <div class="grid">
  <div class="card"><div class="k">Device name</div><div class="v" id="dev">-</div></div>
  <div class="card"><div class="k">System</div><div class="v" id="os">-</div></div>
  <div class="card"><div class="k">Local IP</div><div class="v" id="ip">-</div></div>
  <div class="card"><div class="k">Threat level (last minute)</div><div class="v"><span class="badge gray" id="lvl">-</span></div></div>
  <div class="card"><div class="k">Connections flagged</div><div class="v" id="fl">-</div></div>
  <div class="card"><div class="k">Highest attack probability</div><div class="v" id="top">-</div></div>
  <div class="card"><div class="k">Packets captured</div><div class="v" id="pk">-</div></div>
 </div>
 <h3>Flagged connections</h3><table id="al"></table>
 <h3>Latest connections</h3><table id="rc"></table>
</div>
<script>
 var URL = "http://127.0.0.1:__PORT__/api/status?code=__CODE__";
 function fill(id, rows) {
   var t = document.getElementById(id);
   t.innerHTML = "<tr><th>Time</th><th>Connection</th><th>Service</th><th>Flag</th><th>Attack prob.</th></tr>";
   rows.forEach(function (r) {
     var tr = t.insertRow();
     [r.time, r.flow, r.service, r.flag, (r.probability * 100).toFixed(1) + "%"].forEach(function (x) {
       tr.insertCell().textContent = x;
     });
   });
   if (!rows.length) { var tr = t.insertRow(); var td = tr.insertCell(); td.colSpan = 5; td.textContent = "None yet"; }
 }
 function ok(s) {
   document.getElementById("err").style.display = "none";
   document.getElementById("dev").textContent = s.device.name;
   document.getElementById("os").textContent = s.device.system;
   document.getElementById("ip").textContent = s.device.ip;
   var l = document.getElementById("lvl"); l.textContent = s.level; l.className = "badge " + s.color;
   document.getElementById("fl").textContent = s.window_flagged + " of " + s.window_flows + " (" + (s.share * 100).toFixed(0) + "%)";
   document.getElementById("top").textContent = (s.top_probability * 100).toFixed(1) + "%";
   document.getElementById("pk").textContent = s.packets;
   document.getElementById("sub").textContent = "Live from your device. Alert threshold " + s.threshold + " | " + s.flows_scored + " connections scored | agent running " + s.uptime_s + " s";
   fill("al", s.alerts); fill("rc", s.recent);
 }
 function fail() {
   var e = document.getElementById("err"); e.style.display = "block";
   e.textContent = "Could not reach the agent on this device. Check that: (1) python agent.py is running, (2) the pairing code and port match what the agent shows, (3) you are using Chrome, Edge or Firefox, and you clicked Allow if the browser asked about access to apps or services on this device. Wrong codes are blocked for a minute after 20 tries.";
   document.getElementById("sub").textContent = "Not connected";
 }
 function tick() {
   fetch(URL).then(function (r) { if (!r.ok) throw new Error("http"); return r.json(); }).then(ok).catch(fail);
 }
 tick(); setInterval(tick, 2000);
</script>
"""


# ------------------------------------------------------------------- UI ---
st.title("🛡️ Network Intrusion Detection System")
st.caption(f"Classifies network connections as **Normal** or **Attack** · model: "
           f"{meta['best_model']} · trained on NSL-KDD")

with st.sidebar:
    st.header("⚙️ Detection settings")
    threshold = st.slider("Attack threshold", 0.05, 0.95, 0.65, 0.05,
                          help="Lower = more sensitive (catches more attacks, more false alarms).")
    tm = meta["test_metrics"]
    st.divider()
    st.subheader("Model quality (KDDTest+)")
    st.metric("Accuracy", f"{tm['accuracy']:.1%}")
    st.metric("Attack recall", f"{tm['recall']:.1%}")
    st.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")

tab_manual, tab_csv, tab_info, tab_net, tab_live, tab_device, tab_guide = st.tabs(
    ["✍️ Manual input", "📁 Upload CSV logs", "📊 Model insights",
     "🌐 Network comparison", "📡 Live monitor", "📱 My device", "📖 Feature guide"])

# ---- Tab 1: manual form -------------------------------------------------
with tab_manual:
    st.markdown("Pick a **network** to load a real connection from it (the values change for "
                "every network), or load a demo profile. Tweak the values if you like, "
                "then analyze.")
    if SAMPLE_PATH.exists():
        c_sel, c_btn = st.columns([3, 1])
        with c_sel:
            st.selectbox(
                "🌐 Network", [NET_NONE] + list(DEMO_NETWORKS), key="sel_network",
                on_change=load_random_connection,
                help="Each network has a different mix of normal and attack traffic. "
                     "Choosing one fills the form with a real connection from that network.")
        with c_btn:
            st.markdown("&nbsp;")
            st.button("🎲 Another connection", on_click=load_random_connection,
                      use_container_width=True,
                      disabled=st.session_state.get("sel_network", NET_NONE) == NET_NONE)
        sel = st.session_state.get("sel_network", NET_NONE)
        if sel != NET_NONE:
            share = DEMO_NETWORKS[sel]
            st.caption(f"{sel}: about {share:.0%} of its traffic is attack-like "
                       f"({network_risk(share)} network). Each click loads a different "
                       f"real connection from it.")
    cols = st.columns(len(PRESETS))
    for c, name in zip(cols, PRESETS):
        c.button(name, on_click=apply_preset, args=(name,), use_container_width=True)

    with st.form("manual_form"):
        st.markdown("**Most influential features**")
        st.caption("These matter most to the model. Hover over the ⓘ next to any field "
                   "for a plain-English explanation.")
        top = meta["top_features"]
        cols = st.columns(3)
        for i, f in enumerate(top):
            with cols[i % 3]:
                render_input(f)
        with st.expander("All other features (grouped)"):
            for group, blurb, items in FEATURE_GROUPS:
                rest = [name for name, _, _ in items if name not in top and name in spec]
                if not rest:
                    continue
                st.markdown(f"**{group}**")
                st.caption(blurb)
                cols = st.columns(3)
                for i, f in enumerate(rest):
                    with cols[i % 3]:
                        render_input(f)
        submitted = st.form_submit_button("🔍 Analyze traffic", type="primary")

    if submitted:
        row = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in FEATURES}])
        proba, _ = run_model(row, threshold)
        show_verdict(float(proba[0]), threshold)
        truth = st.session_state.get("loaded_truth")
        if truth:
            actual = "Normal" if truth == "normal" else f"Attack ({truth})"
            st.info(f"The connection originally loaded from "
                    f"**{st.session_state.get('loaded_from')}** is labelled **{actual}** in the "
                    f"dataset. If you edited the values, the result above reflects your edits.")

# ---- Tab 2: CSV upload --------------------------------------------------
with tab_csv:
    st.markdown("Upload a CSV of connection logs with the 41 NSL-KDD feature columns "
                "(a raw NSL-KDD `.txt/.csv` without a header also works).")
    if SAMPLE_PATH.exists():
        st.download_button("⬇️ Download sample CSV to try", SAMPLE_PATH.read_bytes(),
                           "sample_traffic.csv", "text/csv")
    up = st.file_uploader("Network logs (CSV)", type=["csv", "txt"])
    if up is not None:
        try:
            df = clean_uploaded(load_uploaded(up))
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            st.stop()

        proba, pred = run_model(df, threshold)
        out = df.copy()
        out["prediction"] = pd.Series(pred, index=out.index).map({0: "Normal", 1: "Attack"})
        out["attack_probability"] = proba.round(4)

        n, n_att = len(out), int(pred.sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Connections analysed", f"{n:,}")
        c2.metric("Attacks flagged", f"{n_att:,}")
        c3.metric("Attack share", f"{n_att / n:.1%}")
        if n_att:
            st.warning(f"⚠️ {n_att:,} suspicious connections found.")
        else:
            st.success("No attacks detected in this file.")

        st.bar_chart(out["prediction"].value_counts())
        view = out.sort_values("attack_probability", ascending=False)
        lead = ["prediction", "attack_probability"]
        st.dataframe(view[lead + [c for c in view.columns if c not in lead]].head(1000),
                     use_container_width=True)
        st.download_button("⬇️ Download results CSV", out.to_csv(index=False).encode(),
                           "nids_predictions.csv", "text/csv")

        if "label" in df.columns:  # optional ground truth -> quick accuracy check
            lab = df["label"]
            if pd.api.types.is_numeric_dtype(lab):
                truth = (lab.astype(int) > 0).astype(int)
            else:
                truth = (lab.astype(str).str.strip().str.lower() != "normal").astype(int)
            st.info(f"Ground-truth label found. Accuracy on this file: "
                    f"**{accuracy_score(truth, pred):.1%}**")
            st.code(classification_report(truth, pred, target_names=["Normal", "Attack"],
                                          zero_division=0))

# ---- Tab 3: insights ----------------------------------------------------
with tab_info:
    st.subheader("Top feature importances")
    imp = pd.DataFrame(meta["feature_importance"]).set_index("feature")
    st.bar_chart(imp)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion matrix (official test set)")
        cm = pd.DataFrame(tm["confusion_matrix"], index=["Actual Normal", "Actual Attack"],
                          columns=["Pred Normal", "Pred Attack"])
        st.dataframe(cm)
    with c2:
        st.subheader("Model comparison (validation split)")
        comp = pd.DataFrame(meta["validation_comparison"]).T[
            ["accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds"]]
        st.dataframe(comp.style.format("{:.4f}", subset=comp.columns[:-1]))
    st.caption("Validation scores come from a random split of the training file; the "
               "official KDDTest+ set contains attack types unseen in training, so its "
               "scores are lower and more realistic.")

# ---- Tab 4: network comparison -----------------------------------------
with tab_net:
    st.subheader("Compare the threat level of different networks")
    st.markdown(
        "A single connection gets its own attack probability. A **network** is a collection of "
        "connections, so its threat level comes from **how much of its traffic looks like an "
        "attack**. Different networks therefore get different scores."
    )
    st.caption("Network threat level, by share of flagged connections: "
               "🟢 Low < 10% · 🟡 Medium 10–30% · 🟠 High 30–60% · 🔴 Critical ≥ 60%. "
               "Uses the attack threshold from the sidebar.")

    source = st.radio("Which networks do you want to compare?",
                      ["Demo networks", "Upload my own logs"], horizontal=True)
    networks = {}
    if source == "Demo networks":
        st.caption("Demo networks are made by re-sampling real NSL-KDD records with different "
                   "attack mixes, to show how threat levels differ. They are simulations, not "
                   "captures from real places.")
        if SAMPLE_PATH.exists():
            try:
                networks = build_demo_networks(str(SAMPLE_PATH))
            except Exception as exc:
                st.error(f"Could not build demo networks: {exc}")
        else:
            st.info("Sample data not found. Run `python train.py` to create it, "
                    "or upload your own logs.")
    else:
        st.caption("Upload one CSV per network. The file name is used as the network name "
                   "(e.g. `hostel_wifi.csv`).")
        files = st.file_uploader("Network logs (one CSV per network)", type=["csv", "txt"],
                                 accept_multiple_files=True, key="network_files")
        for f in files or []:
            try:
                networks[Path(f.name).stem] = load_uploaded(f)
            except Exception as exc:
                st.error(f"{f.name}: {exc}")

    if networks:
        results, rows = {}, []
        for name, raw_df in networks.items():
            df_n = clean_uploaded(raw_df)
            r = score_network(df_n, threshold)
            results[name] = (df_n, r)
            rows.append({"Network": name, "Connections": r["n"],
                         "Attacks flagged": r["n_att"],
                         "Attack share (%)": round(r["share"] * 100, 1),
                         "Avg attack probability (%)": round(r["avg"] * 100, 1),
                         "Threat level": network_risk(r["share"])})
        summary = pd.DataFrame(rows).sort_values("Attack share (%)", ascending=False)

        top = summary.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Networks compared", len(summary))
        c2.metric("Highest threat", top["Network"])
        c3.metric("Its attack share", f"{top['Attack share (%)']:.1f}%")

        st.dataframe(
            summary, hide_index=True, use_container_width=True,
            column_config={"Attack share (%)": st.column_config.ProgressColumn(
                "Attack share (%)", min_value=0, max_value=100, format="%.1f%%")})
        st.bar_chart(summary.set_index("Network")["Attack share (%)"])
        st.download_button("⬇️ Download comparison report (CSV)",
                           summary.to_csv(index=False).encode(),
                           "network_threat_comparison.csv", "text/csv")

        st.markdown("#### Most suspicious connections per network")
        show_cols = [c for c in ["protocol_type", "service", "flag", "src_bytes",
                                 "dst_bytes", "count"] if c in FEATURES]
        for name in summary["Network"]:
            df_n, r = results[name]
            with st.expander(f"{name}: {network_risk(r['share'])} "
                             f"({r['share']:.1%} flagged)"):
                worst = df_n[show_cols].copy()
                worst["attack_probability"] = r["proba"].round(3)
                st.dataframe(worst.sort_values("attack_probability", ascending=False).head(5),
                             hide_index=True, use_container_width=True)

# ---- Tab 5: live monitor (real traffic from the agent on the visitor's device) ----
with tab_live:
    st.subheader("📡 Live monitor of your own device")
    st.markdown(
        "This page shows **real, live results from the agent running on your own device**: "
        "your real device name and the threat level of your real network traffic. "
        "A website cannot see this by itself, so the agent does the capturing and this page only "
        "displays it. The data goes from the agent straight to **your browser** and is never "
        "sent to our server."
    )
    st.markdown(
        "**How to use it**\n"
        "1. Download `agent.py` and `requirements-agent.txt` from the project's GitHub repo "
        "(or from the presenter).\n"
        "2. Install **Npcap** (npcap.com, tick *WinPcap API-compatible mode*) and run "
        "`pip install -r requirements-agent.txt`.\n"
        "3. Open PowerShell **as Administrator** in the project folder and run `python agent.py`.\n"
        "4. Enter the **6-character pairing code** that the agent prints below. "
        "Use Chrome, Edge or Firefox, and click **Allow** if the browser asks about access to "
        "apps or services on your device."
    )
    c_code, c_port = st.columns([2, 1])
    code_in = c_code.text_input("Pairing code (shown by the agent)", max_chars=6,
                                key="live_code", placeholder="e.g. K7M4QX")
    port_in = c_port.number_input("Agent port", min_value=1024, max_value=65535,
                                  value=8765, step=1, key="live_port")
    code_clean = re.sub(r"[^A-Z0-9]", "", code_in.upper())
    if len(code_clean) == 6:
        components.html(
            LIVE_HTML.replace("__PORT__", str(int(port_in))).replace("__CODE__", code_clean),
            height=1050, scrolling=True)
    else:
        st.info("Enter the 6-character pairing code from your agent window to start the live view.")
    st.caption("Limits: the model was trained on 1999-era NSL-KDD data, so modern traffic can "
               "cause false alarms. The login/system features cannot be seen from network traffic "
               "and are set to 0. Treat this as a research prototype.")

# ---- Tab 6: visitor device ----------------------------------------------
with tab_device:
    st.subheader("📱 Your device and connection")
    st.caption("This is what any website can see about the device you are using right now. "
               "It is shown only to you and nothing is stored or sent anywhere.")

    ua, ip = get_request_info()
    device, os_name, browser = parse_user_agent(ua)
    c1, c2, c3 = st.columns(3)
    c1.metric("Device type", device)
    c2.metric("Operating system", os_name)
    c3.metric("Browser", browser)
    if ip:
        st.markdown(f"**Public IP address seen by the server:** `{ip}`")
        st.caption("This is the address of your network (Wi-Fi router or mobile carrier), "
                   "so everyone on the same network shows the same one.")
    components.html(CLIENT_INFO_HTML, height=370)

    st.markdown("#### What a web page cannot do")
    st.info(
        "A browser deliberately hides your **device name** (for example \"Rahul's phone\"), "
        "your other apps, and your network traffic from every website. For that reason, this "
        "page **cannot detect threats on your device** or tell you whether your own network "
        "is under attack. Any site that claims to do that from a link alone is guessing."
    )
    st.markdown(
        "**How real per-device threat detection works**\n"
        "1. A small **agent program** is installed on the device (or router) and records its "
        "connections.\n"
        "2. The agent turns them into features such as the 41 used here.\n"
        "3. A model like ours scores them, and the result is shown on a dashboard.\n\n"
        "To check a real network with this project today, export its connection log and use the "
        "**📁 Upload CSV logs** or **🌐 Network comparison** tabs."
    )

# ---- Tab 7: feature guide -----------------------------------------------
with tab_guide:
    st.subheader("What do these features mean?")
    st.markdown(
        "Each network connection is described by **41 measurements**. "
        "The model looks at all of them together to decide if the traffic is Normal or an Attack. "
        "The *column name* is what you need in an uploaded CSV."
    )

    st.markdown("#### Quick patterns")
    st.markdown(
        "- **Normal web browsing:** status `sf`, logged in, steady same-service traffic, little or no errors.\n"
        "- **DoS / SYN flood:** status `s0`, huge `count`, error rates near 1.\n"
        "- **Port scan (Probe):** status `rej`, high different-service rate, rejected rates near 1."
    )

    for group, blurb, items in FEATURE_GROUPS:
        st.markdown(f"#### {group}")
        st.caption(blurb)
        table = pd.DataFrame(
            [{"Column name": name, "Plain name": friendly, "What it means": meaning}
             for name, friendly, meaning in items]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)