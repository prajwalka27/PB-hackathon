"""Real-time Network Intrusion Detection agent  (runs on YOUR device).

What it does
  1. Captures the real network traffic of this device with Scapy.
  2. Groups packets into connections and computes the NSL-KDD features from them.
  3. Scores every finished connection with the trained model (model/nids_model.joblib).
  4. Shows the device name and the live threat level on a local dashboard
     (http://127.0.0.1:8765, reachable only from this device) and in the terminal.

Setup (Windows)
  - Install Npcap from https://npcap.com  (tick "WinPcap API-compatible mode")
  - pip install scapy
  - Open PowerShell AS ADMINISTRATOR, go to the project folder, activate the venv, then:
        python agent.py
  Mac / Linux:  pip install scapy   and run   sudo python agent.py

Website pairing
  The agent prints a 6-character pairing code. Enter it on the web app's "Live monitor"
  page and that page shows YOUR device's live results. The data goes from this agent
  straight to your own browser; it is never sent to any server.

Honest limits
  - Packet capture can only compute the network-level features. The 13 login/system
    features (logged_in, root_shell, ...) cannot be seen on the wire and are set to 0.
  - NSL-KDD was recorded in 1999, so modern traffic (QUIC, CDNs, streaming) can cause
    false alarms. Treat the output as a research prototype, not a security product.
"""
import argparse
import csv
import ipaddress
import json
import os
import platform
import queue
import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "nids_model.joblib"
META_PATH = ROOT / "model" / "metadata.json"
LOG_PATH = ROOT / "agent_log.csv"

# Modern web traffic is HTTPS. In NSL-KDD the 'http_443' service is almost only seen in
# attacks, so we treat port 443 as ordinary web traffic ('http') to avoid false alarms.
HTTPS_AS_HTTP = True

SYN_ERRORS = {"s0", "s1", "s2", "s3"}
REJ_ERRORS = {"rej", "rsto", "rstr", "rstos0"}

TCP_SERVICES = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "domain",
    70: "gopher", 79: "finger", 80: "http", 110: "pop_3", 113: "auth", 119: "nntp",
    143: "imap4", 389: "ldap", 443: "http" if HTTPS_AS_HTTP else "http_443",
    8080: "http",
}
UDP_SERVICES = {53: "domain_u", 69: "tftp_u", 123: "ntp_u"}
ICMP_SERVICES = {0: "ecr_i", 8: "eco_i", 13: "tim_i", 14: "tim_i"}

MIN_FLOWS_FOR_RISK = 10     # need this many connections in the last minute to rate risk
WINDOW_SECONDS = 60


# ------------------------------------------------------------ flow tracking ---
def service_of(proto, dport, icmp_type):
    if proto == "tcp":
        return TCP_SERVICES.get(dport, "private")
    if proto == "udp":
        return UDP_SERVICES.get(dport, "private")
    return ICMP_SERVICES.get(icmp_type, "other")


class Flow:
    """One connection (both directions) as seen on the wire."""

    def __init__(self, p):
        self.proto, self.src, self.dst = p["proto"], p["src"], p["dst"]
        self.sport, self.dport = p["sport"], p["dport"]
        self.start = self.last = p["ts"]
        self.src_bytes = self.dst_bytes = 0
        self.urgent = 0
        self.syn = self.synack = self.established = False
        self.fin_o = self.fin_r = self.rst_o = self.rst_r = False
        self.closed = False
        self.service = service_of(self.proto, self.dport, p.get("icmp_type"))
        # a TCP flow whose first packet is not a SYN was already running when we started
        self.midstream = self.proto == "tcp" and not ("S" in p["flags"] and "A" not in p["flags"])

    def update(self, p, from_orig):
        self.last = p["ts"]
        if from_orig:
            self.src_bytes += p["len"]
        else:
            self.dst_bytes += p["len"]
        if self.proto != "tcp":
            return
        f = p["flags"]
        if "U" in f:
            self.urgent += 1
        if from_orig:
            if "S" in f and "A" not in f:
                self.syn = True
            if "A" in f and self.synack:
                self.established = True
            if "F" in f:
                self.fin_o = True
            if "R" in f:
                self.rst_o = True
        else:
            if "S" in f and "A" in f:
                self.synack = True
            if "F" in f:
                self.fin_r = True
            if "R" in f:
                self.rst_r = True
        if self.rst_o or self.rst_r or (self.fin_o and self.fin_r):
            self.closed = True

    def flag(self):
        """NSL-KDD connection-state flag (lower-case), derived from the TCP handshake."""
        if self.proto != "tcp":
            return "sf"
        if self.midstream:
            return "sf" if (self.fin_o and self.fin_r) else "oth"
        if not self.established:
            if self.rst_r and not self.synack:
                return "rej"
            if self.rst_o:
                return "rstos0"
            if self.fin_o and not self.synack:
                return "sh"
            return "s0"
        if self.fin_o and self.fin_r:
            return "sf"
        if self.rst_o:
            return "rsto"
        if self.rst_r:
            return "rstr"
        if self.fin_o:
            return "s2"
        if self.fin_r:
            return "s3"
        return "s1"


class FlowTracker:
    def __init__(self, allowed_services=None):
        self.active = {}
        self.recent = deque(maxlen=5000)   # every flow, in order of start time
        self.allowed = set(allowed_services or [])

    def add(self, p):
        a, b = (p["src"], p["sport"]), (p["dst"], p["dport"])
        key = (p["proto"],) + (a + b if a <= b else b + a)
        flow = self.active.get(key)
        if flow is None:
            flow = Flow(p)
            self.active[key] = flow
            self.recent.append(flow)
            from_orig = True
        else:
            from_orig = (p["src"] == flow.src and p["sport"] == flow.sport)
        flow.update(p, from_orig)

    def sweep(self, now):
        """Return the feature rows of every connection that has finished."""
        done = []
        for key, f in list(self.active.items()):
            idle = now - f.last
            unanswered = f.proto == "tcp" and not f.midstream and not f.established
            if ((f.closed and idle >= 1.0) or (unanswered and idle >= 3.0)
                    or idle >= 5.0 or now - f.start >= 60.0):
                done.append(f)
                del self.active[key]
        return [self._row(f) for f in sorted(done, key=lambda x: x.start)]

    def _row(self, f):
        wnd = [g for g in self.recent if f.start - 2.0 <= g.start <= f.start]
        host = [g for g in wnd if g.dst == f.dst]
        srv = [g for g in wnd if g.service == f.service]
        hist = [g for g in self.recent if g.dst == f.dst and g.start <= f.start][-255:]
        hist_srv = [g for g in hist if g.service == f.service]

        def frac(items, test):
            return sum(1 for g in items if test(g)) / len(items) if items else 0.0

        syn = lambda g: g.flag() in SYN_ERRORS
        rej = lambda g: g.flag() in REJ_ERRORS
        same_srv = frac(host, lambda g: g.service == f.service)
        h_same_srv = len(hist_srv) / len(hist) if hist else 0.0

        svc = f.service
        if self.allowed and svc not in self.allowed:
            svc = "private" if "private" in self.allowed else "other"

        feats = {
            "duration": int(f.last - f.start), "protocol_type": f.proto, "service": svc,
            "flag": f.flag(), "src_bytes": f.src_bytes, "dst_bytes": f.dst_bytes,
            "land": int(f.src == f.dst and f.sport == f.dport), "wrong_fragment": 0,
            "urgent": f.urgent,
            # content features need host-level data; they cannot be seen on the wire
            "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
            "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
            "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0,
            # time-based features (past 2 seconds)
            "count": len(host), "srv_count": len(srv),
            "serror_rate": frac(host, syn), "srv_serror_rate": frac(srv, syn),
            "rerror_rate": frac(host, rej), "srv_rerror_rate": frac(srv, rej),
            "same_srv_rate": same_srv, "diff_srv_rate": 1.0 - same_srv,
            "srv_diff_host_rate": frac(srv, lambda g: g.dst != f.dst),
            # host-based features (recent connections to the same destination host)
            "dst_host_count": len(hist), "dst_host_srv_count": len(hist_srv),
            "dst_host_same_srv_rate": h_same_srv,
            "dst_host_diff_srv_rate": 1.0 - h_same_srv if hist else 0.0,
            "dst_host_same_src_port_rate": frac(hist, lambda g: g.sport == f.sport),
            "dst_host_srv_diff_host_rate": frac(hist_srv, lambda g: g.src != f.src),
            "dst_host_serror_rate": frac(hist, syn),
            "dst_host_srv_serror_rate": frac(hist_srv, syn),
            "dst_host_rerror_rate": frac(hist, rej),
            "dst_host_srv_rerror_rate": frac(hist_srv, rej),
        }
        return {"features": feats, "start": f.start,
                "flow": f"{f.src}:{f.sport} -> {f.dst}:{f.dport} ({f.proto})"}


# ---------------------------------------------------------- shared state ---
class State:
    def __init__(self, device, threshold, code=""):
        self.lock = threading.Lock()
        self.device, self.threshold, self.code = device, threshold, code
        self.packets = 0
        self.flows = 0
        self.window = deque()                # (time, probability, flagged)
        self.alerts = deque(maxlen=30)
        self.recent = deque(maxlen=30)
        self.started = time.time()

    def add_flow(self, rec, prob, flagged):
        now = time.time()
        item = {"time": time.strftime("%H:%M:%S", time.localtime(rec["start"])),
                "flow": rec["flow"], "service": rec["features"]["service"],
                "flag": rec["features"]["flag"], "probability": round(float(prob), 3)}
        with self.lock:
            self.flows += 1
            self.window.append((now, float(prob), bool(flagged)))
            self.recent.appendleft(item)
            if flagged:
                self.alerts.appendleft(item)

    def snapshot(self):
        now = time.time()
        with self.lock:
            while self.window and now - self.window[0][0] > WINDOW_SECONDS:
                self.window.popleft()
            total = len(self.window)
            flagged = sum(1 for w in self.window if w[2])
            top = max((w[1] for w in self.window), default=0.0)
            share = flagged / total if total else 0.0
            if total < MIN_FLOWS_FOR_RISK:
                level, color = "Collecting data", "gray"
            elif share >= 0.60:
                level, color = "Critical", "red"
            elif share >= 0.30:
                level, color = "High", "orange"
            elif share >= 0.10:
                level, color = "Medium", "yellow"
            else:
                level, color = "Low", "green"
            return {"device": self.device, "threshold": self.threshold, "code": self.code,
                    "uptime_s": int(now - self.started), "packets": self.packets,
                    "flows_scored": self.flows, "window_flows": total,
                    "window_flagged": flagged, "share": round(share, 3),
                    "top_probability": round(top, 3), "level": level, "color": color,
                    "alerts": list(self.alerts)[:15], "recent": list(self.recent)[:15]}


# ---------------------------------------------------------- local dashboard ---
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>NIDS agent</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font-family:system-ui,sans-serif;background:#0e1117;color:#eaeaea;margin:0;padding:24px}
h1{margin:0 0 4px}.sub{color:#9aa0aa;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:18px}
.card{background:#1b1f2a;border-radius:10px;padding:14px 16px}.k{color:#9aa0aa;font-size:12px}
.v{font-size:22px;font-weight:600;margin-top:4px;word-break:break-word}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;color:#000}
.green{background:#3ddc84}.yellow{background:#ffd54a}.orange{background:#ff9f43}
.red{background:#ff5252}.gray{background:#8b93a1}
table{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:22px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #2a2f3a}th{color:#9aa0aa;font-weight:500}
</style></head><body>
<h1>🛡️ Network threat monitor</h1><div class="sub" id="sub">Starting…</div>
<div class="grid">
<div class="card"><div class="k">Device name</div><div class="v" id="dev">-</div></div>
<div class="card"><div class="k">System</div><div class="v" id="os">-</div></div>
<div class="card"><div class="k">Local IP</div><div class="v" id="ip">-</div></div>
<div class="card"><div class="k">Threat level (last minute)</div><div class="v"><span class="badge gray" id="lvl">-</span></div></div>
<div class="card"><div class="k">Connections flagged</div><div class="v" id="fl">-</div></div>
<div class="card"><div class="k">Highest attack probability</div><div class="v" id="top">-</div></div>
<div class="card"><div class="k">Packets captured</div><div class="v" id="pk">-</div></div>
<div class="card"><div class="k">Website pairing code</div><div class="v" id="code">-</div></div>
</div>
<h3>🚨 Flagged connections</h3><table id="al"></table>
<h3>Latest connections</h3><table id="rc"></table>
<script>
function fill(id,rows){var t=document.getElementById(id);
 var h="<tr><th>Time</th><th>Connection</th><th>Service</th><th>Flag</th><th>Attack prob.</th></tr>";
 t.innerHTML=h;rows.forEach(function(r){var tr=t.insertRow();
  [r.time,r.flow,r.service,r.flag,(r.probability*100).toFixed(1)+"%"].forEach(function(x){tr.insertCell().textContent=x;});});
 if(!rows.length){var tr=t.insertRow();var td=tr.insertCell();td.colSpan=5;td.textContent="None yet";}}
function tick(){fetch("/api/status").then(function(r){return r.json();}).then(function(s){
 document.getElementById("dev").textContent=s.device.name;
 document.getElementById("os").textContent=s.device.system;
 document.getElementById("ip").textContent=s.device.ip;
 var l=document.getElementById("lvl");l.textContent=s.level;l.className="badge "+s.color;
 document.getElementById("fl").textContent=s.window_flagged+" of "+s.window_flows+" ("+(s.share*100).toFixed(0)+"%)";
 document.getElementById("top").textContent=(s.top_probability*100).toFixed(1)+"%";
 document.getElementById("pk").textContent=s.packets;
 document.getElementById("code").textContent=s.code;
 document.getElementById("sub").textContent="Live. Alert threshold "+s.threshold+" · "+s.flows_scored+" connections scored · running "+s.uptime_s+" s";
 fill("al",s.alerts);fill("rc",s.recent);}).catch(function(){
 document.getElementById("sub").textContent="Agent not reachable. Is it still running?";});}
tick();setInterval(tick,2000);
</script></body></html>"""


CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no look-alike characters


def make_code():
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))


class Limiter:
    """Blocks guessing of the pairing code (20 wrong tries per minute)."""

    def __init__(self, max_fails=20, window=60):
        self.fails, self.max_fails, self.window = deque(), max_fails, window

    def blocked(self):
        now = time.time()
        while self.fails and now - self.fails[0] > self.window:
            self.fails.popleft()
        return len(self.fails) >= self.max_fails

    def fail(self):
        self.fails.append(time.time())


def make_handler(state, limiter):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body=b"", ctype="application/json", cors=False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            if cors:  # only ever sent when the correct pairing code was supplied
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Access-Control-Max-Age", "600")
            if status != 204:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body and status != 204:
                self.wfile.write(body)

        def _host_ok(self):  # blocks DNS-rebinding style access
            return self.headers.get("Host", "").rsplit(":", 1)[0] in ("127.0.0.1", "localhost")

        def _code_state(self, query):
            """None = no code sent, True = correct, False = wrong, 'blocked' = too many tries."""
            code = parse_qs(query).get("code", [""])[0]
            if not code:
                return None
            if limiter.blocked():
                return "blocked"
            if secrets.compare_digest(code.upper().encode(), state.code.encode()):
                return True
            limiter.fail()
            return False

        def do_OPTIONS(self):  # browser pre-flight for the website's request
            u = urlparse(self.path)
            if self._host_ok() and u.path == "/api/status" and self._code_state(u.query) is True:
                self._send(204, cors=True)
            else:
                self._send(403)

        def do_GET(self):
            u = urlparse(self.path)
            if not self._host_ok():
                self._send(403)
            elif u.path == "/api/status":
                cs = self._code_state(u.query)
                body = json.dumps(state.snapshot()).encode()
                if cs == "blocked":
                    self._send(429)
                elif cs is True:                      # website with the right pairing code
                    self._send(200, body, cors=True)
                elif cs is None and self.headers.get("Sec-Fetch-Site", "same-origin") in (
                        "same-origin", "none"):       # this agent's own local page
                    self._send(200, body)
                else:
                    self._send(403, b'{"error":"forbidden"}')
            elif u.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            else:
                self._send(404)

        def log_message(self, *args):
            pass
    return Handler


# ---------------------------------------------------------------- helpers ---
def local_ip():
    try:  # UDP "connect" sends nothing; it only asks the OS which interface would be used
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def device_info():
    return {"name": socket.gethostname(),
            "system": f"{platform.system()} {platform.release()}",
            "ip": local_ip()}


def is_admin():
    try:
        if os.name == "nt":
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except Exception:
        return True


def ignorable(src, dst):
    try:
        for ip in (src, dst):
            a = ipaddress.ip_address(ip)
            if a.is_loopback or a.is_multicast or ip == "255.255.255.255":
                return True
    except ValueError:
        return True
    return False


def packet_to_dict(pkt, IP, TCP, UDP, ICMP, own_port):
    """Turn a Scapy packet into the plain dict the FlowTracker understands."""
    if IP not in pkt:
        return None
    ip = pkt[IP]
    if ignorable(ip.src, ip.dst):
        return None
    d = {"ts": float(pkt.time), "src": ip.src, "dst": ip.dst, "sport": 0, "dport": 0,
         "flags": "", "len": 0, "icmp_type": None}
    if TCP in pkt:
        t = pkt[TCP]
        d.update(proto="tcp", sport=int(t.sport), dport=int(t.dport),
                 flags=str(t.flags), len=len(t.payload))
    elif UDP in pkt:
        u = pkt[UDP]
        d.update(proto="udp", sport=int(u.sport), dport=int(u.dport), len=len(u.payload))
    elif ICMP in pkt:
        d.update(proto="icmp", icmp_type=int(pkt[ICMP].type), len=len(pkt[ICMP].payload))
    else:
        return None
    if own_port in (d["sport"], d["dport"]) and d["proto"] == "tcp":
        return None
    return d


def sniff_loop(q, iface, own_port):
    try:
        from scapy.all import ICMP, IP, TCP, UDP, sniff
    except ImportError:
        print("Scapy is not installed. Run:  pip install scapy")
        os._exit(1)

    def cb(pkt):
        d = packet_to_dict(pkt, IP, TCP, UDP, ICMP, own_port)
        if d:
            q.put(d)

    try:
        sniff(prn=cb, store=False, iface=iface, filter="ip")
    except Exception as exc:
        print(f"\nCould not capture packets: {exc}")
        print("  - Windows: install Npcap (https://npcap.com) and run PowerShell as Administrator")
        print("  - Mac/Linux: run with sudo")
        os._exit(1)


def score_rows(rows, model, features, threshold, state, writer=None):
    import pandas as pd
    df = pd.DataFrame([r["features"] for r in rows])[features]
    proba = model.predict_proba(df)[:, 1]
    for rec, p in zip(rows, proba):
        flagged = p >= threshold
        state.add_flow(rec, p, flagged)
        if writer:
            writer.writerow([rec["features"][c] for c in features]
                            + [round(float(p), 4), time.strftime("%Y-%m-%d %H:%M:%S",
                               time.localtime(rec["start"])), rec["flow"]])
        if flagged:
            print(f"[ALERT] {time.strftime('%H:%M:%S')}  {rec['flow']}  "
                  f"service={rec['features']['service']} flag={rec['features']['flag']}  "
                  f"attack probability {p:.0%}")


def main():
    ap = argparse.ArgumentParser(description="Real-time NIDS agent for this device")
    ap.add_argument("--threshold", type=float, default=0.65, help="attack threshold (default 0.65)")
    ap.add_argument("--iface", default=None, help="network interface (default: automatic)")
    ap.add_argument("--port", type=int, default=8765, help="local dashboard port")
    ap.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    ap.add_argument("--no-log", action="store_true", help="do not write agent_log.csv")
    args = ap.parse_args()

    if not MODEL_PATH.exists() or not META_PATH.exists():
        sys.exit("Model not found. Run  python train.py  first (or pull the model/ folder).")
    import joblib
    from preprocess import FEATURES
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    services = meta["feature_spec"]["service"].get("options", [])

    device = device_info()
    code = make_code()
    state = State(device, args.threshold, code)
    tracker = FlowTracker(services)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state, Limiter()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"

    q = queue.Queue()
    threading.Thread(target=sniff_loop, args=(q, args.iface, args.port), daemon=True).start()

    print(f"Device: {device['name']} · {device['system']} · {device['ip']}")
    print(f"Dashboard: {url}   (only visible on this device)")
    print(f"Website pairing code: {code}   (enter it on the web app's Live monitor page)")
    if not is_admin():
        print("WARNING: not running as Administrator/root. Packet capture may fail.")
    print("Capturing real traffic. Press Ctrl+C to stop.\n")
    if not args.no_browser:
        webbrowser.open(url)

    log_file = writer = None
    if not args.no_log:
        new = not LOG_PATH.exists()
        log_file = open(LOG_PATH, "a", newline="", encoding="utf-8")
        writer = csv.writer(log_file)
        if new:
            writer.writerow(FEATURES + ["attack_probability", "time", "flow"])

    last_sweep = time.time()
    try:
        while True:
            try:
                tracker.add(q.get(timeout=0.5))
                state.packets += 1
                while True:                       # drain whatever else is waiting
                    tracker.add(q.get_nowait())
                    state.packets += 1
            except queue.Empty:
                pass
            now = time.time()
            if now - last_sweep >= 1.0:
                last_sweep = now
                rows = tracker.sweep(now)
                if rows:
                    score_rows(rows, model, FEATURES, args.threshold, state, writer)
                    if log_file:
                        log_file.flush()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()