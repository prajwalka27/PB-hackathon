"""Streamlit web app: Network Intrusion Detection (Normal vs Attack).

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
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


for f in FEATURES:  # initialise widget state once
    st.session_state.setdefault(f"in_{f}", default_value(f))


def render_input(f):
    s, key = spec[f], f"in_{f}"
    if s["kind"] == "cat":
        st.selectbox(f, s["options"], key=key)
    elif s["kind"] == "binary":
        st.selectbox(f, [0, 1], key=key)
    elif s["kind"] == "rate":
        st.number_input(f, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key=key)
    else:
        st.number_input(f, min_value=0, step=1, key=key)


# -------------------------------------------------------------- inference ---
def run_model(df, threshold):
    proba = model.predict_proba(df[FEATURES])[:, 1]  # P(attack)
    return proba, (proba >= threshold).astype(int)


def show_verdict(p, threshold):
    if p >= threshold:
        st.error(f"🚨 **ATTACK DETECTED**: attack probability {p:.1%}")
    else:
        st.success(f"✅ **NORMAL TRAFFIC**: attack probability {p:.1%}")
    st.progress(float(min(max(p, 0.0), 1.0)))


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


# ------------------------------------------------------------------- UI ---
st.title("🛡️ Network Intrusion Detection System")
st.caption(f"Classifies network connections as **Normal** or **Attack** · model: "
           f"{meta['best_model']} · trained on NSL-KDD")

with st.sidebar:
    st.header("⚙️ Detection settings")
    threshold = st.slider("Attack threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Lower = more sensitive (catches more attacks, more false alarms).")
    tm = meta["test_metrics"]
    st.divider()
    st.subheader("Model quality (KDDTest+)")
    st.metric("Accuracy", f"{tm['accuracy']:.1%}")
    st.metric("Attack recall", f"{tm['recall']:.1%}")
    st.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")

tab_manual, tab_csv, tab_info = st.tabs(["✍️ Manual input", "📁 Upload CSV logs", "📊 Model insights"])

# ---- Tab 1: manual form -------------------------------------------------
with tab_manual:
    st.markdown("Load a demo profile, tweak features if you like, then analyze.")
    cols = st.columns(len(PRESETS))
    for c, name in zip(cols, PRESETS):
        c.button(name, on_click=apply_preset, args=(name,), use_container_width=True)

    with st.form("manual_form"):
        st.markdown("**Most influential features**")
        top = meta["top_features"]
        cols = st.columns(3)
        for i, f in enumerate(top):
            with cols[i % 3]:
                render_input(f)
        with st.expander("All other features"):
            cols = st.columns(3)
            for i, f in enumerate([x for x in FEATURES if x not in top]):
                with cols[i % 3]:
                    render_input(f)
        submitted = st.form_submit_button("🔍 Analyze traffic", type="primary")

    if submitted:
        row = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in FEATURES}])
        proba, _ = run_model(row, threshold)
        show_verdict(float(proba[0]), threshold)

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