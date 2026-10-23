"""Phase 8: Streamlit dashboard. Talks only to the FastAPI backend."""
import os
import time

import pandas as pd
import requests
import streamlit as st

#API = os.getenv("API_URL", "http://127.0.0.1:8000")
API = os.getenv("API_URL", "")
if not API:
    try:
        API = st.secrets["API_URL"]          # set in Streamlit Cloud's Secrets box
    except Exception:
        API = "http://127.0.0.1:8000"
API = API.rstrip("/")
ICON = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
st.set_page_config(page_title="Predictive Maintenance", layout="wide")


def call(method: str, path: str, timeout: int = 30, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=timeout, **kw)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API problem on {path}: {e}")
        return None


st.title("🏭 Predictive Maintenance Dashboard")
machines = call("GET", "/machines")
#if not machines:
#    st.warning("Cannot reach the API. Start it first:  uvicorn src.api.main:app --port 8000")
#    st.stop()
#last = {m["unit"]: m["last_cycle"] for m in machines}

def wake_backend() -> bool:
    """Free servers sleep when idle. Knock until the backend answers."""
    for _ in range(12):
        try:
            if requests.get(f"{API}/health", timeout=25).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


if not st.session_state.get("awake"):
    with st.spinner("Waking up the backend (free servers sleep, this can take about a minute)..."):
        st.session_state.awake = wake_backend()

machines = call("GET", "/machines") if st.session_state.awake else None
if not machines:
    st.session_state.awake = False
    st.warning("Cannot reach the backend yet. Wait a minute, then refresh the page.")
    st.stop()

ss = st.session_state


def reset(unit: int) -> None:
    ss.unit, ss.trace, ss.running, ss.analysis = unit, [], False, None
    ss.cycle = max(10, last[unit] - 80)        # start 80 cycles before the end of its life


ss.setdefault("unit", None)
with st.sidebar:
    st.header("Simulation")
    units = list(last)
    unit = st.selectbox("Machine", units, index=units.index(35) if 35 in units else 0)
    step = st.slider("Cycles per tick", 1, 5, 2)
    delay = st.slider("Seconds per tick", 0.1, 2.0, 0.4)
    a, b, c = st.columns(3)
    if a.button("▶ Start"):
        ss.running = True
    if b.button("⏸ Pause"):
        ss.running = False
    if c.button("↺ Reset"):
        reset(unit)
    st.caption("Each tick replays the machine at a later cycle and asks the API for a prediction.")
if unit != ss.unit:
    reset(unit)

# ---- one simulation tick ----
if ss.running:
    if ss.cycle > last[unit]:
        ss.running = False
    else:
        pred = call("POST", "/predict", json={"unit": unit, "cycle": ss.cycle})
        if pred:
            ss.trace.append(pred)
            ss.cycle += step
        else:
            ss.running = False

live, hist, perf = st.tabs(["Live monitor", "Alerts & history", "Model metrics"])

with live:
    if not ss.trace:
        st.info("Pick a machine and press ▶ Start in the sidebar.")
    else:
        cur = ss.trace[-1]
        lvl = cur["level"]
        banner = f"{ICON[lvl]} Machine {unit}, cycle {cur['cycle']}: Level {lvl} {cur['level_name']}"
        {4: st.error, 3: st.warning, 2: st.info}.get(lvl, st.success)(banner)
        for why in cur["reasons"]:
            st.caption("• " + why)

        m = st.columns(4)
        m[0].metric("Failure risk", f"{cur['risk']:.1%}", f"alarm line {cur['threshold']:.0%}",
                    delta_color="off")
        m[1].metric("Estimated cycles left",
                    f"{cur['rul']:.0f}" + ("+" if cur["rul"] >= 124 else ""))
        m[2].metric("Isolation Forest", "FLAGGED" if cur["iforest_flag"] else "normal")
        #m[3].metric("Autoencoder", "FLAGGED" if cur["ae_flag"] else "normal")
        m[3].metric("Autoencoder", "off (lite)" if cur.get("lite")
                    else ("FLAGGED" if cur["ae_flag"] else "normal"))
        if cur.get("lite"):
            st.caption("The live demo runs in lightweight mode (XGBoost remaining-life model, "
                       "Isolation Forest only). The full version with LSTM and autoencoder "
                       "runs through Docker; see the README.")
        df = pd.DataFrame(ss.trace).set_index("cycle")
        df["alarm line"] = cur["threshold"]
        left, right = st.columns(2)
        left.subheader("Failure risk")
        left.line_chart(df[["risk", "alarm line"]])
        right.subheader("Estimated cycles left")
        right.line_chart(df[["rul"]])

        st.subheader("Why? Biggest pushes on risk right now (SHAP)")
        drv = pd.DataFrame(cur["drivers"]).set_index("label")["shap"]
        st.bar_chart(drv)
        st.caption("Above zero = pushes risk up. Below zero = pushes risk down.")

        if st.button("🤖 Explain & recommend (uses the AI agents)"):
            ss.running = False
            with st.spinner("Detector → Explainer → Recommender..."):
                ss.analysis = call("POST", "/analyze", timeout=90,
                                   json={"unit": unit, "cycle": cur["cycle"]})
        if ss.analysis:
            an = ss.analysis
            st.markdown(f"**AI report for cycle {an['cycle']}**  ·  agents: "
                        f"{' → '.join(an['agents_ran'])}")
            st.write(an["explanation"])
            st.markdown(an["actions"])
            if an["unknown_citations"]:
                st.error(f"Unverified citations: {an['unknown_citations']}")

with hist:
    level = st.selectbox("Show predictions at or above level", [1, 2, 3, 4], index=1)
    rows = call("GET", "/history", params={"min_level": level, "limit": 300})
    if rows:
        d = pd.DataFrame(rows)[["ts", "unit", "cycle", "risk", "rul", "level_name",
                                "iforest_flag", "ae_flag", "n_drift"]]
        st.dataframe(d, use_container_width=True, hide_index=True)
    else:
        st.info("Nothing logged at this level yet. Run a simulation first.")

with perf:
    mt = call("GET", "/metrics")
    if mt:
        def flatten(d: dict) -> pd.DataFrame:
            rows = []
            for model, v in d.items():
                if all(isinstance(x, dict) for x in v.values()):
                    rows += [{"model": model, "data": split, **m} for split, m in v.items()]
                else:                                   # the dumb baseline has no splits
                    rows.append({"model": model, "data": "test_last", **v})
            return pd.DataFrame(rows).round(3)

        for title, key in [("Failure classifier (danger within 30 cycles)", "failure"),
                           ("Remaining-useful-life models (errors in cycles)", "rul")]:
            st.subheader(title)
            if mt.get(key):
                st.dataframe(flatten(mt[key]), use_container_width=True, hide_index=True)
        st.caption("test_all = every test row. test_last = last row of each engine (NASA style). "
                   "Validation numbers are optimistic because validation tuned the thresholds.")
        

if ss.running:
    time.sleep(delay)
    st.rerun()
