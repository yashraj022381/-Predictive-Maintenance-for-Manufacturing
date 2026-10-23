"""Standalone Streamlit app: the models run inside this process (no separate API).
Lightweight mode: XGBoost + Isolation Forest, no PyTorch."""
import os
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from src.models.detector_lite import DetectorLite

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                      # the project uses paths relative to the project root
sys.path.insert(0, str(ROOT))
os.environ["PM_LITE"] = "1"
os.environ.setdefault("PM_DB_PATH", os.path.join(tempfile.gettempdir(), "pm_history.db"))

import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
ICON = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}


def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return None


if not os.getenv("GROQ_API_KEY") and _secret("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = str(_secret("GROQ_API_KEY"))
MAX_AI_PER_HOUR = int(os.getenv("MAX_AI_PER_HOUR") or _secret("MAX_AI_PER_HOUR") or 30)


# ---------- loaded once, shared by every visitor ----------
@st.cache_resource(show_spinner="Loading models (first visit only)...")
def load_detector():
    #from src.models.detector_lite import DetectorLite
    return DetectorLite()


@st.cache_resource(show_spinner="Loading machine data...")
def load_fleet():
    from src.data.loader import load_config
    cfg = load_config()
    fleet = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "test.csv")
    return fleet.sort_values(["unit", "cycle"]).reset_index(drop=True)


@st.cache_resource(show_spinner="Starting the AI agents...")
def load_crew():
    from src.genai.agents import MaintenanceCrew
    return MaintenanceCrew(detector=load_detector())


@st.cache_resource
def ai_calls():
    return deque()


@st.cache_resource
def history_db():
    from src.api import db
    db.init_db()
    return db


def ai_budget_left() -> bool:
    q, now = ai_calls(), time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    return len(q) < MAX_AI_PER_HOUR


detector, fleet = load_detector(), load_fleet()
last = fleet.groupby("unit")["cycle"].max().to_dict()


def predict(unit: int, cycle: int) -> dict:
    h = fleet[(fleet["unit"] == unit) & (fleet["cycle"] <= cycle)]
    return detector.analyze(h)


st.title("🏭 Predictive Maintenance Dashboard")
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
    st.caption("Each tick replays the machine at a later cycle and runs the models on it. "
               "Try machine 35 for a full green-to-red story.")
    st.caption("Live demo = lightweight mode (XGBoost remaining-life model, Isolation Forest only). "
               "The full version with LSTM and autoencoder runs through Docker; see the README.")
if unit != ss.unit:
    reset(unit)

# ---- one simulation tick ----
if ss.running:
    cyc = min(ss.cycle, last[unit])
    pred = predict(unit, cyc)
    ss.trace.append(pred)
    history_db().log_prediction(pred)
    if cyc >= last[unit]:
        ss.running = False
    else:
        ss.cycle += step

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
        m[3].metric("Autoencoder", "off (lite)")

        df = pd.DataFrame(ss.trace).set_index("cycle")
        df["alarm line"] = cur["threshold"]
        left, right = st.columns(2)
        left.subheader("Failure risk")
        left.line_chart(df[["risk", "alarm line"]])
        right.subheader("Estimated cycles left")
        right.line_chart(df[["rul"]])

        st.subheader("Why? Biggest pushes on risk right now (SHAP-style)")
        drv = pd.DataFrame(cur["drivers"]).set_index("label")["shap"]
        st.bar_chart(drv)
        st.caption("Above zero = pushes risk up. Below zero = pushes risk down.")

        if st.button("🤖 Explain & recommend (uses the AI agents)"):
            ss.running = False
            if not os.getenv("GROQ_API_KEY"):
                st.error("No GROQ_API_KEY found. Add it in the app's Secrets.")
            elif not ai_budget_left():
                st.warning("The hourly limit of AI reports has been reached. Try again later.")
            else:
                with st.spinner("Detector → Explainer → Recommender..."):
                    try:
                        crew = load_crew()
                        hist_rows = fleet[(fleet["unit"] == unit) & (fleet["cycle"] <= cur["cycle"])]
                        out = crew.run(hist_rows)
                        if "explainer" in out["trace"]:      # only real LLM calls count
                            ai_calls().append(time.time())
                        det = out["detection"]
                        history_db().log_prediction(det, out["explanation"], out["actions"])
                        ss.analysis = {"cycle": det["cycle"], "agents_ran": out["trace"],
                                       "explanation": out["explanation"], "actions": out["actions"],
                                       "unknown_citations": out["unknown_citations"]}
                    except Exception as e:
                        st.error(f"AI report failed: {e}")
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
    rows = history_db().get_history(None, level, 300)
    if rows:
        d = pd.DataFrame(rows)[["ts", "unit", "cycle", "risk", "rul", "level_name",
                                "iforest_flag", "n_drift"]]
        st.dataframe(d, use_container_width=True, hide_index=True)
    else:
        st.info("Nothing logged at this level yet. Run a simulation first.")
    st.caption("This log is shared by all visitors and resets when the app restarts.")

with perf:
    art = ROOT / "artifacts"

    def flatten(d: dict) -> pd.DataFrame:
        rows = []
        for model, v in d.items():
            if all(isinstance(x, dict) for x in v.values()):
                rows += [{"model": model, "data": split, **mm} for split, mm in v.items()]
            else:
                rows.append({"model": model, "data": "test_last", **v})
        return pd.DataFrame(rows).round(3)

    for title, fname in [("Failure classifier (danger within 30 cycles)", "failure_metrics.json"),
                         ("Remaining-useful-life models (errors in cycles)", "rul_metrics.json")]:
        st.subheader(title)
        p = art / fname
        if p.exists():
            st.dataframe(flatten(json.loads(p.read_text())), use_container_width=True,
                         hide_index=True)
    st.caption("test_all = every test row. test_last = last row of each engine (NASA style). "
               "The live demo uses the XGBoost remaining-life model; LSTM numbers come from the "
               "full version. Validation numbers are optimistic because validation tuned the "
               "thresholds.")

if ss.running:
    time.sleep(delay)
    st.rerun()
