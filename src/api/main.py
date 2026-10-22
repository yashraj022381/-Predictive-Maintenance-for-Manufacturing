"""Phase 7: FastAPI backend for the predictive-maintenance system."""
import os
import json
import time
import threading
import pandas as pd

from collections import deque
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from src.api import db
from src.api.schemas import AnalysisResponse, Prediction, PredictRequest
from src.data.loader import load_config
from src.models.detector import Detector
from src.genai.agents import MaintenanceCrew


load_dotenv()
state: dict = {}
_crew_lock = threading.Lock()
_llm_calls: deque = deque()
MAX_AI_PER_HOUR = int(os.getenv("MAX_AI_PER_HOUR", "60"))



@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    state["art"] = Path(cfg["paths"]["artifacts_dir"])
    fleet = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "test.csv")
    state["fleet"] = fleet.sort_values(["unit", "cycle"]).reset_index(drop=True)
    state["detector"] = Detector()
    state["crew"] = None
    db.init_db()
    yield
    state.clear()

app = FastAPI(title="Predictive Maintenance API", version="1.0", lifespan=lifespan)


def machine_history(unit: int, cycle: int | None) -> pd.DataFrame:
    fleet = state["fleet"]
    h = fleet[fleet["unit"] == unit]
    if h.empty:
        raise HTTPException(404, f"Unknown machine {unit}")
    if cycle is not None:
        h = h[h["cycle"] <= cycle]
        if h.empty:
            raise HTTPException(404, f"Machine {unit} has no data at or before cycle {cycle}")
    return h


def get_crew():
    with _crew_lock:
        if state.get("crew") is None:
            if not os.getenv("GROQ_API_KEY"):
                raise HTTPException(503, "GROQ_API_KEY is not set, so /analyze is unavailable. "
                                         "/predict still works.")
            state["crew"] = MaintenanceCrew(detector=state["detector"])
        return state["crew"]


def _ai_budget_left() -> bool:
    now = time.time()
    while _llm_calls and now - _llm_calls[0] > 3600:
        _llm_calls.popleft()
    return len(_llm_calls) < MAX_AI_PER_HOUR


@app.get("/health")
def health():
    return {"status": "ok", "machines": int(state["fleet"]["unit"].nunique()),
            "llm_ready": bool(os.getenv("GROQ_API_KEY"))}



@app.get("/machines")
def machines():
    last = state["fleet"].groupby("unit")["cycle"].max()
    return [{"unit": int(u), "last_cycle": int(c)} for u, c in last.items()]


# Plain `def` (not `async def`): FastAPI runs these in worker threads,
# so slow model calls don't freeze the whole server.
@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest):
    det = state["detector"].analyze(machine_history(req.unit, req.cycle))
    db.log_prediction(det)
    return det




@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: PredictRequest):
    if not _ai_budget_left():
        raise HTTPException(429, "Hourly limit for AI reports reached. Try again later.")
    history = machine_history(req.unit, req.cycle)
    out = get_crew().run(history)
    if "explainer" in out["trace"]:
        _llm_calls.append(time.time())
    det = out["detection"]
    db.log_prediction(det, out["explanation"], out["actions"])
    return {**det, "explanation": out["explanation"], "actions": out["actions"],
            "agents_ran": out["trace"], "unknown_citations": out["unknown_citations"]}


@app.get("/history")
def history(unit: int | None = None,
            min_level: int = Query(1, ge=1, le=4),
            limit: int = Query(100, ge=1, le=1000)):
    return db.get_history(unit, min_level, limit)



@app.get("/alerts")
def alerts(unit: int | None = None, limit: int = Query(100, ge=1, le=1000)):
    return db.get_history(unit, min_level=2, limit=limit)



@app.get("/metrics")
def metrics():
    out = {}
    for key, name in [("failure", "failure_metrics.json"), ("rul", "rul_metrics.json"),
                      ("anomaly_thresholds", "anomaly_thresholds.json")]:
        path = state["art"] / name
        out[key] = json.loads(path.read_text()) if path.exists() else None
    return out
    
