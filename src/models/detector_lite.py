"""Lightweight Detector for small servers: no PyTorch.
Risk + drivers: XGBoost. Anomaly check: Isolation Forest only. Remaining life: XGBoost.
Returns the same fields as the full Detector."""
import json
import joblib

import numpy as np
import pandas as pd

from pathlib import Path
from src.data.loader import load_config
from src.explain.risk_lite import RiskExplainerLite
from src.models.urgency import LEVELS, WARMUP_CYCLES, decide_urgency


class DetectorLite:
    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = load_config(config_path)
        art = Path(cfg["paths"]["artifacts_dir"])
        meta = json.loads((art / "feature_cols.json").read_text())
        self.cols, self.sensors = meta["features"], meta["sensors"]
        self.risk = RiskExplainerLite(config_path)
        self.iforest = joblib.load(art / "isolation_forest.joblib")
        self.thr = json.loads((art / "anomally_thresholds.json").read_text())
        self.rul_model = joblib.load(art / "rul_xgb.joblib")
        self.cap = json.loads((art / "rul_config.json").read_text())["rul_cap"]

    def analyze(self, history: pd.DataFrame) -> dict:
        history = history.sort_values("cycle").reset_index(drop=True)
        row = history.iloc[[-1]]

        r = self.risk.explain(row, top_k=5)
        rul = float(np.clip(self.rul_model.predict(row[self.cols])[0], 0, self.cap))
        if_score = float(-self.iforest.score_samples(row[self.cols])[0])
        warm = int(row["cycle"].iloc[0]) < WARMUP_CYCLES
        if_flag = bool(if_score > self.thr["isolation_forest"]) and not warm

        n_drift = len({d["feature"].partition("_")[0] for d in r["drivers"] if d["shap"] > 0})
        level, reasons = decide_urgency(r["risk_probability"], r["alarm"], rul, n_drift)

        rul_text = (f"{self.cap}+ cycles (healthy range)" if rul >= self.cap - 1
                    else f"about {rul:.0f} cycles (typical error about 12 cycles)")
        return {
            "unit": int(row["unit"].iloc[0]), "cycle": int(row["cycle"].iloc[0]),
            "risk": r["risk_probability"], "threshold": r["threshold"], "alarm": r["alarm"],
            "rul": rul, "rul_text": rul_text,
            "iforest_score": if_score, "iforest_limit": self.thr["isolation_forest"],
            "iforest_flag": if_flag,
            "ae_score": None, "ae_limit": None, "ae_flag": False,   # autoencoder not in lite mode
            "n_drift": n_drift, "level": level, "level_name": LEVELS[level],
            "reasons": reasons, "drivers": r["drivers"], "lite": True,
        }
