
"""Lightweight risk explainer: SHAP-style contributions straight from XGBoost
(no 'shap' package, no matplotlib). Same interface as RiskExplainer."""
import json
import joblib

import numpy as np
import pandas as pd
import xgboost as xgb

from pathlib import Path
from src.data.loader import load_config
from src.explain.names import pretty


class RiskExplainerLite:
    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = load_config(config_path)
        art = Path(cfg["paths"]["artifacts_dir"])
        self.cols = json.loads((art / "feature_cols.json").read_text())["features"]
        self.model = joblib.load(art / "xgboost_clf.joblib")
        self.threshold = json.loads((art / "failure_thresholds.json").read_text())["xgboost"]
        self.booster = self.model.get_booster()
        try:   # use only the trees the classifier itself uses (early stopping)
            self.iter_range = (0, int(self.model.best_iteration) + 1)
        except AttributeError:
            self.iter_range = (0, 0)

    def contributions(self, x: pd.DataFrame) -> np.ndarray:
        """One number per feature plus a final 'bias' number (log-odds units)."""
        return self.booster.predict(xgb.DMatrix(x), pred_contribs=True,
                                    iteration_range=self.iter_range)[0]

    def explain(self, row: pd.DataFrame, top_k: int = 5) -> dict:
        x = row[self.cols]
        sv = self.contributions(x)[:-1]
        prob = float(self.model.predict_proba(x)[0, 1])
        top = np.argsort(-np.abs(sv))[:top_k]
        return {
            "risk_probability": prob,
            "alarm": prob >= self.threshold,
            "threshold": self.threshold,
            "drivers": [{
                "feature": self.cols[i],
                "label": pretty(self.cols[i]),
                "value": float(x.iloc[0, i]),
                "shap": float(sv[i]),
                "effect": "raises risk" if sv[i] > 0 else "lowers risk",
            } for i in top],
        }


if __name__ == "__main__":
    cfg = load_config()
    test = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "test.csv")
    ex = RiskExplainerLite()
    last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    row = last[last["unit"] == 35].iloc[[0]]
    c = ex.contributions(row[ex.cols])
    p_from_parts = 1 / (1 + np.exp(-c.sum()))
    p_model = ex.model.predict_proba(row[ex.cols])[0, 1]
    print(f"Additivity check (should be ~0): {abs(p_from_parts - p_model):.6f}")
    r = ex.explain(row)
    print(f"Unit 35 risk {r['risk_probability']:.1%}")
    for d in r["drivers"]:
        print(f"  {d['label']:42s} {d['shap']:+.2f}  ({d['effect']})")
