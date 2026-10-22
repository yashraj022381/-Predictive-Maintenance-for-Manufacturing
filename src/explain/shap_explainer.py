import json
import joblib
import matplotlib
matplotlib.use("Agg")                                    # draw to files, no pop-up windows
import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from src.data.loader import load_config
from src.explain.names import pretty
from src.models.anomaly import load_splits



class RiskExplainer:
    """Wraps the XGBoost classifier. Used here AND later by the API / GenAI layer."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = load_config(config_path)
        art = Path(cfg["paths"]["artifacts_dir"])
        self.cols = json.loads((art / "feature_cols.json").read_text())["features"]
        self.model = joblib.load(art / "xgboost_clf.joblib")
        self.threshold = json.loads((art / "failure_thresholds.json").read_text())["xgboost"]
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, row: pd.DataFrame, top_k: int = 5) -> dict:
        """Explain ONE machine-state (a one-row DataFrame)."""
        x = row[self.cols]
        sv = self.explainer.shap_values(x)[0]
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
                "effect": "raise risk" if sv[i] > 0 else "lowers risk",
            } for i in top],
        }

def plot_local(result: dict, title: str, path: Path) -> None:
    d = result["drivers"][::-1]
    colors = ["#d62728" if x["shap"] > 0 else "#1f77b4" for x in d]
    plt.figure(figsize=(8, 4))
    plt.barh([x["label"] for x in d], [x["shap"] for x in d], color=colors)
    plt.axvline(0, color="black", lw="0.8")
    plt.xlabel("Push on risk  (red = raises, blue = lowers)")
    plt.title(f"{title} | risk {result['risk_probability']:.0%}")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def run(config_path: str = "configs/config.yaml") -> None:
    cfg, train, val, test, cols = load_splits(config_path)
    art = Path(cfg["paths"]["artifacts_dir"])
    ex = RiskExplainer(config_path)
    labels = [pretty(c) for c in cols]

    # ---- 1) GLOBAL: which sensors matter overall? ----
    sample = val.sample(min(1000, len(val)), random_state=cfg["random_seed"])
    sv = ex.explainer.shap_values(sample[cols])

    top = pd.Series(np.abs(sv).mean(axis=0), index=labels).sort_values(ascending=False)
    print("Top 10 features overall (average push on risk):")
    print(top.head(10).round(3).to_string())

    for kind, fname in [("dot", "shap_beeswarm.png"), ("bar", "shap_importance.png")]:
        plt.figure()
        shap.summary_plot(sv, sample[cols], feature_names=labels,
                          plot_type=kind, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(art / fname, dpi=120, bbox_inches="tight")
        plt.close()

    # ---- 2) TRUST CHECK: do SHAP pieces add up to the model's answer? ----
    chk = sample.head(5)
    s5 = ex.explainer.shap_values(chk[cols])
    base = float(np.ravel(ex.explainer.expected_value)[0])
    from_shap = 1 / (1 + np.exp(-(base + s5.sum(axis=1))))
    from_model = ex.model.predict_proba(chk[cols])[:, 1]
    print(f"\nAdditivity check (should be ~0): max difference = "
          f"{np.abs(from_shap - from_model).max():.6f}")

    # ---- 3) LOCAL: the riskiest and the safest test engine ----
    last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
    probs = ex.model.predict_proba(last[cols])[:, 1]
    for tag, i in [("HIGHEST", probs.argmax()), ("LOWEST", probs.argmin())]:
        r = ex.explain(last.iloc[[i]])
        row = last.iloc[i]
        print(f"\n=== {tag} risk engine: unit {int(row['unit'])} at cycle {int(row['cycle'])} "
              f"(true cycles left: {int(row['rul'])}) ===")
        print(f"Risk {r['risk_probability']:.1%} | alarm line {r['threshold']:.2f} | "
              f"ALARM: {r['alarm']}")
        for d in r["drivers"]:
            print(f"  {d['label']:42s} {d['shap']:+.2f}  ({d['effect']})")
        if tag == "HIGHEST":
            plot_local(r, f"Engine {int(row['unit'])}", art / "shap_local_example.png")

    print("\nSaved plots to artifacts/: shap_beeswarm.png, shap_importance.png, shap_local_example.png")


if __name__ == "__main__":
    run()
                        
        
