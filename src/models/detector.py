"""Phase 6b: the Detector. Plain code, NO LLM.
Runs every model on one machine's history and decides urgency by fixed rules."""
import json
import joblib
import torch
import pandas as pd

from pathlib import Path
from src.data.loader import load_config
from src.explain.shap_explainer import RiskExplainer
from src.models.autoencoder import Autoencoder, reconstruction_error
from src.models.rul_models import SEQ_LEN, RULNet, make_windows, predict_lstm

LEVELS = {1: "WATCH", 2: "PLAN", 3: "URGENT", 4: "STOP"}

WARMUP_CYCLES = 10   # rolling features need history; ignore anomaly flags before this

def decide_urgency(risk: float, alarm: bool, rul: float, n_drift: int):
    """Maintenance policy POL-01 + POL-03 written as code."""
    reasons = []
    if rul < 10:
        level = 4
        reasons.append(f"estimated cycles left {rul:.0f} is under 10 [POL-01]")

    elif risk > 0.95 and n_drift >= 3:
        level = 4
        #reasons.append(f"fewer than 10 cycles left or risk above 95% with {n_drift} sensors "
        #               f" worsening [POL-01]")
        reasons.append(f"risk {risk:.1%} is above 95% with {n_drift} sensors worsening [POL-01]")
                
    elif rul <= 30:
        level = 3
        reasons.append(f"estimated cycles left {rul:.0f} is between 10 and 30 [POL-01]")

    elif alarm or rul <= 60:
        level = 2
        reasons.append("risk at/above the alarm line or estimated cycles left between 30 and 60 "
                       "[POL-01]")

    else:
        level = 1
        reasons.append("risk below alarm line more than 60 cycles left [POL-01]")

    if n_drift >= 3 and 2 <= level < 4:
        level += 1
        reasons.append(f"{n_drift} sensors drifting together, urgency raised one level [POL-03]")
    return level, reasons


class Detector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = load_config(config_path)
        art = Path(cfg["paths"]["artifacts_dir"])
        meta = json.loads((art / "feature_cols.json").read_text())
        self.cols, self.sensors = meta["features"], meta["sensors"]

        self.risk = RiskExplainer(config_path)
        self.iforest = joblib.load(art / "isolation_forest.joblib")
        self.ae_cols = json.loads((art / "ae_cols.json").read_text())
        self.ae_scaler = joblib.load(art / "ae_scaler.joblib")
        self.ae = Autoencoder(len(self.ae_cols), bottleneck=4)
        self.ae.load_state_dict(torch.load(art / "autoencoder.pt", map_location="cpu"))
        self.thr = json.loads((art / "anomally_thresholds.json").read_text())

        
        rc = json.loads((art / "rul_config.json").read_text())
        self.cap = rc["rul_cap"]
        self.lstm = RULNet(len(rc["sensors"]), rc["hidden"], rc["layers"])
        self.lstm.load_state_dict(torch.load(art / "rul_lstm.pt", map_location="cpu"))

    def analyze(self, history: pd.DataFrame) -> dict:
        """`history` = all feature rows of ONE machine so far (from the processed CSVs)."""
        history = history.sort_values("cycle").reset_index(drop=True)
        row = history.iloc[[-1]]

        r = self.risk.explain(row, top_k = 5)
        rul = float(predict_lstm(self.lstm, make_windows(history, self.sensors, SEQ_LEN)[-1:],
                                 self.cap)[0])
        if_score = float(-self.iforest.score_samples(row[self.cols])[0])
        ae_score = float(reconstruction_error(
            self.ae, self.ae_scaler.transform(row[self.ae_cols]))[0])
        
        warm = int(row["cycle"].iloc[0]) < WARMUP_CYCLES
        # sensors among the top-5 risk drivers that push risk UP
        n_drift = len({d["feature"].partition("_")[0] for d in r["drivers"] if d["shap"] > 0})
        level, reasons = decide_urgency(r["risk_probability"], r["alarm"], rul, n_drift)

        rul_text = (f"{self.cap}+ cycles (healthy range)" if rul >= self.cap - 1
                    else f"about {rul:.0f} cycles (typical error about 10 cycles)")

        return {
            "unit": int(row["unit"].iloc[0]), "cycle": int(row["cycle"].iloc[0]),
            "risk": r["risk_probability"], "threshold": r["threshold"], "alarm": r["alarm"],
            "rul": rul, "rul_text": rul_text,
            "iforest_score": if_score, "iforest_limit": self.thr["isolation_forest"],
            #"iforest_flag": if_score > self.thr["isolation_forest"],
            "iforest_flag": bool(if_score > self.thr["isolation_forest"]) and not warm,
            "ae_score": ae_score, "ae_limit": self.thr["autoencoder"],
            #"ae_flag": ae_score > self.thr["autoencoder"],
            "ae_flag": bool(ae_score > self.thr["autoencoder"]) and not warm,
            "n_drift": n_drift, "level": level, "level_name": LEVELS[level],
            "reasons": reasons, "drivers": r["drivers"],
        }


def pick_engines(det: Detector, test: pd.DataFrame) -> dict:
    last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
    p = det.risk.model.predict_proba(last[det.cols])[:, 1]
    pick = lambda i: int(last.loc[int(i), "unit"])
    return {"HIGHEST": pick(p.argmax()),
            "BORDERLINE": pick(abs(p - det.risk.threshold).argmin()),
            "LOWEST": pick(p.argmin())}


if __name__ == "__main__":
    cfg = load_config()
    test = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "test.csv")
    det = Detector()
    for tag, unit in pick_engines(det, test).items():
        d = det.analyze(test[test["unit"] == unit])
        true_rul = int(test[test["unit"] == unit].sort_values("cycle")["rul"].iloc[-1])
        print(f"\n=== {tag}: unit {d['unit']} at cycle {d['cycle']} (true cycles left {true_rul}) ===")
        print(f"Risk {d['risk']:.1%} (alarm line {d['threshold']:.0%}, alarm {d['alarm']}) | "
              f"estimated cycles left: {d['rul_text']}")
        print(f"Isolation Forest {d['iforest_score']:.3f} (limit {d['iforest_limit']:.3f}) "
              f"flag={d['iforest_flag']} | Autoencoder {d['ae_score']:.3f} "
              f"(limit {d['ae_limit']:.3f}) flag={d['ae_flag']}")
        print(f"Drifting sensors: {d['n_drift']} | URGENCY: Level {d['level']} {d['level_name']}")
        for why in d["reasons"]:
            print("   -", why)
