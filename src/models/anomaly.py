"""Anomaly detection, part 1: shared helpers + Isolation Forest.
Trained on HEALTHY data only. Higher score = weirder."""

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from src.data.loader import load_config

HEALTHY_RUL = 100  # rows with at least this many cycles left count as "healthy"


def load_splits(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    proc, art = Path(cfg["paths"]["processed_dir"]), Path(cfg["paths"]["artifacts_dir"])
    cols = json.loads((art / "feature_cols.json").read_text())["features"]
    train, val, test = (pd.read_csv(proc / f"{n}.csv") for n in ("train", "val", "test"))
    return cfg, train, val, test, cols


def save_threshold(art_dir: Path, name: str, value: float) -> None:
    path = art_dir / "anomally_thresholds.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = float(value)
    path.write_text(json.dumps(data, indent=2))


def report(title: str, scores: np.ndarray, df: pd.DataFrame, threshold: float) -> None:
    y = df["failure"].to_numpy()
    flagged = scores > threshold
    print(f"\n-- {title} --")
    print(f"ROC-AUC: {roc_auc_score(y, scores):.3f}   "
          f"PR-AUC: {average_precision_score(y,  scores):.3f} (a random guesser gets {y.mean():.3f})")
    print(f"Alarm rate on healthy rows (false_alarms): {flagged[y == 0].mean():.1%}")
    print(f"Alarm rate on danger-zone rows (caught):   {flagged[y == 1].mean():.1%}")
    bucket = pd.cut(df["rul"], [-1, 30, 60, 100, 10_000],
                    labels = ["0-30", "31-60", "61-100", "100+"])
    print("Average weirdness score by cycles-left:")
    print(pd.Series(scores).groupby(bucket.to_numpy(), observed=True).mean().round(4).to_string())


def run(config_path: str = "configs/config.yaml") -> None:
    cfg, train, val, test, cols = load_splits(config_path)
    art = Path(cfg["paths"]["artifacts_dir"])

    healthy = train[train["rul"] >= HEALTHY_RUL]
    print(f"Training on {len(healthy)} healthy rows (of {len(train)})")

    model = IsolationForest(n_estimators=300, random_state=cfg["random_seed"], n_jobs=-1)
    model.fit(healthy[cols])

    def score(df):  # sklearn says "higher = more normal", so we flip the sign
        return -model.score_samples(df[cols])

    # Alarm line: the score that only 1% of healthy training rows exceed
    threshold = np.percentile(score(healthy), 99)

    report("Isolation Forest | validation", score(val), val, threshold)
    report("Isolation Forest | test", score(test), test, threshold)

    joblib.dump(model, art / "isolation_forest.joblib")
    save_threshold(art, "isolation_forest", threshold)
    print("\nSaved model + threshold to artifacts/")

if __name__ == "__main__":
    run()
