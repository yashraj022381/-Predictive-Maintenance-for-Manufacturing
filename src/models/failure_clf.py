import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.models.anomaly import load_splits
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y, p) -> float:
    """Alarm line that gives the best balance of precision and recall."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1[:-1])])


def evaluate(name: str, df: pd.DataFrame, p: np.ndarray, thr: float) -> dict:
    y = df["failure"].to_numpy()
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "caught": int(tp), "missed": int(fn), "false_alarms": int(fp),
    }
    print(f"  {name:22s} PR-AUC {m['pr_auc']:.3f} | ROC_AUC {m['roc_auc']:.3f} | "
          f"P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f} | "
          f"caught {tp}, missed {fn}, false alarms {fp}")
    return m


def warning_report(df: pd.DataFrame, p: np.ndarray, thr: float) -> None:
    """Business view: did each engine get a warning inside its danger zone,
    and how many cycles before failure?"""
    d = df.assign(pred=(p >= thr).astype(int))
    danger = d[(d["failure"] == 1) & (d["pred"] == 1)]
    lead = danger.groupby("unit")["rul"].max()
    n_units = d["unit"].nunique()
    print(f" Engines warned inside danger zone: {len(lead)} of {n_units}"
          f" | median notice: {lead.median():.0f} cycles" if len(lead) else
          f" Engines warned inside danger zone: 0 of {n_units}")
    

def run(config_path: str = "configs/config.yaml") -> None:
    cfg, train, val, test, cols = load_splits(config_path)
    art = Path(cfg["paths"]["artifacts_dir"])
    seed = cfg["random_seed"]

    y_tr = train["failure"]
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()  # how much to boost failures
    print(f"Failure rows in training: {y_tr.mean():.1%} -> weight on failures: {spw:.1f}x")
    print(f"Lazy 'always fine' baseline on val: accuracy "
          f"{1 - val['failure'].mean():.1%}, failures caught: 0\n")

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
            early_stopping_rounds=30, n_jobs=-1, random_state=seed),
    }

    test_last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    thresholds, results = {}, {}

    for name, model in models.items():
        print(f"=== {name} ===")
        if name == "xgboost":
            model.fit(train[cols], y_tr, eval_set=[(val[cols], val["failure"])], verbose=False)
            print(f"  best number of trees: {model.best_iteration + 1}")
        else:
            model.fit(train[cols], y_tr)

        p_val = model.predict_proba(val[cols])[:, 1]
        thr = best_f1_threshold(val["failure"], p_val)   # tuned on VAL only
        thresholds[name] = thr
        print(f"  alarm threshold (tuned on validation): {thr:.3f}")

        results[name] = {
            "val": evaluate("validation (all rows)", val, p_val, thr),
            "test_all": evaluate("test (all rows)", test,
                                 model.predict_proba(test[cols])[:, 1], thr),
            "test_last": evaluate("test (last row/engine)", test_last,
                                  model.predict_proba(test_last[cols])[:, 1], thr),
        }
        print("  Business view, validation engines:")
        warning_report(val, p_val, thr)
        print()

    # Which features did XGBoost lean on most? (a preview of Phase 5: SHAP)
    imp = pd.Series(models["xgboost"].feature_importances_, index=cols)
    print("Top 10 features for XGBoost:")
    print(imp.sort_values(ascending=False).head(10).round(3).to_string())

    for name, model in models.items():
        joblib.dump(model, art / f"{name}_clf.joblib", compress=3)
    (art / "failure_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (art / "failure_metrics.json").write_text(json.dumps(results, indent=2))
    print("\nSaved models, thresholds and metrics to artifacts/")

if __name__ == "__main__":
    run()
