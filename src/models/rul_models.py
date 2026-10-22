import json
import joblib
import torch

import numpy as np
import pandas as pd
import torch.nn as nn

from pathlib import Path
from xgboost import XGBRegressor
from src.models.anomaly import load_splits

SEQ_LEN = 30    # the LSTM reads the last 30 cycles



def metrics(y, p, with_score=False) -> dict:
    d = p - y   # positive = predicted too LATE (dangerous)
    m = {"mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d ** 2).mean()))}
    if with_score:
        m["nasa_score"] = float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())
    return m


def show(name, m) -> None:
    extra = f" | NASA score {m['nasa_score']:.0f}" if "nasa_score" in m else ""
    print(f"  {name:26s} MAE {m['mae']:5.1f} | RMSE {m['rmse']:5.1f}{extra}")


def make_windows(df: pd.DataFrame, sensors: list[str], seq_len: int) -> np.ndarray:
    """For every row, collect that engine's last `seq_len` cycles.
    Early rows are padded by repeating the first reading."""
    out = []
    for _, g in df.groupby("unit", sort=False):        # df is sorted by unit, cycle
        arr = g[sensors].to_numpy(dtype=np.float32)
        full = np.vstack([np.repeat(arr[:1], seq_len - 1, axis=0), arr])
        win = np.lib.stride_tricks.sliding_window_view(full, seq_len, axis=0)
        out.append(win.transpose(0, 2, 1))             # (rows, seq_len, sensors)
    return np.ascontiguousarray(np.concatenate(out))


class RULNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)         # use the last time step

    
def predict_lstm(model, x: np.ndarray, cap: int) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            preds.append(model(torch.tensor(x[i:i + 1024])).numpy())
    return np.clip(np.concatenate(preds) * cap, 0, cap)


def run(config_path: str = "configs/config.yaml", epochs: int = 60, patience: int = 8) -> None:
    cfg, train, val, test, cols = load_splits(config_path)
    art = Path(cfg["paths"]["artifacts_dir"])
    seed, cap = cfg["random_seed"], cfg["data"]["rul_cap"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    sensors = json.loads((art / "feature_cols.json").read_text())["sensors"]

    # make sure every table is ordered by engine, then cycle
    train, val, test = (d.sort_values(["unit", "cycle"]).reset_index(drop=True)
                        for d in (train, val, test))
    last = test.groupby("unit").tail(1).index           # last row of each test engine
    y_tr, y_val, y_te = (d["rul_capped"].to_numpy(np.float32) for d in (train, val, test))

    results = {}

    # ---- 0) Dumb baseline: always guess the training average ----
    print("=== Dumb baseline (always guess the average) ===")
    base = np.full(len(test), y_tr.mean())
    results["baseline"] = metrics(y_te[last], base[last], True)
    show("test (last row/engine)", results["baseline"])

    # ---- 1) XGBoost regressor ----
    print("\n=== XGBoost regressor ===")
    xgb = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=4, subsample=0.8,
                       colsample_bytree=0.8, early_stopping_rounds=30,
                       n_jobs=-1, random_state=seed)
    xgb.fit(train[cols], y_tr, eval_set=[(val[cols], y_val)], verbose=False)
    pv = np.clip(xgb.predict(val[cols]), 0, cap)
    pt = np.clip(xgb.predict(test[cols]), 0, cap)
    results["xgb"] = {"val": metrics(y_val, pv), "test_all": metrics(y_te, pt),
                      "test_last": metrics(y_te[last], pt[last], True)}
    for k, label in [("val", "validation (all rows)"), ("test_all", "test (all rows)"),
                     ("test_last", "test (last row/engine)")]:
        show(label, results["xgb"][k])

    # ---- 2) LSTM ----
    print("\n=== LSTM (reads last 30 cycles) ===")
    x_tr, x_val, x_te = (make_windows(d, sensors, SEQ_LEN) for d in (train, val, test))
    xt, yt = torch.tensor(x_tr), torch.tensor(y_tr / cap)   # target scaled to 0..1

    model = RULNet(len(sensors))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best, best_state, wait = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        val_mae = np.abs(predict_lstm(model, x_val, cap) - y_val).mean()
        if val_mae < best - 1e-3:
            best, wait = val_mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch}")
                break
        if epoch % 5 == 0:
            print(f"  epoch {epoch:2d} | validation MAE {val_mae:.2f} cycles")

    model.load_state_dict(best_state)
    pv = predict_lstm(model, x_val, cap)
    pt = predict_lstm(model, x_te, cap)
    results["lstm"] = {"val": metrics(y_val, pv), "test_all": metrics(y_te, pt),
                       "test_last": metrics(y_te[last], pt[last], True)}
    for k, label in [("val", "validation (all rows)"), ("test_all", "test (all rows)"),
                     ("test_last", "test (last row/engine)")]:
        show(label, results["lstm"][k])

    # ---- Save ----
    torch.save(model.state_dict(), art / "rul_lstm.pt")
    joblib.dump(xgb, art / "rul_xgb.joblib", compress=3)
    (art / "rul_config.json").write_text(json.dumps(
        {"seq_len": SEQ_LEN, "rul_cap": cap, "sensors": sensors, "hidden": 64, "layers": 2}, indent=2))
    (art / "rul_metrics.json").write_text(json.dumps(results, indent=2))
    print("\nSaved models, config and metrics to artifacts/")


if __name__ == "__main__":
    run()
