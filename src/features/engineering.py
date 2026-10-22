"""Turn raw sensor rows into 'clue' features and split data safely."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from src.data.loader import SENSOR_COLS, load_config, load_datasets


def find_constant_sensors(train: pd.DataFrame) -> list[str]:
    """Sensors that never change carry no information."""
    stds = train[SENSOR_COLS].std()
    return stds[stds < 1e-6].index.tolist()


def build_feature_table(df: pd.DataFrame, sensors: list[str], scaler: MinMaxScaler, window: int) -> pd.DataFrame:
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    df[sensors] = scaler.transform(df[sensors])  # scale to 0..1

    # Group by engine so one engine's history never mixes with another's.
    g = df.groupby("unit")[sensors]

    mean = (g.rolling(window, min_periods=1).mean()
            .reset_index(level=0, drop=True).add_suffix(f"_mean{window}"))
    
    std = (g.rolling(window, min_periods=2).std()
           .reset_index(level=0, drop=True).add_suffix(f"_std{window}")
           .fillna(0))                     # first cycle has no std yet
    
    diff = (g.diff(window).fillna(0)       # first cycles have no "past"
            .add_suffix(f"_diff{window}"))

    return pd.concat([df, mean, std, diff], axis=1)


def run(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    seed = cfg["random_seed"]
    window = cfg["features"]["window"]
    out_dir = Path(cfg["paths"]["processed_dir"])
    art_dir = Path(cfg["paths"]["artifacts_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_datasets(config_path)

    # 1) Split by ENGINE so no engine appears in both train and validation.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=cfg["features"]["val_fraction"], random_state=seed)
    tr_idx, val_idx = next(splitter.split(train_raw, groups=train_raw["unit"]))
    tr, val = train_raw.iloc[tr_idx], train_raw.iloc[val_idx]

    # 2) Pick sensors and fit the scaler on TRAIN engines only.
    constant = find_constant_sensors(train_raw)
    sensors = [s for s in SENSOR_COLS if s not in constant]
    scaler = MinMaxScaler().fit(tr[sensors])

    # 3) Build features for each split.
    tr_f = build_feature_table(tr, sensors, scaler, window)
    val_f = build_feature_table(val, sensors, scaler, window)
    test_f = build_feature_table(test_raw, sensors, scaler, window)

    feature_cols = sensors + [
        f"{s}_{kind}{window}" for kind in ("mean", "std", "diff") for s in sensors
    ]

    # 4) Save everything the later phases need.
    tr_f.to_csv(out_dir / "train.csv", index=False)
    val_f.to_csv(out_dir / "val.csv", index=False)
    test_f.to_csv(out_dir / "test.csv", index=False)
    joblib.dump(scaler, art_dir / "scaler.joblib")
    (art_dir / "feature_cols.json").write_text(
        json.dumps({"sensors": sensors, "features": feature_cols}, indent=2))

    # 5) Sanity checks, so we never trust blindly.
    print(f"Dropped constant columns: {constant}")
    print(f"Sensors kept: {len(sensors)} | total features: {len(feature_cols)}")
    for name, d in [("train", tr_f), ("val", val_f), ("test", test_f)]:
        print(f"{name:5s}: {d.shape[0]:6d} rows, {d['unit'].nunique():3d} engines, "
              f"failure rate {d['failure'].mean():.3f}, "
              f"NaNs in features: {int(d[feature_cols].isna().sum().sum())}")
    overlap = set(tr_f["unit"]) & set(val_f["unit"])
    print(f"Engines in BOTH train and val (must be 0): {len(overlap)}")


if __name__ == "__main__":
    run()
            
