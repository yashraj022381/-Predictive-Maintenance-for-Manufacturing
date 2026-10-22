"""Load C-MAPSS data and create our two 'answer' columns:
   1. rul      -> how many cycles are left (a number)
   2. failure  -> 1 if the engine fails within N cycles, else 0 (yes/no)
"""

from pathlib import Path

import pandas as pd
import yaml

# The raw files have no header, so we name the columns ourselves.
INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _read_raw(path: Path) -> pd.DataFrame:
    # sep=r"\s+" means "columns are separated by any amount of spaces"
    return pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)


def add_rul_train(df:pd.DataFrame) -> pd.DataFrame:
    """Training engines run until death, so:
       RUL = (last cycle of that engine) - (current cycle)"""
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = last_cycle - df["cycle"]
    return df


def add_rul_test(df: pd.DataFrame, rul_at_end: pd.Series) -> pd.DataFrame:
    """Test engines stop early. NASA tells us how many cycles were left at the
    stopping point, so we count backwards from there."""
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    end_rul = df["unit"].map(rul_at_end)  # RUL at the last recorded cycle
    df["rul"] = end_rul + (last_cycle - df["cycle"])
    return df


def add_labels(df: pd.DataFrame, rul_cap: int, horizon: int) -> pd.DataFrame:
    df = df.copy()
    # Capped RUL: a brand-new engine with 300 cycles left looks the same as one
    # with 200 left (both are healthy), so we say "125 or more = 125".
    df["rul_capped"] = df["rul"].clip(upper=rul_cap)
    # Failure label: 1 = "danger zone", 0 = "fine for now"
    df["failure"] = (df["rul"] <= horizon).astype(int)
    return df

def load_datasets(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    raw = Path(cfg["paths"]["raw_dir"])
    name = cfg["data"]["dataset"]

    train = _read_raw(raw / f"train_{name}.txt")
    test = _read_raw(raw / f"test_{name}.txt")

    # RUL file: one number per test engine, in engine order (1, 2, 3...)
    rul_end = pd.read_csv(raw / f"RUL_{name}.txt", header=None)[0]
    rul_end.index = range(1, len(rul_end) + 1)

    train = add_rul_train(train)
    test = add_rul_test(test, rul_end)

    cap, horizon = cfg["data"]["rul_cap"], cfg["data"]["failure_horizon"]
    return add_labels(train, cap, horizon), add_labels(test, cap, horizon)

if __name__ == "__main__":
    train, test = load_datasets()
    print("Train shape: ", train.shape, "| engines: ", train["unit"].nunique())
    print("Test shape: ", test.shape, "| engines: ", test["unit"].nunique())
    print(train.head())
