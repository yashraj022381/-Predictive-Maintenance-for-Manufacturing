"""SQLite notebook of every prediction (history + alerts)."""
import os
import sqlite3

from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = Path(os.getenv("PM_DB_PATH", "artifacts/history.db"))


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    unit INTEGER NOT NULL, cycle INTEGER NOT NULL,
    risk REAL, rul REAL, level INTEGER, level_name TEXT,
    iforest_flag INTEGER, ae_flag INTEGER, n_drift INTEGER,
    explanation TEXT, actions TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_unit ON predictions(unit, cycle);
"""

@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True, )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)

def log_prediction(det: dict, explanation: str | None = None,
                   actions: str | None = None) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO predictions (ts, unit, cycle, risk, rul, level, level_name, "
            "iforest_flag, ae_flag, n_drift, explanation, actions) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             det["unit"], det["cycle"], det["risk"], det["rul"], det["level"],
             det["level_name"], int(det["iforest_flag"]), int(det["ae_flag"]),
             det["n_drift"], explanation, actions))
        return cur.lastrowid


def get_history(unit: int | None = None, min_level: int = 1, limit: int = 1000) -> list[dict]:
    query, args = "SELECT * FROM predictions WHERE level >= ?", [min_level]
    if unit is not None:
        query += " AND unit = ?"
        args.append(unit)
    query += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with connect() as c:
        return [dict(r) for r in c.execute(query, args).fetchall()]
