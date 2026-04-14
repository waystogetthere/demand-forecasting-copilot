import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.forecaster import FEATURE_COLS, FEATURE_LABELS

DB_PATH = "data/forecasting.db"


def init_db(db_path=DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS forecast_results (
            run_id TEXT,
            timestamp TEXT,
            category TEXT,
            store TEXT,
            horizon INTEGER,
            item_id TEXT,
            prior_avg_daily REAL,
            predicted_avg_daily REAL,
            pct_change REAL,
            total_predicted REAL
        );
        CREATE TABLE IF NOT EXISTS shap_results (
            run_id TEXT,
            item_id TEXT,
            feature TEXT,
            shap_value REAL,
            snap_lift REAL
        );
        CREATE TABLE IF NOT EXISTS model_metrics (
            run_id TEXT,
            category TEXT,
            store TEXT,
            mae REAL,
            top_features TEXT
        );
    """)
    conn.commit()
    conn.close()


def save_run(run_id, forecast_summary, shap_df, mae,
             importance, category, store, horizon, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    timestamp = datetime.now().isoformat()

    # forecast_results
    for item_id, stats in forecast_summary.items():
        conn.execute(
            "INSERT INTO forecast_results VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, timestamp, category, store, horizon, item_id,
             stats["prior_avg_daily"], stats["predicted_avg_daily"],
             stats["pct_change"], stats["total_predicted"])
        )

    # shap_results — melt to long format, map to business-readable feature names
    snap_lift = float(shap_df["snap"].mean())
    shap_long = shap_df.melt(
        id_vars=["item_id", "date"],
        value_vars=FEATURE_COLS,
        var_name="feature_raw",
        value_name="shap_value"
    )
    shap_long["feature"] = shap_long["feature_raw"].map(
        lambda x: FEATURE_LABELS.get(x, x)
    )
    shap_long["run_id"] = run_id
    shap_long["snap_lift"] = snap_lift
    shap_long[["run_id", "item_id", "feature", "shap_value", "snap_lift"]].to_sql(
        "shap_results", conn, if_exists="append", index=False
    )

    # model_metrics
    top5_labels = [FEATURE_LABELS.get(f, f) for f in importance.head(5).index]
    conn.execute(
        "INSERT INTO model_metrics VALUES (?,?,?,?,?)",
        (run_id, category, store, mae, json.dumps(top5_labels))
    )

    conn.commit()
    conn.close()


def get_forecast_summary(run_id, db_path=DB_PATH) -> list:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM forecast_results WHERE run_id = ?", (run_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_shap_drivers(run_id, item_id=None, db_path=DB_PATH) -> list:
    """Return top 3 features per item ranked by mean absolute SHAP."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if item_id:
        rows = conn.execute(
            "SELECT item_id, feature, AVG(shap_value) as shap_value "
            "FROM shap_results WHERE run_id = ? AND item_id = ? "
            "GROUP BY item_id, feature",
            (run_id, item_id)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT item_id, feature, AVG(shap_value) as shap_value "
            "FROM shap_results WHERE run_id = ? "
            "GROUP BY item_id, feature",
            (run_id,)
        ).fetchall()
    conn.close()

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["item_id"]].append(dict(r))

    result = []
    for iid, drivers in grouped.items():
        top3 = sorted(drivers, key=lambda x: abs(x["shap_value"]), reverse=True)[:3]
        result.extend(top3)
    return result


def get_snap_lift(run_id, db_path=DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT snap_lift FROM shap_results WHERE run_id = ? LIMIT 1", (run_id,)
    ).fetchone()
    conn.close()
    snap_lift = float(row[0]) if row else 0.0
    if snap_lift > 0.1:
        interpretation = (
            f"SNAP days drive +{snap_lift:.2f} units/day on average "
            f"(controlling for other factors)"
        )
    else:
        interpretation = "SNAP has minimal impact on demand in this dataset"
    return {"snap_lift": snap_lift, "interpretation": interpretation}


def get_model_metrics(run_id, db_path=DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM model_metrics WHERE run_id = ?", (run_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {}
    d = dict(row)
    d["top_features"] = json.loads(d["top_features"])
    return d


def get_recent_runs(limit=5, db_path=DB_PATH) -> list:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT fr.run_id, fr.timestamp, fr.category, fr.store, mm.mae
        FROM forecast_results fr
        LEFT JOIN model_metrics mm ON fr.run_id = mm.run_id
        GROUP BY fr.run_id
        ORDER BY fr.timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
