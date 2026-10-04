"""Monitoring flow: read recent predictions from PostgreSQL and check for input drift.

The detector's threshold is set so that about 4% of normal digits are flagged. If the live
unusual-input rate climbs far above that, the inputs no longer look like the training data.
Run once with ``python run_flow.py --config configs/monitor_flow_config.yaml`` or on a schedule
with ``python -m flows.serve_monitor`` (the ``monitor`` profile in docker-compose.yml).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import mlflow
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE
from sqlalchemy import select

from dimred_mlops import tracking
from dimred_mlops.db import Prediction, init_db, make_engine
from dimred_mlops.monitoring import summarize_predictions


@task(name="fetch-predictions", cache_policy=NONE)
def fetch_predictions(url: str, window_hours: float, exclude_prefixes=()) -> list[dict]:
    """Only the columns the checks need (not the 784 stored pixels) of real traffic.

    Requests whose ``source`` starts with one of ``exclude_prefixes`` (the UI's demo samples,
    CI smoke tests) are left out, so demos of corrupted inputs do not raise drift alerts.
    """
    engine = make_engine(url)
    init_db(engine)
    since = datetime.now(UTC) - timedelta(hours=window_hours)
    query = select(
        Prediction.digit_raw,
        Prediction.digit_pca,
        Prediction.is_unusual,
        Prediction.reconstruction_error,
        Prediction.source,
    ).where(Prediction.created_at >= since)
    for prefix in exclude_prefixes:
        query = query.where(~Prediction.source.startswith(prefix))
    with engine.connect() as conn:
        out = [dict(r._mapping) for r in conn.execute(query)]
    engine.dispose()
    return out


@flow(name="monitor-flow", log_prints=True)
def monitor_flow(config: dict) -> dict:
    logger = get_run_logger()
    mcfg = config["monitor"]
    rows = fetch_predictions(
        config["database"]["url"],
        float(mcfg["window_hours"]),
        tuple(mcfg.get("exclude_source_prefixes", [])),
    )
    report = summarize_predictions(rows, mcfg)
    if report["status"] == "insufficient-data":
        logger.info("Only %d predictions in the window - not enough for a verdict.", len(rows))
        return report

    tracking.configure(
        config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"] + "-monitoring"
    )
    with mlflow.start_run(run_name=f"monitor-{datetime.now(UTC):%Y%m%d-%H%M}"):
        mlflow.log_metrics(
            {
                "n_predictions": report["n_predictions"],
                "unusual_rate_pct": report["unusual_rate_pct"],
                "models_agree_pct": report["models_agree_pct"],
                "top_digit_share_pct": report["top_digit_share_pct"],
                "median_reconstruction_error": report["median_reconstruction_error"],
            }
        )
        tracking.log_json(report, "monitor_report.json")
    for a in report["alerts"]:
        logger.warning("ALERT: %s", a)
    logger.info(
        "Monitoring status: %s (%d predictions, unusual-input rate %.1f%%)",
        report["status"],
        report["n_predictions"],
        report["unusual_rate_pct"],
    )
    return report


@flow(name="scheduled-monitor")
def scheduled_monitor(config_path: str = "configs/monitor_flow_config.yaml") -> dict:
    """Entry point for the schedule: loads the config at run time (no secrets in Prefect)."""
    from dimred_mlops.config import load_config

    return monitor_flow(load_config(config_path))


def start(config: dict) -> dict:
    return monitor_flow(config)
