"""Train flow: Parts A-F -> MLflow (params, metrics, tables, figures, models) -> Model Registry.

Every part runs as a Prefect task and logs to its own nested MLflow run under one parent run.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import mlflow
import numpy as np
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE

from dimred_mlops import reporting, tracking
from dimred_mlops.classification import run_part_e
from dimred_mlops.compression import run_part_a
from dimred_mlops.data import MnistSplit, load_mnist_split
from dimred_mlops.detector import error_rate_by_flag, run_part_f
from dimred_mlops.manifold import run_part_c
from dimred_mlops.maps import run_part_d
from dimred_mlops.scaling import run_part_b

ROOT = Path(__file__).resolve().parents[1]


def load_split(cfg: dict) -> MnistSplit:
    data = cfg["data"]
    return load_mnist_split(
        data.get("cache_dir", "data"), data.get("n_train"), data.get("n_test"), int(cfg["seed"])
    )


def check_reports_writable(cfg: dict) -> None:
    """Fail in seconds, not after 15 minutes of training, if the report folder is read-only
    (typically a Docker bind mount owned by another user: run ``make up`` on Linux)."""
    for key in ("dir", "figures_dir"):
        d = ROOT / cfg["reports"][key]
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write-test"
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            raise PermissionError(
                f"Cannot write to {d} ({exc}). On Linux start the stack with `make up` "
                "(it passes your user id) or set HOST_UID/HOST_GID in .env."
            ) from exc


def data_fingerprint(split: MnistSplit) -> str:
    h = hashlib.md5()
    for a in (split.X_train, split.y_train, split.X_test, split.y_test):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return os.getenv("GIT_COMMIT", "unknown")


@task(name="load-mnist", cache_policy=NONE, retries=2, retry_delay_seconds=10)
def load_data(cfg: dict) -> MnistSplit:
    split = load_split(cfg)
    get_run_logger().info("MNIST split: %s", split.sizes)
    return split


def _log_tables(tables: dict) -> None:
    for name, df in tables.items():
        if df is not None:
            tracking.log_table(df, f"tables/{name}.csv")


def _nested(name: str, parent_run_id: str):
    return mlflow.start_run(run_name=name, nested=True, parent_run_id=parent_run_id)


@task(name="A-compression", cache_policy=NONE)
def part_a(split, cfg, pid):
    with _nested("A-compression", pid):
        res = run_part_a(split, cfg["part_a"], int(cfg["seed"]))
        tracking.log_params(res.params)
        tracking.log_metrics(res.metrics)
        _log_tables({"variance_levels": res.variance_table, "storage": res.storage})
        res.run_id = mlflow.active_run().info.run_id
    get_run_logger().info(
        "A: %d components keep %.2f%% of the variance",
        res.metrics["n_components_95"],
        100 * res.metrics["train_explained_variance_95"],
    )
    return res


@task(name="B-scaling", cache_policy=NONE)
def part_b(split, cfg, pid):
    with _nested("B-scaling", pid):
        res = run_part_b(split, cfg["part_b"], int(cfg["seed"]))
        tracking.log_params(res.params)
        tracking.log_metrics(res.metrics)
        _log_tables({"solvers": res.solvers, "timing_curve": res.timing_curve})
        res.run_id = mlflow.active_run().info.run_id
    get_run_logger().info(
        "B: out-of-core IPCA peak %.0f MB vs full SVD %.0f MB",
        res.metrics["memmap_partial_fit_peak_MB"],
        res.metrics["full_svd_total_RAM_MB"],
    )
    return res


@task(name="C-kernel-pca-manifolds", cache_policy=NONE)
def part_c(cfg, pid):
    with _nested("C-kernel-pca-manifolds", pid):
        res = run_part_c(cfg["part_c"], int(cfg["seed"]))
        tracking.log_params(res.params)
        tracking.log_metrics(res.metrics)
        _log_tables(
            {"kpca_grid": res.grid, "kpca_preimage": res.preimage, "manifold_methods": res.methods}
        )
        res.run_id = mlflow.active_run().info.run_id
    get_run_logger().info(
        "C: kPCA %s gamma=%.4f -> held-out accuracy %.3f",
        res.params["best_kernel"],
        res.metrics["best_gamma"],
        res.metrics["test_accuracy_supervised_choice"],
    )
    return res


@task(name="D-mnist-maps", cache_policy=NONE)
def part_d(split, cfg, pid):
    with _nested("D-mnist-maps", pid):
        res = run_part_d(split, cfg["part_d"], int(cfg["seed"]))
        tracking.log_params(res.params)
        tracking.log_metrics(res.metrics)
        _log_tables({"maps": res.table})
        res.run_id = mlflow.active_run().info.run_id
    get_run_logger().info("D: best 2D map %s", res.params["best_method"])
    return res


@task(name="E-classification", cache_policy=NONE)
def part_e(split, cfg, pid):
    with _nested("E-classification", pid):
        res = run_part_e(split, cfg["part_e"], int(cfg["seed"]))
        tracking.log_params(res.params)
        tracking.log_metrics(res.metrics)
        _log_tables({"classifiers": res.table})
        res.run_id = mlflow.active_run().info.run_id
    get_run_logger().info(
        "E: served pair raw %.4f / PCA %.4f",
        res.metrics["served_raw_accuracy"],
        res.metrics["served_pca_accuracy"],
    )
    return res


@task(name="F-unusual-input-detector", cache_policy=NONE)
def part_f(split, cfg, pid):
    with _nested("F-unusual-input-detector", pid):
        res = run_part_f(split, cfg["part_f"], int(cfg["seed"]))
        tracking.log_params(res.params)
        tracking.log_metrics(res.metrics)
        _log_tables(
            {
                "corruptions": res.corruption_table,
                "threshold_sweep": res.threshold_sweep,
                "variance_sweep": res.variance_sweep,
            }
        )
        res.run_id = mlflow.active_run().info.run_id
    get_run_logger().info(
        "F: false alarms %.2f%%, ROC-AUC %.3f",
        res.metrics["clean_false_alarm_pct"],
        res.metrics["roc_auc"],
    )
    return res


@task(name="register-models", cache_policy=NONE)
def register_models(a, d, e, f, split: MnistSplit, lineage: dict) -> dict[str, str]:
    example = split.X_test[:2]
    entries = [
        ("compressor", a, a.pca_full, {"n_components_95": a.metrics["n_components_95"]}),
        ("classifier_raw", e, e.models["raw"], {"test_accuracy": e.metrics["served_raw_accuracy"]}),
        ("classifier_pca", e, e.models["pca"], {"test_accuracy": e.metrics["served_pca_accuracy"]}),
        (
            "detector",
            f,
            f.detector,
            {"threshold": f.detector.threshold_, "flag_rate": f.params["flag_rate"]},
        ),
        ("atlas", d, d.atlas, {"n_points": len(d.atlas.labels)}),
    ]
    versions = {}
    for key, res, model, extra in entries:
        with mlflow.start_run(run_id=res.run_id, nested=True):
            versions[key] = tracking.log_model(model, tracking.MODEL_NAMES[key], example, extra)
        # lineage: which data, code and config produced this version
        tracking.set_version_tags(tracking.MODEL_NAMES[key], versions[key], lineage)
    get_run_logger().info("Registered model versions: %s", versions)
    return versions


@task(name="write-reports", cache_policy=NONE)
def write_reports(a, b, c, d, e, f, split, cfg, extra, run_info) -> dict:
    figs = reporting.make_figures(a, b, c, d, e, f, split, ROOT / cfg["reports"]["figures_dir"])
    for p in figs.values():
        mlflow.log_artifact(str(p), "figures")
    rep = ROOT / cfg["reports"]["dir"]
    reporting.write_metrics_json(rep / "metrics.json", a, b, c, d, e, f, extra, run_info)
    reporting.write_results_md(rep / "RESULTS.md", a, b, c, d, e, f, extra, run_info)
    mlflow.log_artifact(str(rep / "metrics.json"))
    mlflow.log_artifact(str(rep / "RESULTS.md"))
    return {k: str(v) for k, v in figs.items()}


@flow(name="train-flow", log_prints=True)
def train_flow(config: dict) -> dict:
    check_reports_writable(config)
    tracking.configure(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"])
    split = load_data(config)
    safe_config = tracking.redact(config)  # no database passwords in MLflow artifacts
    config_text = json.dumps(safe_config, indent=2, sort_keys=True)
    lineage = {
        "data_md5": data_fingerprint(split),
        "config_md5": hashlib.md5(config_text.encode()).hexdigest(),
        "git_commit": git_commit(),
        "config_file": config.get("_config_path", "n/a"),
    }
    with mlflow.start_run(run_name="train", tags={"flow": "train", **lineage}) as parent:
        pid = parent.info.run_id
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            cfg_path.write_text(config_text)
            mlflow.log_artifact(str(cfg_path))
        mlflow.log_params({"seed": config["seed"], **split.sizes})
        a = part_a(split, config, pid)
        b = part_b(split, config, pid)
        c = part_c(config, pid)
        d = part_d(split, config, pid)
        e = part_e(split, config, pid)
        f = part_f(split, config, pid)
        extra = error_rate_by_flag(e.models["pca"], f.detector, split.X_test, split.y_test)
        tracking.log_metrics(extra, prefix="F_")
        versions = register_models(a, d, e, f, split, lineage)
        run_info = {
            "parent_run_id": pid,
            "config": config.get("_config_path", "n/a"),
            "model_versions": versions,
            "split_sizes": split.sizes,
            **lineage,
        }
        write_reports(a, b, c, d, e, f, split, config, extra, run_info)
        mlflow.log_metrics(
            {
                "A_n_components_95": a.metrics["n_components_95"],
                "C_kpca_test_accuracy": c.metrics["test_accuracy_supervised_choice"],
                "D_tsne_knn_accuracy_2d": d.metrics["tsne_knn_accuracy_2d"],
                "E_served_pca_accuracy": e.metrics["served_pca_accuracy"],
                "F_clean_false_alarm_pct": f.metrics["clean_false_alarm_pct"],
            }
        )
    return {"parent_run_id": pid, "model_versions": versions}


def start(config: dict) -> dict:
    return train_flow(config)
