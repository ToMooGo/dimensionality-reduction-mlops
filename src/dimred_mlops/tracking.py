"""Thin helpers around MLflow tracking and the Model Registry."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import mlflow.models  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

log = logging.getLogger(__name__)

# Registered model names - the API loads all five with the ``champion`` alias.
MODEL_NAMES = {
    "compressor": "mnist-pca-compressor",  # PCA with all 784 components: any compression level
    "classifier_raw": "mnist-svm-raw-pixels",  # SVM on the 784 raw pixels
    "classifier_pca": "mnist-svm-pca",  # PCA (95% variance) -> SVM
    "detector": "mnist-reconstruction-detector",  # PCA reconstruction error -> "unusual input"
    "atlas": "mnist-embedding-atlas",  # 2D maps (t-SNE, PCA, Kernel PCA, LLE, ...) for the UI
}
CHAMPION = "champion"
# Custom estimators that skops may deserialise (scikit-learn's own types are trusted by default).
TRUSTED_TYPES = [
    "dimred_mlops.detector.ReconstructionErrorDetector",
    "dimred_mlops.maps.EmbeddingAtlas",
]


def configure(tracking_uri: str | None, experiment: str) -> str:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment)
    return exp.experiment_id


def _is_number(v) -> bool:
    return (
        isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, bool)
        and math.isfinite(float(v))
    )


def log_metrics(metrics: dict, prefix: str = "") -> None:
    mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in metrics.items() if _is_number(v)})


def log_params(params: dict, prefix: str = "") -> None:
    mlflow.log_params({f"{prefix}{k}": str(v)[:500] for k, v in params.items()})


def log_table(df: pd.DataFrame, artifact_file: str) -> None:
    mlflow.log_text(df.to_csv(index=False), artifact_file)


def log_json(obj, artifact_file: str) -> None:
    mlflow.log_text(json.dumps(obj, indent=2, default=float), artifact_file)


def log_model(
    model, registered_name: str, input_example: np.ndarray, extra: dict | None = None
) -> str:
    """Log a scikit-learn model with skops serialisation and register it; return the version."""
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=registered_name,
        # transformers without predict() (the PCA compressor) get no pyfunc signature
        input_example=input_example if hasattr(model, "predict") else None,
        skops_trusted_types=TRUSTED_TYPES,
        metadata=extra or {},
    )
    return str(info.registered_model_version)


def set_alias(name: str, version: str, alias: str = CHAMPION) -> None:
    MlflowClient().set_registered_model_alias(name, alias, str(version))


# what MLflow answers when the model or the alias does not exist yet (anything else, such as a
# connection error, is a real failure and is raised)
_MISSING = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}


def get_alias_version(name: str, alias: str = CHAMPION):
    try:
        return MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        if exc.error_code in _MISSING:
            return None
        raise


def set_version_tags(name: str, version: str, tags: dict) -> None:
    client = MlflowClient()
    for k, v in tags.items():
        client.set_model_version_tag(name, str(version), k, str(v))


_SECRET = re.compile(r"(://[^:/@\s]+:)[^@\s]+@")


def redact(obj):
    """Copy of a config with passwords in connection URLs replaced by ``***``."""
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return _SECRET.sub(r"\1***@", obj)
    return obj


def latest_version(name: str) -> str | None:
    try:
        versions = MlflowClient().search_model_versions(f"name='{name}'")
    except MlflowException:
        return None
    return str(max(int(v.version) for v in versions)) if versions else None


def check_flavor(flavor: dict, ref: str = "model") -> None:
    """Refuse a model that is not skops-serialised or that asks to trust unknown types.

    The trusted-type list is stored inside the artifact, so anyone who can write artifacts
    could extend it. The allow-list in this module, not the artifact, decides.
    """
    if flavor.get("serialization_format") != "skops":
        raise ValueError(f"{ref}: only skops-serialised models are loaded")
    unknown = set(flavor.get("skops_trusted_types") or []) - set(TRUSTED_TYPES)
    if unknown:
        raise ValueError(f"{ref}: untrusted types requested: {sorted(unknown)}")


def load_model(name: str, version_or_alias: str):
    ref = (
        f"models:/{name}@{version_or_alias}"
        if not str(version_or_alias).isdigit()
        else f"models:/{name}/{version_or_alias}"
    )
    check_flavor(mlflow.models.get_model_info(ref).flavors.get("sklearn", {}), ref)
    return mlflow.sklearn.load_model(ref)


def download_run_artifact(run_id: str, path: str, dst: str | Path) -> Path:
    return Path(
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=str(dst))
    )
