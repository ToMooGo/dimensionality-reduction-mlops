"""End-to-end: train -> evaluate -> deploy with Prefect and a local MLflow (SQLite) on a small
MNIST subset. Downloads MNIST once (cached in data/). Run with ``pytest -m integration``."""

import pytest

pytestmark = pytest.mark.integration


def test_full_flow_quick(tmp_path, monkeypatch):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    from dimred_mlops.config import load_config
    from flows.full_flow import full_flow

    cfg = load_config(
        root / "configs/quick_flow_config.yaml",
        overrides={
            "data": {"n_train": 3000, "n_test": 1000},
            "reports": {
                "dir": str(tmp_path / "reports"),
                "figures_dir": str(tmp_path / "reports/figures"),
            },
            "part_b": {"sample_sizes": [1000, 3000]},
            "part_d": {
                "n_samples": 500,
                "n_atlas": 200,
                "methods": ["PCA", "Kernel PCA", "LLE", "t-SNE"],
            },
            "part_e": {"models": ["softmax", "svm"], "svm_train_size": 1500},
            "deploy": {"api_url": "http://127.0.0.1:9", "require_api": False, "reload_retries": 1},
        },
    )
    out = full_flow(cfg)
    assert out["gates_passed"] and out["deployed"]
    assert set(out["model_versions"]) == {
        "compressor",
        "classifier_raw",
        "classifier_pca",
        "detector",
        "atlas",
    }
    assert (tmp_path / "reports" / "RESULTS.md").exists()
    assert len(list((tmp_path / "reports" / "figures").glob("*.png"))) == 14

    from dimred_mlops import tracking

    atlas = tracking.load_model(tracking.MODEL_NAMES["atlas"], "champion")
    assert "PCA" in atlas.project([[0.0] * 784])
