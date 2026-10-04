"""API tests with small in-memory models (no MLflow, no download)."""

import numpy as np
import pytest
from app.main import create_app
from app.model_store import ModelBundle, ModelStore
from fastapi.testclient import TestClient
from sklearn.decomposition import PCA

from dimred_mlops.classification import make_pair
from dimred_mlops.detector import ReconstructionErrorDetector
from dimred_mlops.maps import run_part_d


@pytest.fixture(scope="module")
def bundle(split):
    pair = make_pair("svm", {"pca_variance": 0.9}, 0)
    for m in pair.values():
        m.fit(split.X_train[:600], split.y_train[:600])
    maps = run_part_d(
        split,
        {
            "n_samples": 300,
            "n_atlas": 120,
            "n_api_samples": 20,
            "methods": ["PCA", "Kernel PCA", "LLE", "t-SNE"],
        },
    )
    return ModelBundle(
        compressor=PCA(svd_solver="full").fit(split.X_train),
        classifier_raw=pair["raw"],
        classifier_pca=pair["pca"],
        detector=ReconstructionErrorDetector(0.9, 0.05, 300).fit(split.X_train),
        atlas=maps.atlas,
        versions={
            k: "1" for k in ("compressor", "classifier_raw", "classifier_pca", "detector", "atlas")
        },
    )


@pytest.fixture()
def client(bundle, tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    store = ModelStore()
    store.set_bundle(bundle)
    app = create_app(store=store, database_url=f"sqlite:///{tmp_path}/app.db", load_models=False)
    with TestClient(app) as c:
        yield c


def test_health_ready_and_ui(client):
    assert client.get("/ready").status_code == 200
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_versions"]["atlas"] == "1"
    assert "PCA Lab" in client.get("/").text


def test_compress_all_components_is_lossless(client, split):
    px = split.X_test[0].tolist()
    r = client.post("/compress", json={"pixels": px, "n_components": 784}).json()
    assert r["rmse_grey_levels"] < 1.0 and r["variance_kept"] == pytest.approx(1.0, abs=1e-4)
    small = client.post("/compress", json={"pixels": px, "n_components": 5}).json()
    assert small["mse"] > r["mse"] and small["bytes_float32_code"] == 20
    assert (
        len(small["reconstruction"]) == 784
        and 0 <= min(small["reconstruction"]) <= max(small["reconstruction"]) <= 255
    )


def test_compress_info(client):
    info = client.get("/compress/info").json()
    assert len(info["cumulative_variance"]) == 784 and "95%" in info["marks"]


def test_predict_logs_and_flags(client, split):
    clean = client.post(
        "/predict", json={"pixels": split.X_test[1].tolist(), "source": "test"}
    ).json()
    assert clean["prediction_id"] == 1
    assert 0 <= clean["raw"]["digit"] <= 9 and clean["raw"]["n_features"] == 784
    assert clean["pca"]["n_features"] < 784 and len(clean["pca"]["scores"]) == 10
    assert set(clean["map_positions"]) == {"PCA", "Kernel PCA", "LLE"}
    noise = np.random.default_rng(0).uniform(0, 255, 784).tolist()
    r = client.post("/predict", json={"pixels": noise, "source": "noise"}).json()
    assert r["unusual"]["is_unusual"] and r["unusual"]["error_percentile"] == 100
    summary = client.get("/monitoring/summary").json()
    assert summary["n_predictions"] == 2 and summary["by_source"]["noise"]["unusual"] == 1
    assert summary["expected_unusual_rate_pct"] == 5.0  # the detector's own flag rate
    assert len(summary["recent"]) == 2 and len(summary["recent"][0]["pixels"]) == 784


@pytest.mark.parametrize(
    "payload",
    [
        {"pixels": [0] * 783},
        {"pixels": [0] * 783 + [300]},
        {"pixels": [0] * 784, "source": "<script>"},
        {"pixels": [0] * 784, "n_components": 0},
    ],
)
def test_validation(client, payload):
    path = "/compress" if "n_components" in payload else "/predict"
    assert client.post(path, json=payload).status_code == 422


def test_nan_is_rejected_without_crashing(client):
    body = '{"pixels": [' + ",".join(["NaN"] + ["0"] * 783) + "]}"
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_maps_payload(client):
    m = client.get("/maps").json()
    assert m["n_points"] == 120 and len(m["labels"]) == 120
    names = {x["name"]: x for x in m["methods"]}
    assert (
        names["t-SNE"]["projects_new_points"] is False
        and names["LLE"]["projects_new_points"] is True
    )


def test_samples(client):
    s = client.get("/samples?kind=inverted&seed=3").json()
    assert s["kind"] == "inverted" and len(s["pixels"]) == 784
    assert client.get("/samples?kind=clean&seed=3").json()["true_label"] in range(10)
    assert client.get("/samples?kind=blurred").status_code == 422


def test_no_models_gives_503(tmp_path):
    app = create_app(
        store=ModelStore(), database_url=f"sqlite:///{tmp_path}/x.db", load_models=False
    )
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "no-models"
        assert c.get("/ready").status_code == 503
        assert c.post("/compress", json={"pixels": [0] * 784}).status_code == 503


def test_reload_requires_token(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    store = ModelStore()
    store.set_bundle(bundle)
    app = create_app(store=store, database_url=f"sqlite:///{tmp_path}/y.db", load_models=False)
    with TestClient(app) as c:
        assert c.post("/reload").status_code == 401
        # a non-ASCII header must be rejected cleanly, not crash the comparison
        bad = {"X-Admin-Token": "s\xe9cret".encode("latin-1")}
        assert c.post("/reload", headers=bad).status_code == 401
        # right token, but no MLflow registry here -> 503 rather than a crash
        assert c.post("/reload", headers={"X-Admin-Token": "s3cret"}).status_code == 503
