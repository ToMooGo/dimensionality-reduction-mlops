"""Operational behaviour: secrets, exit codes, safe model reloads, rollback and drift filtering."""

import logging

import pytest

from dimred_mlops import tracking


def test_only_allow_listed_skops_types_are_loaded():
    ok = {"serialization_format": "skops", "skops_trusted_types": tracking.TRUSTED_TYPES}
    tracking.check_flavor(ok)
    with pytest.raises(ValueError, match="untrusted"):
        tracking.check_flavor({**ok, "skops_trusted_types": ["os.system"]})
    with pytest.raises(ValueError, match="skops"):
        tracking.check_flavor({"serialization_format": "cloudpickle"})


def test_redact_hides_passwords_in_urls():
    cfg = {
        "database": {"url": "postgresql+psycopg://mlops:s3cret@postgres:5432/app"},
        "mlflow": {"tracking_uri": "http://mlflow:5000"},
        "list": ["sqlite:///app.db", "postgresql://u:p@h/db"],
    }
    out = tracking.redact(cfg)
    assert out["database"]["url"] == "postgresql+psycopg://mlops:***@postgres:5432/app"
    assert out["mlflow"]["tracking_uri"] == "http://mlflow:5000"
    assert out["list"] == ["sqlite:///app.db", "postgresql://u:***@h/db"]
    assert "s3cret" not in str(out)


def test_exit_code_reflects_gates_and_deployment():
    from run_flow import exit_code

    assert exit_code({"gates_passed": True, "deployed": True}) == 0
    assert exit_code({"gates_passed": False, "deployed": False}) == 2
    assert exit_code({"passed": False}) == 2
    assert exit_code({"gates_passed": True, "deployed": False}) == 3
    assert exit_code({"status": "ok"}) == 0  # monitor flow


def test_model_store_waits_for_a_complete_promotion(monkeypatch):
    from app.model_store import ModelBundle, ModelStore

    keys = list(tracking.MODEL_NAMES)
    store = ModelStore()
    store.set_bundle(ModelBundle(*([None] * 5), versions={k: "1" for k in keys}))
    calls = []
    monkeypatch.setattr(store, "try_load", lambda: calls.append(1) or True)

    # two of five aliases moved: a promotion is in progress -> keep serving the old set
    partial = {k: ("2" if i < 2 else "1") for i, k in enumerate(keys)}
    monkeypatch.setattr(store, "registry_versions", lambda: partial)
    assert store.refresh_if_stale() is False and not calls

    monkeypatch.setattr(store, "registry_versions", lambda: {k: "2" for k in keys})
    assert store.refresh_if_stale() is True and calls == [1]

    monkeypatch.setattr(store, "registry_versions", lambda: {k: "1" for k in keys})
    calls.clear()
    assert store.refresh_if_stale() is False and not calls  # unchanged


def test_deploy_restores_the_previous_champion_when_reload_fails(monkeypatch):
    import flows.deploy_flow as deploy

    aliases = {name: "1" for name in tracking.MODEL_NAMES.values()}

    class MV:
        def __init__(self, v):
            self.version = v

    monkeypatch.setattr(deploy, "get_run_logger", lambda: logging.getLogger("test"))
    monkeypatch.setattr(deploy.tracking, "configure", lambda *a, **k: None)
    monkeypatch.setattr(
        deploy.tracking, "get_alias_version", lambda name, alias="champion": MV(aliases[name])
    )

    def promote(versions, alias):
        for key, v in versions.items():
            aliases[tracking.MODEL_NAMES[key]] = v

    def reload_api(*args, **kwargs):
        raise RuntimeError("API cannot load the new models")

    monkeypatch.setattr(deploy, "promote", promote)
    monkeypatch.setattr(deploy, "reload_api", reload_api)
    cfg = {
        "mlflow": {"tracking_uri": None, "experiment": "x"},
        "deploy": {"alias": "champion", "api_url": "http://api", "require_api": "true"},
    }
    with pytest.raises(RuntimeError):
        deploy.deploy_flow.fn(cfg, {k: "2" for k in tracking.MODEL_NAMES}, True)
    assert set(aliases.values()) == {"1"}  # rolled back


def test_drift_check_ignores_demo_and_ci_traffic(tmp_path):
    from dimred_mlops.db import Prediction, init_db, make_engine
    from flows.monitor_flow import fetch_predictions

    url = f"sqlite:///{tmp_path}/app.db"
    Session = init_db(make_engine(url))
    with Session() as s:
        for src in ["canvas", "canvas", "sample-noise", "sample-clean", "ci", "api"]:
            s.add(
                Prediction(
                    source=src,
                    pixels=[0] * 784,
                    digit_raw=1,
                    digit_pca=1,
                    latency_raw_ms=1.0,
                    latency_pca_ms=1.0,
                    reconstruction_error=100.0,
                    error_threshold=400.0,
                    is_unusual=src == "sample-noise",
                    model_versions={},
                    latency_ms=2.0,
                )
            )
        s.commit()
    rows = fetch_predictions.fn(url, 24, ("sample-", "ci"))
    assert sorted(r["source"] for r in rows) == ["api", "canvas", "canvas"]
    assert "pixels" not in rows[0]
    assert len(fetch_predictions.fn(url, 24, ())) == 6
