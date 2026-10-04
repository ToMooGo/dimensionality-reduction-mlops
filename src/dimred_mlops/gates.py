"""Deployment quality gates and the champion-vs-challenger rule (pure functions, unit-tested)."""

from __future__ import annotations

import numpy as np

from .compression import n_components_for, reconstruct, reconstruction_mse


def evaluate_models(models: dict, split, seed: int = 42) -> dict:
    """Test-set metrics for the five registered models (used by the gates and champion checks)."""
    from .detector import evaluate_on_corruptions

    comp = models["compressor"]
    d = n_components_for(np.cumsum(comp.explained_variance_ratio_), 0.95)
    mse = reconstruction_mse(split.X_test, reconstruct(comp, split.X_test, d))
    total = float(np.var(split.X_test, axis=0).sum()) * len(split.X_test)
    table, summary = evaluate_on_corruptions(models["detector"], split.X_test, seed)
    atlas = models["atlas"]
    projected = atlas.project(split.X_test[:2])
    return {
        "compressor_n_components_95": d,
        # share of the test set's variance that survives compression -> decompression
        "compressor_test_explained_variance_95": 1
        - float(mse.sum()) * split.X_test.shape[1] / total,
        "classifier_raw_accuracy": float(
            np.mean(models["classifier_raw"].predict(split.X_test) == split.y_test)
        ),
        "classifier_pca_accuracy": float(
            np.mean(models["classifier_pca"].predict(split.X_test) == split.y_test)
        ),
        "detector_clean_false_alarm_pct": summary["clean_false_alarm_pct"],
        "detector_roc_auc": summary["roc_auc"],
        **{
            f"detector_detection_pct_{r.input}": float(r.flagged_pct)
            for r in table.itertuples()
            if r.input != "clean test digits"
        },
        "atlas_n_maps": len(atlas.coords),
        "atlas_n_projectable": sum(len(v) == 2 for v in projected.values()),
    }


def _check(name: str, value: float, limit: float, passed: bool) -> dict:
    return {"check": name, "value": float(value), "limit": float(limit), "passed": bool(passed)}


def evaluate_gates(
    candidate: dict, champion: dict | None, gates: dict, tolerance_pp: float = 0.5
) -> list[dict]:
    """One record per check: ``{"check", "value", "limit", "passed"}``.

    A candidate set of models is deployable only if every check passes.
    """
    m = candidate
    checks = [
        _check(
            "compressor: components for 95% variance",
            m["compressor_n_components_95"],
            gates["compressor_max_components_95"],
            m["compressor_n_components_95"] <= gates["compressor_max_components_95"],
        ),
        _check(
            "compressor: test variance kept at 95% setting",
            m["compressor_test_explained_variance_95"],
            gates["compressor_min_test_explained_variance_95"],
            m["compressor_test_explained_variance_95"]
            >= gates["compressor_min_test_explained_variance_95"],
        ),
        _check(
            "classifier (raw pixels): accuracy",
            m["classifier_raw_accuracy"],
            gates["classifier_min_accuracy"],
            m["classifier_raw_accuracy"] >= gates["classifier_min_accuracy"],
        ),
        _check(
            "classifier (PCA): accuracy",
            m["classifier_pca_accuracy"],
            gates["classifier_min_accuracy"],
            m["classifier_pca_accuracy"] >= gates["classifier_min_accuracy"],
        ),
        _check(
            "classifier (PCA): accuracy vs raw pixels (pp)",
            100 * (m["classifier_pca_accuracy"] - m["classifier_raw_accuracy"]),
            -gates["classifier_pca_max_drop_pp"],
            100 * (m["classifier_pca_accuracy"] - m["classifier_raw_accuracy"])
            >= -gates["classifier_pca_max_drop_pp"],
        ),
        _check(
            "detector: false alarms on clean digits (%)",
            m["detector_clean_false_alarm_pct"],
            gates["detector_max_false_alarm_pct"],
            m["detector_clean_false_alarm_pct"] <= gates["detector_max_false_alarm_pct"],
        ),
        _check(
            "atlas: maps that can place a new digit",
            m["atlas_n_projectable"],
            gates.get("atlas_min_projectable", 1),
            m["atlas_n_projectable"] >= gates.get("atlas_min_projectable", 1),
        ),
    ]
    for kind, minimum in gates.get("detector_min_detection_pct", {}).items():
        value = m.get(f"detector_detection_pct_{kind}", 0.0)
        checks.append(
            _check(f"detector: catches '{kind}' inputs (%)", value, minimum, value >= minimum)
        )
    if champion:
        tol = tolerance_pp / 100
        for key in ("classifier_raw_accuracy", "classifier_pca_accuracy"):
            checks.append(
                _check(
                    f"{key} vs champion", m[key], champion[key] - tol, m[key] >= champion[key] - tol
                )
            )
    return checks


def all_passed(checks: list[dict]) -> bool:
    return all(c["passed"] for c in checks)
