from dimred_mlops.gates import all_passed, evaluate_gates
from dimred_mlops.monitoring import summarize_predictions

GATES = {
    "compressor_max_components_95": 200,
    "compressor_min_test_explained_variance_95": 0.94,
    "classifier_min_accuracy": 0.95,
    "classifier_pca_max_drop_pp": 1.0,
    "detector_max_false_alarm_pct": 6.0,
    "detector_min_detection_pct": {"noise": 95, "inverted": 95},
    "atlas_min_projectable": 3,
}
GOOD = {
    "compressor_n_components_95": 154,
    "compressor_test_explained_variance_95": 0.949,
    "classifier_raw_accuracy": 0.970,
    "classifier_pca_accuracy": 0.975,
    "detector_clean_false_alarm_pct": 3.3,
    "detector_detection_pct_noise": 100.0,
    "detector_detection_pct_inverted": 100.0,
    "atlas_n_projectable": 3,
}


def test_good_candidate_passes():
    checks = evaluate_gates(GOOD, None, GATES)
    assert all_passed(checks)
    assert len(checks) == 7 + 2


def test_each_gate_can_fail():
    for key, bad in [
        ("compressor_n_components_95", 250),
        ("classifier_pca_accuracy", 0.94),
        ("detector_clean_false_alarm_pct", 9.0),
        ("detector_detection_pct_noise", 50.0),
        ("atlas_n_projectable", 1),
    ]:
        assert not all_passed(evaluate_gates({**GOOD, key: bad}, None, GATES)), key


def test_pca_model_may_not_fall_far_behind_the_raw_pixel_model():
    cand = {**GOOD, "classifier_raw_accuracy": 0.99, "classifier_pca_accuracy": 0.975}
    failed = [c["check"] for c in evaluate_gates(cand, None, GATES) if not c["passed"]]
    assert failed == ["classifier (PCA): accuracy vs raw pixels (pp)"]


def test_champion_vs_challenger():
    champion = {**GOOD, "classifier_pca_accuracy": 0.985}
    checks = evaluate_gates(GOOD, champion, GATES, tolerance_pp=0.5)
    assert not all_passed(checks)  # 1 pp worse than the champion
    assert all_passed(evaluate_gates(GOOD, champion, GATES, tolerance_pp=1.5))


MON = {
    "min_predictions": 5,
    "max_unusual_rate_pct": 12,
    "max_class_share_pct": 40,
    "min_models_agree_pct": 80,
}


def _rows(n, unusual=0, digit=lambda i: i % 10, disagree=0):
    return [
        {
            "digit_raw": digit(i) if i >= disagree else (digit(i) + 1) % 10,
            "digit_pca": digit(i),
            "is_unusual": i < unusual,
            "reconstruction_error": 200.0 + i,
        }
        for i in range(n)
    ]


def test_monitoring_ok_and_insufficient():
    assert summarize_predictions(_rows(3), MON)["status"] == "insufficient-data"
    rep = summarize_predictions(_rows(50), MON)
    assert rep["status"] == "ok" and rep["unusual_rate_pct"] == 0 and rep["models_agree_pct"] == 100


def test_monitoring_alerts():
    assert summarize_predictions(_rows(50, unusual=10), MON)["status"] == "alert"
    assert summarize_predictions(_rows(50, digit=lambda i: 7), MON)["status"] == "alert"
    rep = summarize_predictions(_rows(50, disagree=20), MON)
    assert any("agree" in a for a in rep["alerts"])
