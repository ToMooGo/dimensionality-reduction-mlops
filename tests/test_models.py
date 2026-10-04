"""Unit tests for Parts A-F on small MNIST-like data (no download)."""

import base64

import numpy as np
import pytest
import skops.io as sio
from sklearn.decomposition import PCA, IncrementalPCA

from dimred_mlops import tracking
from dimred_mlops.classification import make_pair, run_part_e
from dimred_mlops.compression import (
    n_components_for,
    reconstruct,
    reconstruction_mse,
    run_part_a,
    storage_table,
)
from dimred_mlops.detector import ReconstructionErrorDetector, error_rate_by_flag, run_part_f
from dimred_mlops.manifold import kpca_pipeline, preimage_errors, run_part_c
from dimred_mlops.maps import PROJECTABLE, EmbeddingAtlas, run_part_d, stratified_sample
from dimred_mlops.scaling import (
    effective_batches,
    ipca_partial_fit,
    open_memmap,
    run_part_b,
    write_memmap,
)


# ---------------------------------------------------------------- A. compression
def test_n_components_for_uses_the_books_argmax_rule():
    cum = np.array([0.5, 0.8, 0.94, 0.95, 0.99, 1.0])
    assert n_components_for(cum, 0.95) == 4
    assert n_components_for(cum, 0.5) == 1


def test_truncated_reconstruction_equals_a_pca_with_d_components(split):
    full = PCA(svd_solver="full").fit(split.X_train)
    for d in (5, 40):
        pca_d = PCA(n_components=d, svd_solver="full").fit(split.X_train)
        expected = pca_d.inverse_transform(pca_d.transform(split.X_test))
        np.testing.assert_allclose(reconstruct(full, split.X_test, d), expected, atol=1e-2)


def test_manual_svd_matches_scikit_learn_up_to_sign(split):
    # the book's NumPy code (p. 223)
    X_centered = split.X_train - split.X_train.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_centered.astype(np.float64), full_matrices=False)
    pca = PCA(n_components=2, svd_solver="full").fit(split.X_train)
    for i in range(2):
        assert abs(abs(Vt[i] @ pca.components_[i]) - 1) < 1e-4


def test_training_reconstruction_error_is_the_dropped_variance(split):
    a = run_part_a(split, {"target_variance": 0.95, "variance_targets": [0.9, 0.95]})
    m = a.metrics
    assert m["train_mse_95"] == pytest.approx(m["dropped_variance_per_pixel_95"], rel=1e-3)
    assert a.metrics["train_explained_variance_95"] >= 0.95
    assert list(a.variance_table.variance_target) == [0.9, 0.95]
    assert a.variance_table.n_components.is_monotonic_increasing


def test_storage_table_bytes():
    t = storage_table(154, 60_000).set_index("representation")
    assert t.loc["raw pixels, uint8 (how MNIST is shipped)", "bytes_per_image"] == 784
    row = t.loc["PCA codes, float32 (154 values)"]
    assert row.bytes_per_image == 616
    assert row.pct_of_float32_pixels == pytest.approx(19.64, abs=0.01)  # the book's "< 20%"


def test_reconstruction_mse_shape(split):
    assert reconstruction_mse(split.X_test, split.X_test).shape == (len(split.X_test),)


# ---------------------------------------------------------------- B. scaling
def test_effective_batches_keep_at_least_n_components_rows():
    assert effective_batches(60_000, 154, 100) == 100
    assert effective_batches(10_000, 154, 100) == 64
    assert effective_batches(100, 154, 100) == 1


def test_memmap_partial_fit_equals_in_memory_partial_fit(split, tmp_path):
    path, shape = write_memmap(split.X_train, tmp_path)
    a = ipca_partial_fit(open_memmap(path, shape), 20, 10)
    b = ipca_partial_fit(split.X_train, 20, 10)
    np.testing.assert_allclose(a.components_, b.components_, atol=1e-4)
    assert isinstance(a, IncrementalPCA)


def test_run_part_b_reports_every_solver(split):
    b = run_part_b(
        split, {"n_components": 10, "n_batches": 5, "repeats": 1, "sample_sizes": [300, 600]}
    )
    assert len(b.solvers) == 6
    assert (
        b.solvers.mse_vs_exact_pct.abs() < 5
    ).all()  # every solver finds about the same subspace
    assert set(b.timing_curve.n_train) == {300, 600}
    assert "algorithm" in b.timing_curve
    # reading the memmap slice by slice needs less memory than validating the whole array
    assert b.metrics["memmap_partial_fit_peak_MB"] < b.metrics["memmap_fit_peak_MB"]


# ---------------------------------------------------------------- C. Kernel PCA / manifolds
def test_kpca_pipeline_grid_has_twenty_settings():
    from sklearn.model_selection import ParameterGrid

    grid = ParameterGrid(
        [{"kpca__gamma": np.linspace(0.03, 0.05, 10), "kpca__kernel": ["rbf", "sigmoid"]}]
    )
    assert len(grid) == 20
    assert list(kpca_pipeline().named_steps) == ["kpca", "log_reg"]


def test_preimage_errors_table_uses_training_points_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 3))
    t = preimage_errors(X, ["rbf"], [0.03, 0.05], cv=3)
    assert len(t) == 2
    assert (t.cv_preimage_mse > 0).all()
    # the error on the fitting points is optimistic compared with the left-out folds
    assert (t.fit_preimage_mse <= t.cv_preimage_mse + 1e-9).all()


def test_mean_ci_matches_the_books_t_interval():
    from scipy import stats

    from dimred_mlops.stats import mean_ci

    v = np.r_[np.ones(970), np.zeros(30)]
    m, lo, hi = mean_ci(v)
    exp = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
    assert m == pytest.approx(0.97) and (lo, hi) == pytest.approx(exp)
    assert mean_ci(np.ones(10)) == (1.0, 1.0, 1.0)


def test_run_part_c_small():
    c = run_part_c({"n_samples": 300, "gammas": {"start": 0.03, "stop": 0.05, "num": 3}})
    assert c.metrics["n_settings"] == 6
    # unrolling the roll with kPCA beats a linear classifier on linear PCA
    assert c.metrics["test_accuracy_supervised_choice"] > c.metrics["test_accuracy_linear_pca_2d"]
    assert set(c.methods.method) == {"PCA", "Kernel PCA (rbf)", "LLE", "Isomap", "MDS", "t-SNE"}


# ---------------------------------------------------------------- D. maps / atlas
def test_stratified_sample_keeps_class_shares():
    y = np.repeat(np.arange(10), [100, 200, 100, 100, 100, 100, 100, 100, 100, 100])
    idx = stratified_sample(y, 110, 0)
    assert len(set(idx)) == len(idx) == 110
    assert np.bincount(y[idx])[1] == 20


@pytest.fixture(scope="module")
def maps(split):
    return run_part_d(
        split,
        {
            "n_samples": 400,
            "n_atlas": 150,
            "n_api_samples": 30,
            "methods": ["PCA", "Kernel PCA", "LLE", "t-SNE"],
        },
    )


def test_maps_scores_and_atlas(maps, split):
    assert set(maps.table.method) == {"PCA", "Kernel PCA", "LLE", "t-SNE"}
    assert maps.table.knn_accuracy_2d.between(0, 1).all()
    atlas = maps.atlas
    assert set(atlas.reducers) == set(PROJECTABLE)
    pos = atlas.project(split.X_test[:3])
    assert set(pos) == set(PROJECTABLE) and np.asarray(pos["PCA"]).shape == (3, 2)
    assert atlas.predict(split.X_test[:2]).shape == (2, 2)
    payload = atlas.to_payload()
    assert payload["n_points"] == 150
    assert len(base64.b64decode(payload["thumbnails_b64"])) == 150 * 784
    tsne = next(m for m in payload["methods"] if m["name"] == "t-SNE")
    assert tsne["projects_new_points"] is False and len(tsne["coords"]) == 150


def test_atlas_survives_skops_with_declared_trusted_types(maps):
    blob = sio.dumps(maps.atlas)
    untrusted = set(sio.get_untrusted_types(data=blob))
    assert untrusted <= set(tracking.TRUSTED_TYPES)
    restored = sio.loads(blob, trusted=list(untrusted))
    assert isinstance(restored, EmbeddingAtlas)


# ---------------------------------------------------------------- E. classification
def test_pairs_differ_only_by_pca():
    pair = make_pair("svm", {"pca_variance": 0.9}, 0)
    assert list(pair["raw"].named_steps) == ["clf"]
    assert list(pair["pca"].named_steps) == ["pca", "clf"]


def test_run_part_e_small(split):
    e = run_part_e(
        split, {"models": ["softmax", "svm"], "softmax_max_iter": 200, "svm_train_size": 800}
    )
    assert set(e.models) == {"raw", "pca"}
    assert len(e.table) == 5  # softmax raw/pca + svm raw/pca/pca with the raw model's gamma
    # the PCA is unsupervised and fitted on all training images, the SVM on its subset
    assert e.models["pca"].named_steps["pca"].n_samples_ == len(split.X_train)
    row = e.table[(e.table.kind == "svm") & (e.table.features == "pca")].iloc[0]
    assert row.accuracy_ci_low <= row.test_accuracy <= row.accuracy_ci_high
    lo, hi = e.metrics["svm_accuracy_change_ci_low_pp"], e.metrics["svm_accuracy_change_ci_high_pp"]
    assert lo <= e.metrics["svm_accuracy_change_pp"] <= hi
    assert e.metrics["served_pca_accuracy"] > 0.8
    assert (e.table.loc[e.table.features == "pca", "n_features"] < 784).all()


# ---------------------------------------------------------------- F. detector
@pytest.fixture(scope="module")
def detector(split):
    return ReconstructionErrorDetector(variance=0.9, flag_rate=0.05, calibration_size=300).fit(
        split.X_train
    )


def test_detector_threshold_is_the_calibration_quantile(detector):
    flagged = np.mean(detector.calibration_errors_ > detector.threshold_)
    assert flagged == pytest.approx(0.05, abs=0.01)


def test_detector_flags_noise_not_clean_digits(detector, split):
    from dimred_mlops.data import corrupt_digits

    assert detector.is_unusual(corrupt_digits(split.X_test[:50], "noise")).all()
    assert detector.predict(split.X_test).mean() < 0.15
    pct = detector.error_percentile(
        np.vstack([split.X_test[:1], corrupt_digits(split.X_test[:1], "noise")])
    )
    assert pct[1] == 100 and pct[0] < pct[1]


def test_detector_skops_roundtrip(detector, split):
    blob = sio.dumps(detector)
    restored = sio.loads(blob, trusted=sio.get_untrusted_types(data=blob))
    np.testing.assert_allclose(
        restored.reconstruction_error(split.X_test[:5]),
        detector.reconstruction_error(split.X_test[:5]),
    )
    assert set(sio.get_untrusted_types(data=blob)) <= set(tracking.TRUSTED_TYPES)


def test_run_part_f_small(split):
    f = run_part_f(
        split, {"calibration_size": 300, "variance_sweep": [0.9], "threshold_sweep": [2, 4]}
    )
    assert f.metrics["detect_noise_pct"] == 100
    assert len(f.threshold_sweep) == 2 and len(f.variance_sweep) == 1
    assert 0.5 < f.metrics["roc_auc"] <= 1


def test_error_rate_by_flag(detector, split):
    class Constant:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    out = error_rate_by_flag(Constant(), detector, split.X_test, split.y_test)
    assert out["n_errors_total"] == int((split.y_test != 0).sum())
