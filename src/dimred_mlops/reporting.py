"""Figures, ``reports/metrics.json`` and ``reports/RESULTS.md``, regenerated on every training run."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from . import plots


def make_figures(a, b, c, d, e, f, split, fig_dir: Path) -> dict[str, Path]:
    fig_dir = Path(fig_dir)
    best_sig = c.grid[c.grid.kernel == "sigmoid"].sort_values("cv_accuracy").iloc[-1]
    return {
        "a_explained_variance": plots.explained_variance(
            a.cumulative, a.variance_table, fig_dir / "a_explained_variance.png"
        ),
        "a_reconstructions": plots.reconstructions(
            split.X_test,
            a.pca_full,
            a.variance_table,
            a.example_index,
            fig_dir / "a_reconstructions.png",
        ),
        "a_principal_components": plots.principal_components(
            a.pca_full, fig_dir / "a_principal_components.png"
        ),
        "b_solvers": plots.solver_comparison(b.solvers, fig_dir / "b_solvers.png"),
        "b_timing_curve": plots.timing_curve(b.timing_curve, fig_dir / "b_timing_curve.png"),
        "c_swiss_roll_kpca": plots.swiss_roll_kpca(
            c.roll,
            fig_dir / "c_swiss_roll_kpca.png",
            c.metrics["best_gamma"],
            float(best_sig.gamma),
        ),
        "c_kpca_selection": plots.kpca_selection(
            c.grid,
            c.preimage,
            c.metrics["best_gamma"],
            c.metrics["preimage_best_gamma"],
            c.params["best_kernel"],
            c.params["preimage_best_kernel"],
            fig_dir / "c_kpca_selection.png",
        ),
        "c_manifold_methods": plots.manifold_methods(
            c.embeddings, c.roll["t"], c.methods, fig_dir / "c_manifold_methods.png"
        ),
        "d_tsne_map": plots.tsne_map(
            d.embeddings["t-SNE"], d.labels, d.images, fig_dir / "d_tsne_map.png"
        ),
        "d_maps_comparison": plots.maps_comparison(
            d.embeddings, d.labels, d.table, fig_dir / "d_maps_comparison.png"
        ),
        "e_classification": plots.classification(e.table, fig_dir / "e_classification.png"),
        "f_error_histograms": plots.error_histograms(
            f.errors, f.detector.threshold_, f.corruption_table, fig_dir / "f_error_histograms.png"
        ),
        "f_corruption_examples": plots.corruption_examples(
            split.X_test, f.detector, fig_dir / "f_corruption_examples.png"
        ),
        "f_detection": plots.detection_rates(
            f.corruption_table, f.threshold_sweep, fig_dir / "f_detection.png"
        ),
    }


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(obj, pd.DataFrame):
        return [_clean(r) for r in obj.to_dict(orient="records")]
    return obj


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "cpus": __import__("os").cpu_count(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_metrics_json(path: Path, a, b, c, d, e, f, extra: dict, run_info: dict) -> Path:
    payload = {
        "run": run_info,
        "environment": environment(),
        "A_compression": {
            "params": a.params,
            "metrics": a.metrics,
            "variance_table": a.variance_table,
            "storage": a.storage,
        },
        "B_scaling": {
            "params": b.params,
            "metrics": b.metrics,
            "solvers": b.solvers,
            "timing_curve": b.timing_curve,
        },
        "C_kernel_pca_manifolds": {
            "params": c.params,
            "metrics": c.metrics,
            "grid": c.grid,
            "preimage": c.preimage,
            "methods": c.methods,
        },
        "D_mnist_maps": {"params": d.params, "metrics": d.metrics, "methods": d.table},
        "E_classification": {"params": e.params, "metrics": e.metrics, "table": e.table},
        "F_unusual_inputs": {
            "params": f.params,
            "metrics": f.metrics,
            "corruptions": f.corruption_table,
            "threshold_sweep": f.threshold_sweep,
            "variance_sweep": f.variance_sweep,
            "classifier_error_by_flag": extra,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=2))
    return path


def _md(df: pd.DataFrame, formats: dict | None = None) -> str:
    formats = formats or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, row, strict=True):
            fmt = formats.get(c)
            if fmt and isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(fmt.format(v))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_md(path: Path, a, b, c, d, e, f, extra: dict, run_info: dict) -> Path:
    env = environment()
    am, bm, cm, dm, em, fm = a.metrics, b.metrics, c.metrics, d.metrics, e.metrics, f.metrics
    serve = e.params["served_classifier"]
    vt = a.variance_table[
        [
            "variance_target",
            "n_components",
            "pct_of_features",
            "train_explained_variance",
            "test_explained_variance",
            "test_mse",
            "test_rmse_pixels",
        ]
    ]
    st = a.storage[
        [
            "representation",
            "bytes_per_image",
            "dataset_MB",
            "pct_of_float32_pixels",
            "pct_of_uint8_pixels",
        ]
    ]
    sv = b.solvers[
        [
            "solver",
            "algorithm",
            "data",
            "fit_seconds",
            "runs",
            "peak_fit_memory_MB",
            "total_RAM_MB",
            "explained_variance",
            "test_mse",
            "mse_vs_exact_pct",
        ]
    ]
    grid_top = c.grid.sort_values("cv_accuracy", ascending=False).head(5)
    pre_top = c.preimage.sort_values("cv_preimage_mse").head(5)
    text = (
        f"""# Results

*Generated by the training pipeline. Do not edit by hand. Config: `{run_info.get("config", "n/a")}`; MLflow parent run `{run_info.get("parent_run_id", "n/a")}`; registered model versions {run_info.get("model_versions", {})}.*
*Machine: {env["cpus"]} CPUs, Python {env["python"]}, scikit-learn {env["scikit_learn"]}, NumPy {env["numpy"]}. Generated {env["generated_at"]}.*
*Data: MNIST, first {split_sizes(run_info)["n_train"]:,} images for training, last {split_sizes(run_info)["n_test"]:,} for testing (Geron, Ch. 8, exercise 9).*

## A. PCA for compression

* **{am["n_components_95"]} of 784 components ({am["pct_of_features_95"]:.1f}% of the features) keep {am["train_explained_variance_95"]:.2%} of the training variance**. On the test set the compressed-then-decompressed images keep {am["test_explained_variance_95"]:.2%} of the variance.
* Test reconstruction error at 95%: MSE {am["test_mse_95"]:.1f}, i.e. a typical error of {am["test_rmse_pixels_95"]:.1f} grey levels out of 255 per pixel.
* Check: the training reconstruction MSE ({am["train_mse_95"]:.2f}) equals the variance of the dropped components per pixel ({am["dropped_variance_per_pixel_95"]:.2f}).

{_md(vt, {"variance_target": "{:.0%}", "pct_of_features": "{:.1f}%", "train_explained_variance": "{:.4f}", "test_explained_variance": "{:.4f}", "test_mse": "{:.1f}", "test_rmse_pixels": "{:.1f}"})}

**Storage.** The book's "less than 20% of its original size" compares numbers of features. In bytes, it depends on how each is stored:

{_md(st, {"dataset_MB": "{:.1f}", "pct_of_float32_pixels": "{:.1f}%", "pct_of_uint8_pixels": "{:.1f}%"})}

![](figures/a_explained_variance.png)
![](figures/a_reconstructions.png)
![](figures/a_principal_components.png)

## B. Scaling PCA: Randomized, Incremental and out-of-core

154 components of the {b.metrics["data_MB"]:.0f} MB float32 training matrix. scikit-learn's `svd_solver="auto"` picked **`{b.params["auto_solver_chosen"]}`** here.

{_md(sv, {"fit_seconds": "{:.2f}", "peak_fit_memory_MB": "{:.0f}", "total_RAM_MB": "{:.0f}", "explained_variance": "{:.4f}", "test_mse": "{:.1f}", "mse_vs_exact_pct": "{:+.2f}%"})}

* Randomized PCA and Incremental PCA find practically the same subspace: test MSE {bm["randomized_mse_gap_pct"]:+.2f}% and {bm["ipca_mse_gap_pct"]:+.2f}% versus the exact SVD.
* **Out-of-core:** feeding memmap slices to `partial_fit` needs at most {bm["memmap_partial_fit_peak_MB"]:.0f} MB, about {bm["memory_saving_factor"]:.0f}x less than full SVD with the data in RAM ({bm["full_svd_total_RAM_MB"]:.0f} MB).
* How memory is measured: peak NumPy/Python heap allocations during `fit` (`tracemalloc`), plus the training matrix when it is held in RAM. Pages of the memory-mapped file are not counted (the operating system can drop them at any time), and neither are the BLAS library's own small internal buffers.
* The book's recipe, `IncrementalPCA(...).fit(memmap)`, peaked at {bm["memmap_fit_peak_MB"]:.0f} MB. `fit()` validates and copies its whole input first (`copy=True`), so the data ends up in RAM anyway. Use `partial_fit` on slices.

{_md(b.timing_curve.pivot(index="n_train", columns="solver", values="fit_seconds").reset_index(), {"n_train": "{:,}"})}

![](figures/b_solvers.png)
![](figures/b_timing_curve.png)

## C. Kernel PCA and manifold learning (Swiss roll, {c.params["n_samples"]:,} points, 25% held out)

* **Supervised selection** (GridSearchCV over {cm["n_settings"]} kernel/gamma settings, kPCA -> Logistic Regression, training points only): **{c.params["best_kernel"]}, gamma = {cm["best_gamma"]:.4f}**, CV accuracy {cm["best_cv_accuracy"]:.1%} ± {cm["best_cv_std"]:.1%}, held-out accuracy **{cm["test_accuracy_supervised_choice"]:.1%}**. {cm["n_settings_within_one_std"]} settings are within one standard deviation of the best, so the exact gamma is not significant: any RBF gamma in the upper half of the grid works. Logistic Regression on the raw 3D points gets {cm["test_accuracy_raw_3d"]:.1%}, after linear PCA to 2D {cm["test_accuracy_linear_pca_2d"]:.1%}.
* **Unsupervised selection** (lowest 3-fold cross-validated pre-image error, training points only): **{c.params["preimage_best_kernel"]}, gamma = {cm["preimage_best_gamma"]:.4f}**, pre-image MSE {cm["preimage_best_cv_mse"]:.2f}. The same pipeline with this setting reaches only **{cm["test_accuracy_unsupervised_choice"]:.1%}** on the held-out points, close to linear PCA ({cm["test_accuracy_linear_pca_2d"]:.1%}). With such a small gamma the sigmoid kernel tanh(gamma x.x' + 1) is close to linear, so it rebuilds the inputs well but does not unroll the roll. The two rules answer different questions.
* Reproduction check: the book's setting (rbf, gamma = 0.0433, fitted and measured on all 1,000 points, as in the book) gives a pre-image MSE of **{cm["book_setting_preimage_mse_all_points"]:.4f}**; the book prints 32.786. (This in-sample number is not comparable with the cross-validated errors above.)

Top 5 settings by CV accuracy:

{_md(grid_top, {"gamma": "{:.4f}", "cv_accuracy": "{:.3f}", "cv_std": "{:.3f}"})}

Top 5 settings by cross-validated pre-image error:

{_md(pre_top, {"gamma": "{:.4f}", "fit_preimage_mse": "{:.2f}", "cv_preimage_mse": "{:.2f}"})}

Manifold methods on the whole roll (accuracy = 3-fold CV of StandardScaler -> Logistic Regression on the 2D embedding, predicting which half of the roll a point is on):

{_md(c.methods, {"seconds": "{:.2f}", "downstream_accuracy": "{:.3f}", "max_abs_corr_with_t": "{:.3f}"})}

![](figures/c_swiss_roll_kpca.png)
![](figures/c_kpca_selection.png)
![](figures/c_manifold_methods.png)

## D. Visualising MNIST in 2D ({dm["n_samples"]:,} training digits)

kNN accuracy in 2D = 5-fold CV accuracy (± std over folds) of a 5-nearest-neighbours classifier on the 2D coordinates. Nonlinear methods run after PCA to {d.params["pca_variance_before_nonlinear"]:.0%} variance. The maps are fitted on all {dm["n_samples"]:,} points (t-SNE and MDS cannot map new points), so this score describes how well each map separates the digits; it is not a held-out classifier accuracy. Reference: the same kNN on all 784 pixels scores {dm["knn_accuracy_784d"]:.1%} ± {dm["knn_std_784d"]:.1%}.

{_md(d.table, {"seconds": "{:.1f}", "knn_accuracy_2d": "{:.3f}", "knn_std": "{:.3f}"})}

* **t-SNE separates the digits best ({dm["tsne_knn_accuracy_2d"]:.1%} kNN accuracy in 2D vs {dm["pca_knn_accuracy_2d"]:.1%} for PCA)**, on a par with a kNN on all 784 pixels ({dm["knn_accuracy_784d"]:.1%}). This flatters t-SNE a little: the map was built with every point, including the ones each fold tests on.
* Running PCA first costs nothing: t-SNE on the raw pixels scores {dm["tsne_raw_knn_accuracy_2d"]:.1%} in {dm["tsne_raw_seconds"]:.1f} s, against {dm["tsne_knn_accuracy_2d"]:.1%} in {dm["tsne_seconds"]:.1f} s after PCA.

![](figures/d_tsne_map.png)
![](figures/d_maps_comparison.png)

## E. Does compression speed up classification? (exercise 9)

The PCA (95% of the variance) is fitted on all 60,000 training images (it is unsupervised) and its fit time is included. Timings are medians of {e.params["timing_repeats"]} runs (one run for fits slower than 60 s). Accuracy intervals are 95% t-intervals on the per-image correctness of the 10,000 test images (Geron, Ch. 2); "pca_same_gamma" is the PCA SVM run with the raw model's RBF bandwidth.

{_md(e.table[["classifier", "features", "n_features", "n_train", "fit_seconds", "fit_runs", "predict_seconds_10k", "test_accuracy", "accuracy_ci_low", "accuracy_ci_high", "model_MB"]], {"n_train": "{:,}", "fit_seconds": "{:.1f}", "predict_seconds_10k": "{:.2f}", "test_accuracy": "{:.4f}", "accuracy_ci_low": "{:.4f}", "accuracy_ci_high": "{:.4f}", "model_MB": "{:.1f}"})}

"""
        + "\n".join(
            f"* **{k.replace('_', ' ')}**: training {em[f'{k}_fit_speedup']:.2f}x and prediction {em[f'{k}_predict_speedup']:.2f}x faster with PCA ({'>1 = faster' if em[f'{k}_fit_speedup'] >= 1 else '<1 = slower'}); accuracy {em[f'{k}_accuracy_change_pp']:+.2f} pp (paired 95% CI {em[f'{k}_accuracy_change_ci_low_pp']:+.2f} to {em[f'{k}_accuracy_change_ci_high_pp']:+.2f} pp)."
            for k in e.table.kind.unique()
        )
        + f"""
* **SVM bandwidth ablation:** with the raw model's gamma, the PCA SVM scores {em.get("svm_pca_same_gamma_accuracy", float("nan")):.2%} ({em.get("svm_same_gamma_change_pp", float("nan")):+.2f} pp vs raw, 95% CI {em.get("svm_same_gamma_change_ci_low_pp", float("nan")):+.2f} to {em.get("svm_same_gamma_change_ci_high_pp", float("nan")):+.2f} pp). Most of the PCA model's accuracy gain therefore comes from `gamma="scale"` choosing a different kernel width on the PCA features; the part due to dropping the 630 low-variance directions alone is small and only just significant.
* The deployed pair is the **{serve}**: raw pixels {em["served_raw_accuracy"]:.2%}, PCA features {em["served_pca_accuracy"]:.2%}.

![](figures/e_classification.png)

## F. Flagging unusual inputs by reconstruction error

PCA with {fm["n_components"]} components; threshold = the {100 - 100 * f.params["flag_rate"]:.0f}th percentile of reconstruction errors on {f.params["calibration_size"]:,} held-out training images = **{fm["threshold"]:.1f}**.

* False alarms on the 10,000 clean test digits: **{fm["clean_false_alarm_pct"]:.2f}%** (95% CI {fm["clean_false_alarm_ci_low_pct"]:.2f}-{fm["clean_false_alarm_ci_high_pct"]:.2f}%; target {100 * f.params["flag_rate"]:.0f}%, the book's rule of thumb, not a cost-based choice). With the threshold set on the fitting images instead: {fm["clean_false_alarm_pct_train_threshold"]:.2f}%. With 50,000 fitting images PCA hardly overfits, so the two thresholds are close ({fm["threshold"]:.0f} vs {fm["train_threshold"]:.0f}). Both give fewer than 4% false alarms because MNIST's test digits are rebuilt slightly better than its training digits (test MSE {am["test_mse_95"]:.1f} vs training {am["train_mse_95"]:.1f}).
* Mean detection over 8 synthetic corruption types: {fm["mean_detection_pct"]:.1f}%; ROC-AUC clean vs corrupted: {fm["roc_auc"]:.3f}. These averages depend on which corruptions are included; the per-type rates below are the informative numbers. No real out-of-distribution data (letters, other image sets) is tested.
* Served classifier ({serve}, PCA features) on clean test digits: error rate **{extra.get("error_rate_flagged_pct", float("nan")):.1f}%** on the {extra.get("n_flagged_clean", 0)} flagged images vs **{extra.get("error_rate_unflagged_pct", float("nan")):.1f}%** on the rest ({extra.get("n_errors_flagged", 0)} of its {extra.get("n_errors_total", 0)} errors are flagged).

{_md(f.corruption_table, {"flagged_pct": "{:.1f}%", "median_error": "{:,.0f}"})}

Threshold sweep:

{_md(f.threshold_sweep[["flag_rate_pct", "threshold", "clean_false_alarm_pct", "mean_detection_pct"]], {"threshold": "{:.1f}", "clean_false_alarm_pct": "{:.2f}", "mean_detection_pct": "{:.1f}"})}

How many components should the detector keep?

{_md(f.variance_sweep, {"variance": "{:.0%}", "clean_false_alarm_pct": "{:.2f}", "mean_detection_pct": "{:.1f}", "roc_auc": "{:.3f}"})}

![](figures/f_error_histograms.png)
![](figures/f_corruption_examples.png)
![](figures/f_detection.png)
"""
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def split_sizes(run_info: dict) -> dict:
    return run_info.get("split_sizes", {"n_train": 0, "n_test": 0})
