"""Part E - does compression speed up classification? (Geron, Ch. 8, exercise 9 and p. 226).

Exercise 9: train a Random Forest on the 784 raw pixels and on the 154 PCA features, and
compare training time and test accuracy. The same comparison is run for Softmax Regression and
for an RBF-kernel SVM. Page 226 says PCA "can speed up a classification algorithm (such as an SVM
classifier) tremendously". The SVM pair is the one deployed behind the API.

* The PCA is unsupervised, so it is fitted on all 60,000 training images (no labels, no test
  data). Its fit time is included in the "PCA" training time.
* Timings are the median of ``timing_repeats`` runs (one run for fits slower than
  ``single_run_above_s``).
* Accuracies come with 95% confidence intervals, and raw vs PCA with a *paired* interval on the
  per-image difference, as in the book's Ch. 2 (see :mod:`dimred_mlops.stats`).
* SVM ablation: ``gamma="scale"`` sets the RBF bandwidth from the variance of the inputs, so the
  raw and PCA models do not use the same kernel width. A third SVM uses the raw model's gamma on
  the PCA features, to separate "fewer, denoised features" from "a different bandwidth".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .data import MAX_PIXEL, MnistSplit
from .stats import mean_ci


def make_classifier(kind: str, cfg: dict, seed: int, gamma="scale"):
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(cfg.get("rf_n_estimators", 100)), random_state=seed, n_jobs=1
        )
    if kind == "softmax":
        return LogisticRegression(
            max_iter=int(cfg.get("softmax_max_iter", 1000)), random_state=seed
        )
    if kind == "svm":
        # gamma="scale" = 1 / (n_features * X.var()): the kernel adapts to the input scale, so raw
        # 0-255 pixels and PCA codes can be used as they are.
        return SVC(kernel="rbf", C=float(cfg.get("svm_C", 1.0)), gamma=gamma, random_state=seed)
    raise ValueError(f"unknown classifier {kind!r}")


def make_pair(kind: str, cfg: dict, seed: int) -> dict[str, Pipeline]:
    """The same classifier on raw pixels and after PCA keeping ``pca_variance`` of the variance."""
    var = float(cfg.get("pca_variance", 0.95))
    return {
        "raw": Pipeline([("clf", make_classifier(kind, cfg, seed))]),
        "pca": Pipeline(
            [
                ("pca", PCA(n_components=var, random_state=seed)),
                ("clf", make_classifier(kind, cfg, seed)),
            ]
        ),
    }


def fit_pair_member(model: Pipeline, X_all: np.ndarray, X_tr: np.ndarray, y_tr: np.ndarray):
    """Fit one member of a pair. The PCA step (unsupervised) sees all training images ``X_all``;
    the classifier sees the labelled subset ``X_tr``."""
    if "pca" in model.named_steps:
        pca = model.named_steps["pca"].fit(X_all)
        model.named_steps["clf"].fit(pca.transform(X_tr), y_tr)
    else:
        model.fit(X_tr, y_tr)
    return model


LABELS = {
    "random_forest": "Random Forest",
    "softmax": "Softmax Regression",
    "svm": "SVM (RBF kernel)",
}


@dataclass
class ClassificationResult:
    params: dict
    metrics: dict
    table: pd.DataFrame  # one row per classifier x representation
    models: dict  # the deployed pair: {"raw": Pipeline, "pca": Pipeline}
    run_id: str | None = None
    extras: dict = field(default_factory=dict)


def _timed_fit(model, X_all, X_tr, y_tr, repeats: int, single_run_above: float):
    times, fitted = [], None
    for _ in range(max(1, repeats)):
        m = clone(model)
        t0 = time.perf_counter()
        fitted = fit_pair_member(m, X_all, X_tr, y_tr)
        times.append(time.perf_counter() - t0)
        if times[-1] > single_run_above:
            break
    return fitted, times


def _timed_predict(model, X, repeats: int, single_run_above: float):
    times, pred = [], None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        pred = model.predict(X)
        times.append(time.perf_counter() - t0)
        if times[-1] > single_run_above:
            break
    return pred, times


def run_part_e(split: MnistSplit, cfg: dict, seed: int = 42) -> ClassificationResult:
    X_test, y_test = split.X_test, split.y_test
    repeats = int(cfg.get("timing_repeats", 1))
    slow = float(cfg.get("single_run_above_s", 60))
    rows, deployed, correct = [], {}, {}
    serve = cfg.get("serve", "svm")
    rng = np.random.default_rng(seed)
    for kind in cfg.get("models", ["softmax", "random_forest", "svm"]):
        n = min(int(cfg.get(f"{kind}_train_size") or len(split.X_train)), len(split.X_train))
        idx = (
            np.sort(rng.choice(len(split.X_train), n, replace=False))
            if n < len(split.X_train)
            else slice(None)
        )
        # Softmax Regression converges far better on pixels scaled to [0, 1].
        scale = MAX_PIXEL if kind == "softmax" else 1.0
        X_all = split.X_train / scale
        X_tr, y_tr, X_te = split.X_train[idx] / scale, split.y_train[idx], X_test / scale
        variants = dict(make_pair(kind, cfg, seed))
        if kind == "svm" and cfg.get("svm_same_gamma_ablation", True):
            # the raw model's bandwidth, 1 / (784 * X.var()), applied to the PCA features
            raw_gamma = 1.0 / (X_tr.shape[1] * float(X_tr.var()))
            variants["pca_same_gamma"] = Pipeline(
                [
                    ("pca", clone(variants["pca"].named_steps["pca"])),
                    ("clf", make_classifier(kind, cfg, seed, gamma=raw_gamma)),
                ]
            )
        for rep, model in variants.items():
            model, fit_times = _timed_fit(model, X_all, X_tr, y_tr, repeats, slow)
            pred, pred_times = _timed_predict(model, X_te, repeats, slow)
            ok = (pred == y_test).astype(float)
            correct[(kind, rep)] = ok
            acc, lo, hi = mean_ci(ok)
            clf = model.named_steps["clf"]
            rows.append(
                {
                    "classifier": LABELS[kind],
                    "kind": kind,
                    "features": rep,
                    "n_features": int(model.named_steps["pca"].n_components_)
                    if "pca" in model.named_steps
                    else X_tr.shape[1],
                    "n_train": len(X_tr),
                    "fit_seconds": float(np.median(fit_times)),
                    "fit_runs": len(fit_times),
                    "predict_seconds_10k": float(np.median(pred_times)) * 10_000 / len(X_test),
                    "test_accuracy": acc,
                    "accuracy_ci_low": lo,
                    "accuracy_ci_high": hi,
                    "svm_gamma": float(clf._gamma) if hasattr(clf, "_gamma") else float("nan"),
                    "model_MB": len(sio.dumps(model)) / 1e6,
                }
            )
            if kind == serve and rep in ("raw", "pca"):
                deployed[rep] = model
    table = pd.DataFrame(rows)

    def val(kind, rep, col):
        return float(table.loc[(table.kind == kind) & (table.features == rep), col].iloc[0])

    metrics = {}
    for kind in table.kind.unique():
        metrics[f"{kind}_raw_accuracy"] = val(kind, "raw", "test_accuracy")
        metrics[f"{kind}_pca_accuracy"] = val(kind, "pca", "test_accuracy")
        metrics[f"{kind}_fit_speedup"] = val(kind, "raw", "fit_seconds") / val(
            kind, "pca", "fit_seconds"
        )
        metrics[f"{kind}_predict_speedup"] = val(kind, "raw", "predict_seconds_10k") / val(
            kind, "pca", "predict_seconds_10k"
        )
        # paired 95% interval on the per-image difference (PCA minus raw), in percentage points
        diff, lo, hi = mean_ci(correct[(kind, "pca")] - correct[(kind, "raw")])
        metrics[f"{kind}_accuracy_change_pp"] = 100 * diff
        metrics[f"{kind}_accuracy_change_ci_low_pp"] = 100 * lo
        metrics[f"{kind}_accuracy_change_ci_high_pp"] = 100 * hi
    if ("svm", "pca_same_gamma") in correct:
        metrics["svm_pca_same_gamma_accuracy"] = val("svm", "pca_same_gamma", "test_accuracy")
        diff, lo, hi = mean_ci(correct[("svm", "pca_same_gamma")] - correct[("svm", "raw")])
        metrics["svm_same_gamma_change_pp"] = 100 * diff
        metrics["svm_same_gamma_change_ci_low_pp"] = 100 * lo
        metrics["svm_same_gamma_change_ci_high_pp"] = 100 * hi
    metrics["served_raw_accuracy"] = val(serve, "raw", "test_accuracy")
    metrics["served_pca_accuracy"] = val(serve, "pca", "test_accuracy")
    params = {
        "served_classifier": serve,
        "pca_variance": float(cfg.get("pca_variance", 0.95)),
        "timing_repeats": repeats,
        "train_sizes": {
            k: int(table.loc[table.kind == k, "n_train"].iloc[0]) for k in table.kind.unique()
        },
    }
    return ClassificationResult(params, metrics, table, deployed)
