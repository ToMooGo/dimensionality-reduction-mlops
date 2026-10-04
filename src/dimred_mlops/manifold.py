"""Part C - Kernel PCA and manifold learning on the Swiss roll (Geron, Ch. 8).

* Kernel PCA is unsupervised, so the book tunes it in two ways (pp. 229-232):
  1. supervised: the kernel and gamma that give the best accuracy of a downstream classifier
     (``GridSearchCV`` over a ``KernelPCA -> LogisticRegression`` pipeline, 2 x 10 = 20 settings);
  2. unsupervised: the kernel and gamma with the lowest reconstruction pre-image error.
* LLE, Isomap, MDS and t-SNE are compared with PCA and Kernel PCA on the same roll (Fig. 8-13).

The book fits everything on all 1,000 points. Here 25% of the roll is held out: both rules
choose using the other 75% only (the unsupervised rule by 3-fold cross-validation of the
pre-image error), and both choices are then judged on the held-out points.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import linspace
from .data import swiss_roll


def kpca_pipeline(seed: int = 42) -> Pipeline:
    """The book's two-step pipeline: reduce to 2D with kPCA, then Logistic Regression."""
    return Pipeline(
        [
            ("kpca", KernelPCA(n_components=2, random_state=seed)),
            ("log_reg", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def _preimage_mse(kpca: KernelPCA, X_fit: np.ndarray, X_eval: np.ndarray) -> float:
    kpca.fit(X_fit)
    return float(np.mean((X_eval - kpca.inverse_transform(kpca.transform(X_eval))) ** 2))


def preimage_errors(X_train, kernels, gammas, cv: int = 3, seed: int = 42) -> pd.DataFrame:
    """Reconstruction pre-image error (MSE) of every kernel/gamma setting.

    ``fit_inverse_transform=True`` learns the pre-image map with kernel ridge regression (p. 231).
    The error on the fitting points is optimistic, because the regression saw them, so the
    selection uses the error on the left-out fold of a ``cv``-fold split of the *training* points.
    The test points are not used to choose anything.
    """
    from sklearn.model_selection import KFold

    folds = list(KFold(cv, shuffle=True, random_state=seed).split(X_train))
    rows = []
    for kernel in kernels:
        for gamma in gammas:

            def make(kernel=kernel, gamma=gamma):
                return KernelPCA(
                    n_components=2,
                    kernel=kernel,
                    gamma=gamma,
                    fit_inverse_transform=True,
                    random_state=seed,
                )

            rows.append(
                {
                    "kernel": kernel,
                    "gamma": gamma,
                    "fit_preimage_mse": _preimage_mse(make(), X_train, X_train),
                    "cv_preimage_mse": float(
                        np.mean([_preimage_mse(make(), X_train[a], X_train[b]) for a, b in folds])
                    ),
                }
            )
    return pd.DataFrame(rows)


def manifold_methods(cfg: dict, seed: int) -> dict:
    """name -> unfitted 2D reducer (the methods of Fig. 8-13 plus PCA and Kernel PCA)."""
    k = int(cfg.get("n_neighbors", 10))
    return {
        "PCA": PCA(n_components=2, random_state=seed),
        "Kernel PCA (rbf)": KernelPCA(
            n_components=2, kernel="rbf", gamma=float(cfg.get("rbf_gamma", 0.04)), random_state=seed
        ),
        "LLE": LocallyLinearEmbedding(n_components=2, n_neighbors=k, random_state=seed),
        "Isomap": Isomap(n_components=2, n_neighbors=k),
        "MDS": MDS(n_components=2, n_init=1, random_state=seed),
        "t-SNE": TSNE(n_components=2, random_state=seed),
    }


def downstream_accuracy(Z: np.ndarray, y: np.ndarray, cv: int, seed: int) -> float:
    """How well a linear classifier separates the two halves of the roll in the 2D embedding.

    The book's answer to "how do you evaluate a dimensionality reduction?" (ex. 7): measure the
    performance of the algorithm that comes after it.
    """
    # Embeddings come in very different scales (LLE's are tiny), so standardise them first.
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    return float(cross_val_score(clf, Z, y, cv=cv).mean())


@dataclass
class ManifoldResult:
    params: dict
    metrics: dict
    grid: pd.DataFrame  # supervised grid search: one row per kernel/gamma
    preimage: pd.DataFrame  # unsupervised selection: one row per kernel/gamma
    methods: pd.DataFrame  # LLE, Isomap, MDS, t-SNE, ... on the roll
    embeddings: dict  # method name -> 2D coordinates of the whole roll (for the figures)
    roll: dict  # X, t, y of the Swiss roll
    run_id: str | None = None
    extras: dict = field(default_factory=dict)


def run_part_c(cfg: dict, seed: int = 42) -> ManifoldResult:
    X, t, y = swiss_roll(
        int(cfg.get("n_samples", 1000)),
        float(cfg.get("noise", 0.2)),
        seed,
        float(cfg.get("threshold", 6.9)),
    )
    idx_train, idx_test = train_test_split(
        np.arange(len(X)),
        test_size=float(cfg.get("test_size", 0.25)),
        random_state=seed,
        stratify=y,
    )
    X_train, X_test, y_train, y_test = X[idx_train], X[idx_test], y[idx_train], y[idx_test]
    kernels = list(cfg.get("kernels", ["rbf", "sigmoid"]))
    gammas = linspace(cfg.get("gammas", {"start": 0.03, "stop": 0.05, "num": 10}))
    cv = int(cfg.get("cv", 3))

    # 1. Supervised selection (the book's GridSearchCV), then a held-out check.
    grid_search = GridSearchCV(
        kpca_pipeline(seed),
        [{"kpca__gamma": gammas, "kpca__kernel": kernels}],
        cv=cv,
        n_jobs=1,
    )
    grid_search.fit(X_train, y_train)
    grid = pd.DataFrame(
        {
            "kernel": grid_search.cv_results_["param_kpca__kernel"].astype(str),
            "gamma": grid_search.cv_results_["param_kpca__gamma"].astype(float),
            "cv_accuracy": grid_search.cv_results_["mean_test_score"],
            "cv_std": grid_search.cv_results_["std_test_score"],
        }
    )
    best = grid_search.best_params_
    test_acc_supervised = float(grid_search.score(X_test, y_test))

    # Baseline: Logistic Regression on the raw 3D coordinates, and on linear PCA to 2D.
    raw_acc = float(
        LogisticRegression(max_iter=1000, random_state=seed)
        .fit(X_train, y_train)
        .score(X_test, y_test)
    )
    lin = Pipeline(
        [
            ("pca", PCA(n_components=2)),
            ("log_reg", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    linear_acc = float(lin.fit(X_train, y_train).score(X_test, y_test))

    # 2. Unsupervised selection: lowest held-out pre-image error.
    pre = preimage_errors(X_train, kernels, gammas, cv, seed)
    pre_best = pre.loc[pre.cv_preimage_mse.idxmin()]
    unsup = kpca_pipeline(seed).set_params(
        kpca__kernel=pre_best.kernel, kpca__gamma=float(pre_best.gamma)
    )
    test_acc_unsupervised = float(unsup.fit(X_train, y_train).score(X_test, y_test))
    book = KernelPCA(
        n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True, random_state=seed
    )
    book_err = float(np.mean((X - book.inverse_transform(book.fit_transform(X))) ** 2))

    # 3. Manifold learning methods on the whole roll (unsupervised; y is only used to score).
    rows, embeddings = [], {}
    for name, reducer in manifold_methods(cfg, seed).items():
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = reducer.fit_transform(X)
        seconds = time.perf_counter() - t0
        embeddings[name] = Z
        rows.append(
            {
                "method": name,
                "seconds": seconds,
                "downstream_accuracy": downstream_accuracy(Z, y, cv, seed),
                # unrolling check: |correlation| between the best 2D axis and the position t
                "max_abs_corr_with_t": float(
                    max(abs(np.corrcoef(Z[:, j], t)[0, 1]) for j in range(Z.shape[1]))
                ),
            }
        )
    methods = pd.DataFrame(rows)

    metrics = {
        "best_gamma": float(best["kpca__gamma"]),
        "best_cv_accuracy": float(grid_search.best_score_),
        "best_cv_std": float(grid.cv_std.iloc[int(np.argmax(grid.cv_accuracy))]),
        # settings whose CV accuracy is within one standard deviation of the best one
        "n_settings_within_one_std": int(
            (
                grid.cv_accuracy
                >= grid_search.best_score_ - grid.cv_std.iloc[int(np.argmax(grid.cv_accuracy))]
            ).sum()
        ),
        "test_accuracy_supervised_choice": test_acc_supervised,
        "test_accuracy_unsupervised_choice": test_acc_unsupervised,
        "test_accuracy_raw_3d": raw_acc,
        "test_accuracy_linear_pca_2d": linear_acc,
        "preimage_best_gamma": float(pre_best.gamma),
        "preimage_best_cv_mse": float(pre_best.cv_preimage_mse),
        "book_setting_preimage_mse_all_points": book_err,
        "n_settings": len(grid),
    }
    params = {
        "best_kernel": best["kpca__kernel"],
        "preimage_best_kernel": pre_best.kernel,
        "n_samples": len(X),
        "noise": float(cfg.get("noise", 0.2)),
        "gammas": [round(g, 5) for g in gammas],
        "kernels": kernels,
        "cv": cv,
    }
    return ManifoldResult(
        params,
        metrics,
        grid,
        pre,
        methods,
        embeddings,
        {"X": X, "t": t, "y": y, "idx_train": idx_train, "idx_test": idx_test},
    )
