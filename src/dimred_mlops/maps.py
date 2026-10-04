"""Part D - visualising MNIST in 2D: t-SNE versus PCA, Kernel PCA, LLE, Isomap and MDS
(Geron, Ch. 8, exercise 10).

* Pixels are scaled to [0, 1].
* The nonlinear methods run after a PCA that keeps 95% of the variance. This is the book's
  exercise 8: chaining a fast linear reduction before a slow nonlinear one. t-SNE is also run on
  the raw pixels, as a control that the chaining costs nothing.
* A map is scored by the 5-fold cross-validated accuracy of a k-nearest-neighbours classifier
  in 2D. This is the book's answer to exercise 7: judge a reduction by the algorithm that uses
  it. A high score means each digit forms its own region.

The deployed :class:`EmbeddingAtlas` holds a sample of the maps for the web UI. It also holds the
fitted PCA, Kernel PCA and LLE reducers, which can place a *new* digit on their maps. t-SNE and MDS
have no ``transform`` method, so they cannot. Isomap can, but only by keeping its geodesic
distance matrix (sample size squared), so it is shown without new points.
"""

from __future__ import annotations

import base64
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .data import MAX_PIXEL, MnistSplit

PROJECTABLE = ("PCA", "Kernel PCA", "LLE")


def map_methods(cfg: dict, seed: int) -> dict:
    """name -> unfitted reducer taking pixels in [0, 1] (n x 784) and returning n x 2."""
    k = int(cfg.get("n_neighbors", 10))
    var = float(cfg.get("pca_variance", 0.95))

    def chain(step):
        return Pipeline([("pca", PCA(n_components=var, random_state=seed)), ("reduce", step)])

    return {
        "PCA": PCA(n_components=2, random_state=seed),
        "Kernel PCA": chain(
            KernelPCA(n_components=2, kernel="rbf", gamma=cfg.get("kpca_gamma"), random_state=seed)
        ),
        "LLE": chain(
            LocallyLinearEmbedding(
                n_components=2, n_neighbors=k, neighbors_algorithm="brute", random_state=seed
            )
        ),
        "Isomap": chain(Isomap(n_components=2, n_neighbors=k)),
        "MDS": chain(MDS(n_components=2, n_init=1, random_state=seed)),
        "t-SNE": chain(
            TSNE(n_components=2, perplexity=float(cfg.get("perplexity", 30)), random_state=seed)
        ),
        "t-SNE (raw pixels)": TSNE(
            n_components=2, perplexity=float(cfg.get("perplexity", 30)), random_state=seed
        ),
    }


def knn_scores(Z: np.ndarray, y: np.ndarray, n_neighbors: int = 5, cv: int = 5) -> np.ndarray:
    """Per-fold accuracies of a k-nearest-neighbours classifier on the coordinates ``Z``."""
    return cross_val_score(KNeighborsClassifier(n_neighbors), Z, y, cv=cv)


def knn_score(Z: np.ndarray, y: np.ndarray, n_neighbors: int = 5, cv: int = 5) -> float:
    return float(knn_scores(Z, y, n_neighbors, cv).mean())


def stratified_sample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` indices with the class proportions of ``y``, in random order."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        take = round(n * len(members) / len(y))
        idx.append(rng.choice(members, take, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx[:n]


class EmbeddingAtlas(BaseEstimator):
    """2D maps of a sample of training digits, served by the API.

    ``coords[method]`` is n x 2, rescaled to [0, 1] for display; ``project(X)`` places new
    images (pixels 0-255) on the maps of the methods that support it, in the same scale.
    """

    def __init__(
        self,
        coords=None,
        labels=None,
        thumbnails=None,
        reducers=None,
        scores=None,
        samples=None,
        sample_labels=None,
    ):
        self.coords = coords
        self.labels = labels
        self.thumbnails = thumbnails
        self.reducers = reducers
        self.scores = scores
        # a few hundred held-out test digits the web UI can send (the API ships without MNIST)
        self.samples = samples
        self.sample_labels = sample_labels

    def fit(self, X=None, y=None):  # built by build_atlas; nothing to learn here
        return self

    def project(self, X: np.ndarray) -> dict[str, list[list[float]]]:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 784) / MAX_PIXEL
        out = {}
        for name, entry in (self.reducers or {}).items():
            lo, span = np.asarray(entry["lo"]), np.asarray(entry["span"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Z = entry["model"].transform(X)
            out[name] = ((Z - lo) / span).round(4).tolist()
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Position of each input on the PCA map (n x 2, display scale) - the pyfunc interface."""
        return np.asarray(self.project(X)["PCA"])

    def to_payload(self) -> dict:
        """JSON for the web UI: coordinates, labels and base64-encoded 28x28 uint8 thumbnails."""
        return {
            "n_points": len(self.labels),
            "labels": [int(v) for v in self.labels],
            "thumbnails_b64": base64.b64encode(
                np.asarray(self.thumbnails, np.uint8).tobytes()
            ).decode(),
            "methods": [
                {
                    "name": name,
                    "coords": np.asarray(xy).round(4).tolist(),
                    "projects_new_points": name in (self.reducers or {}),
                    **(self.scores or {}).get(name, {}),
                }
                for name, xy in self.coords.items()
            ],
        }


def _normalise(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = Z.min(axis=0)
    span = np.maximum(Z.max(axis=0) - lo, 1e-12)
    return (Z - lo) / span, lo, span


@dataclass
class MapsResult:
    params: dict
    metrics: dict
    table: pd.DataFrame  # one row per method: seconds and kNN accuracy in 2D
    embeddings: dict  # method -> n x 2 (raw coordinates, for the figures)
    labels: np.ndarray
    images: np.ndarray  # n x 784 pixels (0-255) of the mapped sample
    atlas: EmbeddingAtlas
    run_id: str | None = None
    extras: dict = field(default_factory=dict)


def run_part_d(split: MnistSplit, cfg: dict, seed: int = 42) -> MapsResult:
    n = min(int(cfg.get("n_samples", 5000)), len(split.X_train))
    idx = stratified_sample(split.y_train, n, seed)
    images, y = split.X_train[idx], split.y_train[idx]
    X = images.astype(np.float64) / MAX_PIXEL
    knn_k, cv = int(cfg.get("knn_neighbors", 5)), int(cfg.get("cv", 5))
    wanted = cfg.get("methods")

    rows, embeddings, fitted = [], {}, {}
    for name, reducer in map_methods(cfg, seed).items():
        if wanted and name not in wanted:
            continue
        reducer = clone(reducer)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = reducer.fit_transform(X)
        seconds = time.perf_counter() - t0
        embeddings[name], fitted[name] = Z, reducer
        folds = knn_scores(Z, y, knn_k, cv)
        rows.append(
            {
                "method": name,
                "seconds": seconds,
                "knn_accuracy_2d": float(folds.mean()),
                "knn_std": float(folds.std()),
            }
        )
    table = pd.DataFrame(rows)
    # reference: the same kNN on all 784 pixels (no reduction) - how much a 2D map gives up
    folds = knn_scores(X, y, knn_k, cv)
    baseline = {"knn_accuracy_784d": float(folds.mean()), "knn_std_784d": float(folds.std())}

    # The atlas: the first n_atlas points of every map (raw-pixel t-SNE is a control, not shown).
    n_atlas = min(int(cfg.get("n_atlas", 2000)), n)
    coords, reducers, scores = {}, {}, {}
    for name, Z in embeddings.items():
        if name == "t-SNE (raw pixels)":
            continue
        Zn, lo, span = _normalise(Z)
        coords[name] = Zn[:n_atlas]
        r = table.loc[table.method == name].iloc[0]
        scores[name] = {
            "knn_accuracy_2d": round(float(r.knn_accuracy_2d), 4),
            "seconds": round(float(r.seconds), 2),
        }
        if name in PROJECTABLE:
            reducers[name] = {"model": fitted[name], "lo": lo.tolist(), "span": span.tolist()}
    test_idx = np.sort(
        stratified_sample(
            split.y_test, min(int(cfg.get("n_api_samples", 500)), len(split.y_test)), seed
        )
    )
    atlas = EmbeddingAtlas(
        coords=coords,
        labels=y[:n_atlas].astype(int),
        thumbnails=images[:n_atlas].astype(np.uint8),
        reducers=reducers,
        scores=scores,
        samples=split.X_test[test_idx].astype(np.uint8),
        sample_labels=split.y_test[test_idx].astype(int),
    )

    best = table.sort_values("knn_accuracy_2d", ascending=False).iloc[0]

    def get(name, col):
        sel = table.loc[table.method == name, col]
        return float(sel.iloc[0]) if len(sel) else float("nan")

    metrics = {
        "n_samples": n,
        "best_knn_accuracy_2d": float(best.knn_accuracy_2d),
        "tsne_knn_accuracy_2d": get("t-SNE", "knn_accuracy_2d"),
        "tsne_raw_knn_accuracy_2d": get("t-SNE (raw pixels)", "knn_accuracy_2d"),
        "pca_knn_accuracy_2d": get("PCA", "knn_accuracy_2d"),
        **baseline,
        "tsne_seconds": get("t-SNE", "seconds"),
        "tsne_raw_seconds": get("t-SNE (raw pixels)", "seconds"),
    }
    params = {
        "best_method": best.method,
        "n_atlas": n_atlas,
        "knn_neighbors": knn_k,
        "pca_variance_before_nonlinear": float(cfg.get("pca_variance", 0.95)),
        "perplexity": float(cfg.get("perplexity", 30)),
        "n_neighbors": int(cfg.get("n_neighbors", 10)),
    }
    return MapsResult(params, metrics, table, embeddings, y, images, atlas)
