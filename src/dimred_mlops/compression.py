"""Part A - PCA for compression (Geron, Ch. 8, "PCA for Compression").

How many principal components keep 95% of MNIST's variance, how much smaller is the data, and
how close are the decompressed images to the originals?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .data import N_PIXELS, MnistSplit


def n_components_for(cumulative: np.ndarray, variance: float) -> int:
    """Smallest d whose first d components explain at least ``variance`` (the book's recipe)."""
    return int(np.argmax(cumulative >= variance) + 1)


def reconstruct(pca: PCA, X: np.ndarray, d: int | None = None) -> np.ndarray:
    """Compress ``X`` to its first ``d`` principal components and decompress it (Eq. 8-2, 8-3).

    With ``d`` smaller than ``pca.n_components_`` the remaining components are simply dropped, so
    one PCA fitted with all components can serve every compression level.
    """
    W = pca.components_ if d is None else pca.components_[:d]
    Z = (X - pca.mean_) @ W.T
    return Z @ W + pca.mean_


def reconstruction_mse(X: np.ndarray, X_rec: np.ndarray) -> np.ndarray:
    """Mean squared error per image, in squared pixel units (pixels are 0-255)."""
    return np.mean((np.asarray(X, dtype=np.float64) - X_rec) ** 2, axis=1)


def storage_table(n_components: int, n_images: int) -> pd.DataFrame:
    """Bytes per image and for the whole set, for the raw pixels and the PCA codes.

    The book's "less than 20% of its original size" compares feature counts (float to float).
    Raw MNIST is stored as one byte per pixel, so the saving in bytes depends on the code's dtype.
    """
    rows = [
        ("raw pixels, uint8 (how MNIST is shipped)", N_PIXELS, 1),
        ("raw pixels, float32", N_PIXELS, 4),
        (f"PCA codes, float32 ({n_components} values)", n_components, 4),
        (f"PCA codes, float16 ({n_components} values)", n_components, 2),
    ]
    out = pd.DataFrame(rows, columns=["representation", "values_per_image", "bytes_per_value"])
    out["bytes_per_image"] = out.values_per_image * out.bytes_per_value
    out["dataset_MB"] = out.bytes_per_image * n_images / 1e6
    out["pct_of_float32_pixels"] = 100 * out.bytes_per_image / (N_PIXELS * 4)
    out["pct_of_uint8_pixels"] = 100 * out.bytes_per_image / N_PIXELS
    return out


@dataclass
class CompressionResult:
    params: dict
    metrics: dict
    cumulative: np.ndarray  # cumulative explained variance ratio, all 784 components
    variance_table: pd.DataFrame  # one row per target variance
    storage: pd.DataFrame
    pca_full: PCA  # all components: serves every compression level in the API
    example_index: np.ndarray  # test images shown in the reconstruction figure
    run_id: str | None = None
    extras: dict = field(default_factory=dict)


def run_part_a(split: MnistSplit, cfg: dict, seed: int = 42) -> CompressionResult:
    target = float(cfg.get("target_variance", 0.95))
    targets = sorted({float(v) for v in cfg.get("variance_targets", [target])} | {target})

    # PCA with every component: explained variance curve (Fig. 8-8) and all compression levels.
    pca_full = PCA(svd_solver="full", random_state=seed).fit(split.X_train)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    # The book's shortcut, PCA(n_components=0.95), must agree with the cumulative-sum recipe.
    d = n_components_for(cumulative, target)
    pca_ratio = PCA(n_components=target, random_state=seed).fit(split.X_train)
    if pca_ratio.n_components_ != d:
        raise RuntimeError(f"PCA({target}) kept {pca_ratio.n_components_} components, expected {d}")

    total_var_test = float(np.var(split.X_test, axis=0).sum())
    rows = []
    for v in targets:
        dv = n_components_for(cumulative, v)
        mse_train = reconstruction_mse(split.X_train, reconstruct(pca_full, split.X_train, dv))
        mse_test = reconstruction_mse(split.X_test, reconstruct(pca_full, split.X_test, dv))
        rows.append(
            {
                "variance_target": v,
                "n_components": dv,
                "pct_of_features": 100 * dv / N_PIXELS,
                "train_explained_variance": float(cumulative[dv - 1]),
                # share of the test set's variance that survives compression -> decompression
                "test_explained_variance": 1
                - float(mse_test.sum()) * N_PIXELS / (len(split.X_test) * total_var_test),
                "train_mse": float(mse_train.mean()),
                "test_mse": float(mse_test.mean()),
                "test_rmse_pixels": float(np.sqrt(mse_test.mean())),
            }
        )
    table = pd.DataFrame(rows)
    row = table.loc[table.variance_target == target].iloc[0]

    # Reconstruction error of the training set = variance of the dropped components (math check).
    dropped = pca_full.explained_variance_[d:].sum() * (len(split.X_train) - 1) / len(split.X_train)
    rng = np.random.default_rng(seed)
    example_index = np.sort(rng.choice(len(split.X_test), 10, replace=False))

    metrics = {
        "n_components_95": d,
        "pct_of_features_95": 100 * d / N_PIXELS,
        "train_explained_variance_95": float(cumulative[d - 1]),
        "test_explained_variance_95": float(row.test_explained_variance),
        "train_mse_95": float(row.train_mse),
        "test_mse_95": float(row.test_mse),
        "test_rmse_pixels_95": float(row.test_rmse_pixels),
        "dropped_variance_per_pixel_95": float(dropped / N_PIXELS),
        "n_components_99": n_components_for(cumulative, 0.99),
        "n_components_50": n_components_for(cumulative, 0.50),
        "explained_variance_pc1": float(pca_full.explained_variance_ratio_[0]),
        "explained_variance_pc2": float(pca_full.explained_variance_ratio_[1]),
    }
    params = {"target_variance": target, "variance_targets": targets, "svd_solver": "full"}
    return CompressionResult(
        params,
        metrics,
        cumulative,
        table,
        storage_table(d, len(split.X_train)),
        pca_full,
        example_index,
    )
