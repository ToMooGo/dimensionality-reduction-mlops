"""Part B - scaling PCA: Randomized PCA, Incremental PCA and memory-mapped data (Geron, Ch. 8).

Every solver is asked for the same 154 components of the 60,000 training images. Each is timed
(median of ``repeats`` runs), its peak memory is traced in a separate run, and its components are
checked against the exact (full SVD) answer through the test-set reconstruction error.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA

from .compression import reconstruction_mse
from .data import MnistSplit


def write_memmap(X: np.ndarray, directory: str | Path) -> tuple[Path, tuple[int, int]]:
    """Store ``X`` as a raw float32 binary file that ``np.memmap`` can open (the book's setup)."""
    path = Path(directory) / "mnist_train.float32"
    mm = np.memmap(path, dtype="float32", mode="w+", shape=X.shape)
    mm[:] = X
    mm.flush()
    del mm
    return path, X.shape


def open_memmap(path: Path, shape: tuple[int, int]) -> np.memmap:
    return np.memmap(path, dtype="float32", mode="r", shape=shape)


def ipca_partial_fit(X, n_components: int, n_batches: int) -> IncrementalPCA:
    """The book's loop: one ``partial_fit`` call per mini-batch.

    Works for in-memory arrays and memmaps alike. With a memmap, only one slice of the file is
    read into memory at a time.
    """
    ipca = IncrementalPCA(n_components=n_components)
    n_batches = effective_batches(len(X), n_components, n_batches)
    bounds = np.linspace(0, len(X), n_batches + 1).astype(int)
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        ipca.partial_fit(np.asarray(X[lo:hi]))
    return ipca


def effective_batches(m: int, n_components: int, n_batches: int) -> int:
    """Every mini-batch needs at least ``n_components`` rows; small subsets use fewer batches."""
    return max(1, min(n_batches, m // n_components))


def solver_factories(n_components: int, n_batches: int, seed: int) -> dict:
    """name -> (fit function taking the data, data kind: 'memory' or 'memmap')."""
    return {
        "PCA, full SVD": (
            lambda X: PCA(n_components, svd_solver="full", random_state=seed).fit(X),
            "memory",
        ),
        "PCA, solver='auto'": (
            lambda X: PCA(n_components, svd_solver="auto", random_state=seed).fit(X),
            "memory",
        ),
        "Randomized PCA": (
            lambda X: PCA(n_components, svd_solver="randomized", random_state=seed).fit(X),
            "memory",
        ),
        "Incremental PCA, partial_fit": (
            lambda X: ipca_partial_fit(X, n_components, n_batches),
            "memory",
        ),
        "Incremental PCA, memmap + fit()": (
            lambda X: IncrementalPCA(
                n_components,
                batch_size=len(X) // effective_batches(len(X), n_components, n_batches),
            ).fit(X),
            "memmap",
        ),
        "Incremental PCA, memmap + partial_fit": (
            lambda X: ipca_partial_fit(X, n_components, n_batches),
            "memmap",
        ),
    }


def _timed(fit, X) -> tuple[object, float]:
    gc.collect()
    t0 = time.perf_counter()
    model = fit(X)
    return model, time.perf_counter() - t0


def _peak_mb(fit, X) -> float:
    """Peak memory allocated while fitting (NumPy reports its buffers to ``tracemalloc``).

    Memory that already holds the data before the call is not counted.
    """
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    fit(X)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def solver_name(model) -> str:
    return getattr(model, "_fit_svd_solver", "incremental")


@dataclass
class ScalingResult:
    params: dict
    metrics: dict
    solvers: pd.DataFrame  # one row per solver on the full training set
    timing_curve: pd.DataFrame  # fit time vs number of training images
    run_id: str | None = None
    extras: dict = field(default_factory=dict)


def run_part_b(split: MnistSplit, cfg: dict, seed: int = 42) -> ScalingResult:
    d = int(cfg.get("n_components", 154))
    n_batches = int(cfg.get("n_batches", 100))
    repeats = int(cfg.get("repeats", 3))
    X, X_test = split.X_train, split.X_test
    data_mb = X.nbytes / 1e6
    factories = solver_factories(d, n_batches, seed)

    # warm-up: the first call of each solver pays one-off costs (BLAS threads, imports)
    for fit, kind in factories.values():
        if kind == "memory":
            fit(X[: max(2 * d, 1000)])

    tmp = Path(tempfile.mkdtemp(prefix="memmap-", dir=cfg.get("memmap_dir")))
    rows, reference_mse = [], None
    try:
        path, shape = write_memmap(X, tmp)
        for name, (fit, kind) in factories.items():
            data = open_memmap(path, shape) if kind == "memmap" else X
            times = []
            model = None
            for _ in range(repeats):
                model, t = _timed(fit, data)
                times.append(t)
                if t > float(cfg.get("single_run_above_s", 30)):
                    break  # slow solvers: one timed run is enough
            peak = _peak_mb(fit, open_memmap(path, shape) if kind == "memmap" else X)
            mse = float(
                reconstruction_mse(X_test, model.inverse_transform(model.transform(X_test))).mean()
            )
            if reference_mse is None:
                reference_mse = mse
            rows.append(
                {
                    "solver": name,
                    "algorithm": solver_name(model),
                    "data": "in RAM" if kind == "memory" else "on disk (memmap)",
                    "fit_seconds": float(np.median(times)),
                    "runs": len(times),
                    "peak_fit_memory_MB": peak,
                    # in-RAM solvers also need the training matrix itself in memory
                    "total_RAM_MB": peak + (data_mb if kind == "memory" else 0.0),
                    "explained_variance": float(model.explained_variance_ratio_.sum()),
                    "test_mse": mse,
                    "mse_vs_exact_pct": 100 * (mse / reference_mse - 1),
                }
            )
            del data
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    solvers = pd.DataFrame(rows)

    # Fit time vs training-set size for the in-memory solvers.
    curve_rows = []
    for m in cfg.get("sample_sizes", [5000, 10000, 20000, 40000, 60000]):
        m = min(int(m), len(X))
        Xm = X[:m]
        for name in cfg.get(
            "curve_solvers",
            [
                "PCA, full SVD",
                "PCA, solver='auto'",
                "Randomized PCA",
                "Incremental PCA, partial_fit",
            ],
        ):
            fit = factories[name][0]
            runs = [_timed(fit, Xm) for _ in range(max(1, repeats if m <= 20000 else 1))]
            curve_rows.append(
                {
                    "n_train": m,
                    "solver": name,
                    # "auto" switches algorithm with the data shape, so record what it ran
                    "algorithm": solver_name(runs[0][0]),
                    "fit_seconds": float(np.median([t for _, t in runs])),
                }
            )
    curve = pd.DataFrame(curve_rows).drop_duplicates(["n_train", "solver"])

    def pick(name, col):
        return float(solvers.loc[solvers.solver == name, col].iloc[0])

    book_mm = "Incremental PCA, memmap + fit()"
    ooc = "Incremental PCA, memmap + partial_fit"
    metrics = {
        "data_MB": data_mb,
        "full_svd_seconds": pick("PCA, full SVD", "fit_seconds"),
        "auto_seconds": pick("PCA, solver='auto'", "fit_seconds"),
        "randomized_seconds": pick("Randomized PCA", "fit_seconds"),
        "ipca_seconds": pick("Incremental PCA, partial_fit", "fit_seconds"),
        "memmap_fit_peak_MB": pick(book_mm, "peak_fit_memory_MB"),
        "memmap_partial_fit_peak_MB": pick(ooc, "peak_fit_memory_MB"),
        "memmap_partial_fit_seconds": pick(ooc, "fit_seconds"),
        "full_svd_total_RAM_MB": pick("PCA, full SVD", "total_RAM_MB"),
        "randomized_mse_gap_pct": pick("Randomized PCA", "mse_vs_exact_pct"),
        "ipca_mse_gap_pct": pick("Incremental PCA, partial_fit", "mse_vs_exact_pct"),
        "memory_saving_factor": pick("PCA, full SVD", "total_RAM_MB")
        / pick(ooc, "peak_fit_memory_MB"),
    }
    params = {
        "n_components": d,
        "n_batches": n_batches,
        "repeats": repeats,
        "auto_solver_chosen": solvers.loc[solvers.solver == "PCA, solver='auto'", "algorithm"].iloc[
            0
        ],
    }
    return ScalingResult(params, metrics, solvers, curve)
