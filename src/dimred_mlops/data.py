"""MNIST loading, the book's train/test split, corrupted inputs and the Swiss roll.

MNIST is loaded exactly as in Geron, *Hands-On ML* (Ch. 3): ``fetch_openml("mnist_784")``.
The first 60,000 images are the training set and the last 10,000 the test set (Ch. 8, ex. 9).
If OpenML is unreachable (firewalls, rate limits), the original MNIST files are downloaded
from a public mirror and checked against the official MD5 checksums. The array is the same
70,000 x 784 matrix in the same order. The result is cached in ``data/mnist_784.npz``.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

N_TRAIN = 60_000
IMAGE_SHAPE = (28, 28)
N_PIXELS = 784
MAX_PIXEL = 255.0

# Official MNIST files and their MD5 checksums (the same files OpenML's mnist_784 was built from).
MNIST_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}
MIRRORS = ("https://raw.githubusercontent.com/fgnt/mnist/master/",)


@dataclass
class MnistSplit:
    """Pixels are float32 in [0, 255]; labels are int64 digits."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def sizes(self) -> dict:
        return {"n_train": len(self.X_train), "n_test": len(self.X_test), "n_features": N_PIXELS}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        _, _, dtype_code, ndim = struct.unpack(">BBBB", fh.read(4))
        if dtype_code != 0x08:
            raise ValueError(f"{path.name}: expected unsigned bytes")
        shape = struct.unpack(">" + "I" * ndim, fh.read(4 * ndim))
        return np.frombuffer(fh.read(), dtype=np.uint8).reshape(shape)


def _download(url: str, dst: Path, timeout: float = 60.0) -> None:
    """Download to a temporary file and rename it, so a broken transfer never leaves a partial
    file behind."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    os.replace(tmp, dst)


def _from_mirror(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, (fname, md5) in MNIST_FILES.items():
        dst = raw_dir / fname
        if not dst.exists() or _md5(dst) != md5:
            last = None
            for base in MIRRORS:
                try:
                    log.info("Downloading %s from %s", fname, base)
                    _download(base + fname, dst)
                    break
                except OSError as exc:
                    last = exc
            else:
                raise RuntimeError(f"Could not download {fname}: {last}")
        if _md5(dst) != md5:
            raise RuntimeError(f"Checksum mismatch for {fname}: the file is not the official MNIST")
        arrays[key] = _read_idx(dst)
    X = np.concatenate([arrays["train_images"], arrays["test_images"]]).reshape(-1, N_PIXELS)
    y = np.concatenate([arrays["train_labels"], arrays["test_labels"]])
    return X, y


def _from_openml(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, data_home=str(cache_dir))
    return mnist["data"].astype(np.uint8), mnist["target"].astype(np.uint8)


def load_mnist(
    cache_dir: str | Path = "data", source: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``X`` (70,000 x 784, uint8) and ``y`` (70,000, uint8) in the order of mnist_784.

    ``source``: ``auto`` (cache -> OpenML -> mirror), ``openml`` or ``mirror``.
    Defaults to the ``MNIST_SOURCE`` environment variable, else ``auto``.
    """
    source = (source or os.getenv("MNIST_SOURCE", "auto")).lower()
    cache_dir = Path(cache_dir)
    cache = cache_dir / "mnist_784.npz"
    if cache.exists():
        with np.load(cache) as f:
            return f["X"], f["y"]
    X = y = None
    if source in ("auto", "openml"):
        try:
            X, y = _from_openml(cache_dir)
        except Exception as exc:  # network, rate limit, proxy...
            if source == "openml":
                raise
            log.warning("OpenML unavailable (%s) - using the checksum-verified mirror", exc)
    if X is None:
        X, y = _from_mirror(cache_dir / "mnist_raw")
    if X.shape != (70_000, N_PIXELS) or y.shape != (70_000,):
        raise RuntimeError(f"Unexpected MNIST shape {X.shape}, {y.shape}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".part.npz")
    np.savez_compressed(tmp, X=X, y=y)
    os.replace(tmp, cache)  # atomic: concurrent readers never see a half-written cache
    return X, y


def load_mnist_split(
    cache_dir: str | Path = "data",
    n_train: int | None = None,
    n_test: int | None = None,
    seed: int = 42,
) -> MnistSplit:
    """The book's split: first 60,000 images for training, last 10,000 for testing.

    ``n_train`` / ``n_test`` take a stratified random subset (used by the quick config and tests).
    """
    X, y = load_mnist(cache_dir)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_train, y_train, X_test, y_test = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]
    rng = np.random.default_rng(seed)
    if n_train and n_train < len(X_train):
        idx = np.sort(rng.choice(len(X_train), n_train, replace=False))
        X_train, y_train = X_train[idx], y_train[idx]
    if n_test and n_test < len(X_test):
        idx = np.sort(rng.choice(len(X_test), n_test, replace=False))
        X_test, y_test = X_test[idx], y_test[idx]
    return MnistSplit(X_train, y_train, X_test, y_test)


# --------------------------------------------------------------------------------------------
# Corrupted inputs: things a digit service may receive that are not handwritten digits.
# --------------------------------------------------------------------------------------------
CORRUPTIONS = (
    "noise",
    "inverted",
    "shuffled",
    "rotated",
    "mirrored",
    "shifted",
    "gaussian",
    "dimmed",
)


def corrupt_digits(X: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """Return corrupted copies of 28x28 digits (rows of ``X``), pixel range [0, 255]."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    imgs = X.reshape(-1, *IMAGE_SHAPE)
    if kind == "noise":  # uniform random pixels: not a digit at all
        out = rng.uniform(0, MAX_PIXEL, size=imgs.shape)
    elif kind == "inverted":  # white digit background
        out = MAX_PIXEL - imgs
    elif kind == "shuffled":  # same pixel values, random positions
        out = np.stack([rng.permutation(im.ravel()).reshape(IMAGE_SHAPE) for im in imgs])
    elif kind == "rotated":  # upside down
        out = imgs[:, ::-1, ::-1]
    elif kind == "mirrored":  # left-right flip
        out = imgs[:, :, ::-1]
    elif kind == "shifted":  # moved 4 pixels right and down (MNIST digits are centred)
        out = np.zeros_like(imgs)
        out[:, 4:, 4:] = imgs[:, :-4, :-4]
    elif kind == "gaussian":  # sensor noise on top of a real digit
        out = imgs + rng.normal(0, 0.3 * MAX_PIXEL, size=imgs.shape)
    elif kind == "dimmed":  # faint scan
        out = imgs * 0.3
    else:
        raise ValueError(f"unknown corruption {kind!r}; choose from {CORRUPTIONS}")
    return np.clip(out, 0, MAX_PIXEL).reshape(len(X), -1).astype(np.float32)


def swiss_roll(n_samples: int = 1000, noise: float = 0.2, seed: int = 42, threshold: float = 6.9):
    """The book's Swiss roll (Ch. 8): ``X`` (n x 3), position ``t`` along the roll and the
    binary target ``y = t > 6.9`` used to tune Kernel PCA by grid search."""
    from sklearn.datasets import make_swiss_roll

    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    return X, t, (t > threshold).astype(int)
