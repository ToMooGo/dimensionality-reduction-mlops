"""Shared fixtures. Unit tests run offline on 'MNIST-like' images made from scikit-learn's bundled
8x8 digits (upscaled to 28x28, 0-255); only the integration test downloads MNIST."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "services" / "api"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dimred_mlops.data import MnistSplit  # noqa: E402


def mnist_like() -> tuple[np.ndarray, np.ndarray]:
    """1,797 digits as 28x28 images: each 8x8 pixel becomes a 3x3 block, plus a 2-pixel border."""
    from sklearn.datasets import load_digits

    d = load_digits()
    imgs = np.kron(d.images, np.ones((3, 3))) * (255.0 / 16.0)
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)))
    return imgs.reshape(len(imgs), -1).astype(np.float32), d.target.astype(np.int64)


@pytest.fixture(scope="session")
def split() -> MnistSplit:
    X, y = mnist_like()
    return MnistSplit(X[:1400], y[:1400], X[1400:], y[1400:])
