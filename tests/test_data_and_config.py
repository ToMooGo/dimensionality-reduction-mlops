import gzip
import hashlib
import struct

import numpy as np
import pytest

from dimred_mlops import data
from dimred_mlops.config import deep_update, linspace, load_config
from dimred_mlops.data import CORRUPTIONS, corrupt_digits, swiss_roll


def _write_idx(path, arr):
    header = struct.pack(">BBBB", 0, 0, 0x08, arr.ndim) + struct.pack(
        ">" + "I" * arr.ndim, *arr.shape
    )
    with gzip.open(path, "wb") as fh:
        fh.write(header + arr.astype(np.uint8).tobytes())


def test_read_idx_roundtrip(tmp_path):
    arr = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    _write_idx(tmp_path / "x.gz", arr)
    np.testing.assert_array_equal(data._read_idx(tmp_path / "x.gz"), arr)


def test_mirror_rejects_files_with_wrong_checksum(tmp_path, monkeypatch):
    # a file that is present but is not the official MNIST must never be used
    fake = {k: (fname, "0" * 32) for k, (fname, _) in data.MNIST_FILES.items()}
    monkeypatch.setattr(data, "MNIST_FILES", fake)
    monkeypatch.setattr(data, "_download", lambda url, dst: _write_idx(dst, np.zeros((1, 1))))
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        data._from_mirror(tmp_path)


def test_md5_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"pca" * 1000)
    assert data._md5(p) == hashlib.md5(b"pca" * 1000).hexdigest()


def test_split_follows_the_book_and_subsamples(monkeypatch):
    X = np.tile(np.arange(70_000, dtype=np.uint8)[:, None], (1, 784))
    y = (np.arange(70_000) % 10).astype(np.uint8)
    monkeypatch.setattr(data, "load_mnist", lambda cache_dir="data": (X, y))
    s = data.load_mnist_split()
    assert s.sizes == {"n_train": 60_000, "n_test": 10_000, "n_features": 784}
    np.testing.assert_array_equal(s.y_test[:3], [0, 1, 2])  # images 60,000.. are the test set
    small = data.load_mnist_split(n_train=500, n_test=100, seed=1)
    assert small.X_train.shape == (500, 784) and small.X_test.shape == (100, 784)
    assert small.X_train.dtype == np.float32


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_corruptions_keep_shape_and_range(split, kind):
    out = corrupt_digits(split.X_test[:5], kind, seed=0)
    assert out.shape == (5, 784)
    assert out.min() >= 0 and out.max() <= 255
    assert not np.allclose(out, split.X_test[:5])


def test_unknown_corruption_raises(split):
    with pytest.raises(ValueError):
        corrupt_digits(split.X_test[:1], "blurred")


def test_swiss_roll_labels():
    X, t, y = swiss_roll(300, 0.2, 0)
    assert X.shape == (300, 3)
    np.testing.assert_array_equal(y, (t > 6.9).astype(int))


def test_config_env_and_extends(tmp_path, monkeypatch):
    (tmp_path / "base.yaml").write_text("a: 1\nb:\n  c: 2\n  d: ${MY_VAR:5}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nb: {c: 3}\n")
    monkeypatch.setenv("MY_VAR", "7")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"a": 1, "b": {"c": 3, "d": 7}}
    assert deep_update({"x": {"y": 1}}, {"x": {"z": 2}}) == {"x": {"y": 1, "z": 2}}


def test_linspace_spec_gives_the_books_grid():
    g = linspace({"start": 0.03, "stop": 0.05, "num": 10})
    assert len(g) == 10 and g[0] == pytest.approx(0.03) and g[-1] == pytest.approx(0.05)
    assert linspace([0.1, 0.2]) == [0.1, 0.2]


def test_shipped_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    full = load_config(root / "full_flow_config.yaml")
    quick = load_config(root / "quick_flow_config.yaml")
    mon = load_config(root / "monitor_flow_config.yaml")
    assert full["part_b"]["n_components"] == 154 and full["part_b"]["n_batches"] == 100
    assert len(linspace(full["part_c"]["gammas"])) * len(full["part_c"]["kernels"]) == 20
    assert quick["data"]["n_train"] < 60_000 and quick["part_a"] == full["part_a"]
    assert mon["flow"] == "monitor" and "max_unusual_rate_pct" in mon["monitor"]
