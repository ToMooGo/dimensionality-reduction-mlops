"""Part F - flagging unusual inputs by their PCA reconstruction error.

This combines two ideas from the book:

* the reconstruction error of PCA (Ch. 8, p. 226): images like the training digits are rebuilt
  well from 154 components, and inputs unlike them are not;
* a percentile threshold for anomalies (Ch. 9, p. 266): flag the 4% worst-scoring inputs.

The threshold is set on held-out training images, which the PCA was not fitted on. With 50,000
fitting images, PCA hardly overfits, so a threshold set on the fitting images would give almost
the same false-alarm rate. The held-out version is kept because it stays correct when the training
set is small and the fitting errors are optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from .data import CORRUPTIONS, MnistSplit, corrupt_digits
from .stats import mean_ci


class ReconstructionErrorDetector(BaseEstimator):
    """PCA keeping ``variance`` of the variance; an input is *unusual* when its reconstruction
    error exceeds the ``1 - flag_rate`` quantile of errors on held-out clean digits."""

    def __init__(self, variance=0.95, flag_rate=0.04, calibration_size=10_000, random_state=42):
        self.variance = variance
        self.flag_rate = flag_rate
        self.calibration_size = calibration_size
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(len(X))
        n_cal = int(min(self.calibration_size, len(X) // 4))
        cal, fit = perm[:n_cal], perm[n_cal:]
        self.pca_ = PCA(n_components=self.variance, random_state=self.random_state).fit(X[fit])
        cal_err = self.reconstruction_error(X[cal])
        self.threshold_ = float(np.quantile(cal_err, 1 - self.flag_rate))
        self.calibration_errors_ = np.sort(cal_err).astype(np.float32)
        # The book-style alternative: the same quantile of the errors on the fitting images.
        self.train_threshold_ = float(
            np.quantile(self.reconstruction_error(X[fit]), 1 - self.flag_rate)
        )
        self.n_components_ = int(self.pca_.n_components_)
        return self

    def reconstruction_error(self, X) -> np.ndarray:
        """Mean squared reconstruction error per image (pixels 0-255)."""
        X = np.asarray(X, dtype=np.float64)
        X_rec = self.pca_.inverse_transform(self.pca_.transform(X))
        return np.mean((X - X_rec) ** 2, axis=1)

    def is_unusual(self, X) -> np.ndarray:
        return self.reconstruction_error(X) > self.threshold_

    def predict(self, X) -> np.ndarray:
        """1 = unusual input, 0 = looks like a training digit."""
        return self.is_unusual(X).astype(int)

    def error_percentile(self, X) -> np.ndarray:
        """Share of held-out clean digits (in %) with a smaller error than each input."""
        err = self.reconstruction_error(X)
        return 100 * np.searchsorted(self.calibration_errors_, err) / len(self.calibration_errors_)


def evaluate_on_corruptions(
    detector, X_clean: np.ndarray, seed: int = 0
) -> tuple[pd.DataFrame, dict]:
    """Flag rate on clean test digits (false alarms) and on each kind of corrupted copy."""
    clean_err = detector.reconstruction_error(X_clean)
    rows = [
        {
            "input": "clean test digits",
            "flagged_pct": 100 * float(np.mean(clean_err > detector.threshold_)),
            "median_error": float(np.median(clean_err)),
        }
    ]
    all_err = []
    for i, kind in enumerate(CORRUPTIONS):
        err = detector.reconstruction_error(corrupt_digits(X_clean, kind, seed + i))
        all_err.append(err)
        rows.append(
            {
                "input": kind,
                "flagged_pct": 100 * float(np.mean(err > detector.threshold_)),
                "median_error": float(np.median(err)),
            }
        )
    table = pd.DataFrame(rows)
    corrupted = np.concatenate(all_err)
    y_true = np.r_[np.zeros(len(clean_err)), np.ones(len(corrupted))]
    summary = {
        "clean_false_alarm_pct": float(table.flagged_pct.iloc[0]),
        "mean_detection_pct": float(table.flagged_pct.iloc[1:].mean()),
        "roc_auc": float(roc_auc_score(y_true, np.r_[clean_err, corrupted])),
    }
    return table, summary


@dataclass
class DetectorResult:
    params: dict
    metrics: dict
    detector: ReconstructionErrorDetector
    corruption_table: pd.DataFrame
    threshold_sweep: pd.DataFrame
    variance_sweep: pd.DataFrame
    errors: dict  # clean / corrupted error samples for the histogram figure
    run_id: str | None = None
    extras: dict = field(default_factory=dict)


def run_part_f(split: MnistSplit, cfg: dict, seed: int = 42) -> DetectorResult:
    flag_rate = float(cfg.get("flag_rate", 0.04))
    det = ReconstructionErrorDetector(
        variance=float(cfg.get("variance", 0.95)),
        flag_rate=flag_rate,
        calibration_size=int(cfg.get("calibration_size", 10_000)),
        random_state=seed,
    ).fit(split.X_train)
    table, summary = evaluate_on_corruptions(det, split.X_test, seed)

    clean_err = det.reconstruction_error(split.X_test)
    corrupted = {
        kind: det.reconstruction_error(corrupt_digits(split.X_test, kind, seed + i))
        for i, kind in enumerate(CORRUPTIONS)
    }
    sweep_rows = []
    for pct in cfg.get("threshold_sweep", [1, 2, 4, 6, 8, 10]):
        thr = float(np.quantile(det.calibration_errors_, 1 - pct / 100))
        row = {
            "flag_rate_pct": pct,
            "threshold": thr,
            "clean_false_alarm_pct": 100 * float(np.mean(clean_err > thr)),
        }
        for kind, err in corrupted.items():
            row[f"detect_{kind}_pct"] = 100 * float(np.mean(err > thr))
        row["mean_detection_pct"] = float(np.mean([row[f"detect_{k}_pct"] for k in corrupted]))
        sweep_rows.append(row)
    threshold_sweep = pd.DataFrame(sweep_rows)

    var_rows = []
    for v in cfg.get("variance_sweep", [0.80, 0.90, 0.95, 0.99]):
        d = ReconstructionErrorDetector(
            v, flag_rate, int(cfg.get("calibration_size", 10_000)), seed
        )
        d.fit(split.X_train)
        _, s = evaluate_on_corruptions(d, split.X_test, seed)
        var_rows.append({"variance": v, "n_components": d.n_components_, **s})
    variance_sweep = pd.DataFrame(var_rows)

    train_thr_fa = 100 * float(np.mean(clean_err > det.train_threshold_))
    _, fa_lo, fa_hi = mean_ci((clean_err > det.threshold_).astype(float))
    metrics = {
        "n_components": det.n_components_,
        "threshold": det.threshold_,
        "train_threshold": det.train_threshold_,
        "clean_false_alarm_pct": summary["clean_false_alarm_pct"],
        "clean_false_alarm_ci_low_pct": 100 * fa_lo,
        "clean_false_alarm_ci_high_pct": 100 * fa_hi,
        "clean_false_alarm_pct_train_threshold": train_thr_fa,
        "mean_detection_pct": summary["mean_detection_pct"],
        "roc_auc": summary["roc_auc"],
        **{
            f"detect_{r.input}_pct": float(r.flagged_pct)
            for r in table.itertuples()
            if r.input != "clean test digits"
        },
    }
    params = {
        "variance": det.variance,
        "flag_rate": flag_rate,
        "calibration_size": det.calibration_size,
    }
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(clean_err), min(3000, len(clean_err)), replace=False)
    errors = {"clean": clean_err[sample], **{k: v[sample] for k, v in corrupted.items()}}
    return DetectorResult(params, metrics, det, table, threshold_sweep, variance_sweep, errors)


def error_rate_by_flag(classifier, detector, X: np.ndarray, y: np.ndarray) -> dict:
    """Classifier error rate on clean test digits the detector flags vs. the ones it passes."""
    flagged = detector.is_unusual(X)
    wrong = classifier.predict(X) != y
    return {
        "n_flagged_clean": int(flagged.sum()),
        "error_rate_flagged_pct": 100 * float(wrong[flagged].mean())
        if flagged.any()
        else float("nan"),
        "error_rate_unflagged_pct": 100 * float(wrong[~flagged].mean()),
        "n_errors_flagged": int(wrong[flagged].sum()),
        "n_errors_total": int(wrong.sum()),
    }
