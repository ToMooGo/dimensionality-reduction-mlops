"""Confidence intervals, computed as in Geron, Ch. 2 ("Fine-Tune Your Model", p. 83).

The book computes a 95% confidence interval for a test-set metric from the per-instance values
with ``scipy.stats.t.interval``. The same recipe works for accuracy (per-image 0/1 correctness),
for false-alarm rates (per-image 0/1 flags) and for the difference between two models evaluated
on the same images (per-image differences, i.e. a paired interval).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean of ``values`` and its ``confidence`` t-interval: ``(mean, low, high)``."""
    v = np.asarray(values, dtype=np.float64)
    m = float(v.mean())
    sem = float(stats.sem(v)) if len(v) > 1 else 0.0
    if sem == 0.0:
        return m, m, m
    lo, hi = stats.t.interval(confidence, len(v) - 1, loc=m, scale=sem)
    return m, float(lo), float(hi)


def half_width(values, confidence: float = 0.95) -> float:
    m, lo, hi = mean_ci(values, confidence)
    return (hi - lo) / 2
