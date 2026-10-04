"""Drift checks on the prediction log (pure function, unit-tested).

The detector's threshold is set so that about 4% of normal digits are flagged as unusual. A live
flag rate far above that means the inputs no longer look like the training data.
"""

from __future__ import annotations

from collections import Counter


def summarize_predictions(rows: list[dict], cfg: dict) -> dict:
    """``rows``: dicts with ``digit_raw``, ``digit_pca``, ``is_unusual`` and
    ``reconstruction_error``. ``cfg``: the ``monitor`` config section."""
    n = len(rows)
    report: dict = {"n_predictions": n, "alerts": []}
    if n < int(cfg["min_predictions"]):
        report["status"] = "insufficient-data"
        return report
    unusual_rate = 100 * sum(bool(r["is_unusual"]) for r in rows) / n
    agree = 100 * sum(int(r["digit_raw"]) == int(r["digit_pca"]) for r in rows) / n
    counts = Counter(int(r["digit_pca"]) for r in rows)
    top_digit, top_count = counts.most_common(1)[0]
    top_share = 100 * top_count / n
    errors = sorted(float(r["reconstruction_error"]) for r in rows)
    report.update(
        {
            "unusual_rate_pct": unusual_rate,
            "models_agree_pct": agree,
            "median_reconstruction_error": errors[n // 2],
            "top_digit": top_digit,
            "top_digit_share_pct": top_share,
            "class_counts": {str(k): v for k, v in sorted(counts.items())},
        }
    )
    if unusual_rate > float(cfg["max_unusual_rate_pct"]):
        report["alerts"].append(
            f"unusual-input rate {unusual_rate:.1f}% > {cfg['max_unusual_rate_pct']}%"
        )
    if top_share > float(cfg["max_class_share_pct"]):
        report["alerts"].append(f"digit {top_digit} is {top_share:.1f}% of predictions")
    if agree < float(cfg["min_models_agree_pct"]):
        report["alerts"].append(f"the two classifiers agree on only {agree:.1f}% of inputs")
    report["status"] = "alert" if report["alerts"] else "ok"
    return report
