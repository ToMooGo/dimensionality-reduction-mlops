"""Figures for the README, RESULTS.md and the technical report (Matplotlib, saved as PNG).

Style: white background, recessive grid, thin marks, text in ink colours (never in series
colours). Categorical colours follow one fixed order. The 10 digit classes need more hues than
that order has, so every digit map also writes the digit itself at its cluster centre. Colour
is therefore never the only way to tell the classes apart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e3e2dc"
SURFACE = "#ffffff"
# fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# 10 digit classes: the eight hues plus two neutrals; digits are also written on the maps
DIGIT_COLORS = SERIES + ["#52514e", "#a8a7a0"]
SEQ = "viridis"  # single perceptual sequential map for positions along the Swiss roll

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

HALO = [patheffects.withStroke(linewidth=3, foreground=SURFACE)]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _digit_grid(ax, images, ncols, cmap="gray_r", vmax=255.0):
    """Tile 28x28 images into one array and show it."""
    images = np.asarray(images).reshape(-1, 28, 28)
    nrows = int(np.ceil(len(images) / ncols))
    canvas = np.zeros((nrows * 30, ncols * 30))
    for i, im in enumerate(images):
        r, c = divmod(i, ncols)
        canvas[r * 30 + 1 : r * 30 + 29, c * 30 + 1 : c * 30 + 29] = im
    ax.imshow(canvas, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------------------------------
# Part A - compression
# ------------------------------------------------------------------------------------------
def explained_variance(cumulative: np.ndarray, table, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    d = np.arange(1, len(cumulative) + 1)
    ax.plot(d, cumulative, color=SERIES[0], lw=2)
    for r in table.itertuples():
        ax.plot(
            [r.n_components],
            [r.train_explained_variance],
            "o",
            ms=7,
            color=SERIES[0],
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            f"{r.variance_target:.0%}: d = {r.n_components}",
            (r.n_components, r.train_explained_variance),
            xytext=(12, -14 if r.variance_target < 0.9 else -16),
            textcoords="offset points",
            fontsize=10,
            color=INK,
            path_effects=HALO,
        )
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("number of principal components d")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("MNIST: explained variance vs number of dimensions (first 400 of 784)", loc="left")
    return _save(fig, path)


def reconstructions(X_test, pca_full, table, index, path: Path) -> Path:
    from .compression import reconstruct

    rows = [("original\n784 pixels", X_test[index])]
    for r in table.itertuples():
        rec = np.clip(reconstruct(pca_full, X_test[index], r.n_components), 0, 255)
        rows.append((f"{r.variance_target:.0%} variance\nd = {r.n_components}", rec))
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(9.5, 1.15 * len(rows) + 0.4), layout="constrained"
    )
    for ax, (label, imgs) in zip(axes, rows, strict=True):
        _digit_grid(ax, imgs, ncols=len(index))
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10, labelpad=8)
    fig.suptitle(
        "Test digits compressed to d components and decompressed (inverse_transform)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, path)


def principal_components(pca_full, path: Path, n: int = 10) -> Path:
    comps = pca_full.components_[:n]
    lim = float(np.abs(comps).max())
    ncols = 5
    fig, axes = plt.subplots(2, ncols, figsize=(8.6, 4.4), layout="constrained")
    im = None
    for i, ax in enumerate(axes.ravel()):
        im = ax.imshow(comps[i].reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"PC{i + 1}: {pca_full.explained_variance_ratio_[i]:.1%}", fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("weight (one shared scale)", fontsize=10)
    cb.set_ticks([-lim, 0, lim], labels=["negative", "0", "positive"])
    fig.suptitle(
        "First 10 principal components of MNIST as images\n"
        "(panel titles: share of the variance each component explains)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, path)


# ------------------------------------------------------------------------------------------
# Part B - scaling
# ------------------------------------------------------------------------------------------
def solver_comparison(solvers, path: Path) -> Path:
    df = solvers.iloc[::-1]
    names = [s.replace(", ", ",\n", 1) if len(s) > 22 else s for s in df.solver]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.9), layout="constrained", sharey=True)
    fig.get_layout_engine().set(wspace=0.08)
    y = np.arange(len(df))
    ax1.barh(y, df.fit_seconds, height=0.62, color=SERIES[0])
    ax1.set_yticks(y, names, fontsize=10)
    ax1.set_xlabel("fit time (s), median of runs")
    ax1.set_title("Time to find 154 components of 60,000 images", loc="left")
    for yi, v in zip(y, df.fit_seconds, strict=True):
        ax1.text(v, yi, f"  {v:.1f} s", va="center", fontsize=10, color=INK, path_effects=HALO)
    ax1.set_xlim(0, df.fit_seconds.max() * 1.25)
    ax1.grid(axis="y", visible=False)

    ax2.barh(y, df.total_RAM_MB, height=0.62, color=SERIES[0])
    ax2.set_xlabel("memory while fitting (MB): data held in RAM + peak heap allocations")
    ax2.set_title("Memory needed", loc="left")
    for yi, v, kind in zip(y, df.total_RAM_MB, df.data, strict=True):
        ax2.text(
            v,
            yi,
            f"  {v:,.0f} MB" + ("  (data stays on disk)" if "disk" in kind else ""),
            va="center",
            fontsize=10,
            color=INK,
            path_effects=HALO,
        )
    ax2.set_xlim(0, df.total_RAM_MB.max() * 1.6)
    ax2.grid(axis="y", visible=False)
    return _save(fig, path)


def timing_curve(curve, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 4.8), layout="constrained")
    solvers = list(dict.fromkeys(curve.solver))
    markers = ["s", "o", "^", "D", "v", "P"]
    styles = ["--", "-", "-", "-", "-", "-"]
    for i, name in enumerate(solvers):
        sub = curve[curve.solver == name].sort_values("n_train")
        ax.plot(
            sub.n_train,
            sub.fit_seconds,
            styles[i],
            marker=markers[i],
            color=SERIES[i],
            lw=2,
            ms=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=name,
            zorder=3 + (i == 0),
        )
        if "algorithm" in curve and name == "PCA, solver='auto'":
            other = sub[sub.algorithm != "covariance_eigh"]
            if len(other):
                ax.plot(
                    other.n_train,
                    other.fit_seconds,
                    ls="",
                    marker="o",
                    ms=12,
                    mfc="none",
                    mec=INK,
                    mew=1.4,
                    zorder=6,
                    label="'auto' ran randomized here (m < 10 \u00d7 784)",
                )
    ax.set_yscale("log")
    sizes = sorted(curve.n_train.unique())
    ax.set_xticks(sizes, [f"{v / 1000:g}k" for v in sizes])
    ax.set_xlabel("training images m")
    ax.set_ylabel("fit time (s, log scale)")
    ax.set_title("Fit time vs dataset size, 154 components (in-RAM solvers)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, path)


# ------------------------------------------------------------------------------------------
# Part C - Kernel PCA and manifold learning on the Swiss roll
# ------------------------------------------------------------------------------------------
def _scatter_t(ax, Z, t, title):
    ax.scatter(Z[:, 0], Z[:, 1], c=t, cmap=SEQ, s=7, linewidths=0)
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([])
    ax.set_yticks([])


def swiss_roll_kpca(
    roll: dict, path: Path, rbf_gamma: float, sigmoid_gamma: float, seed: int = 42
) -> Path:
    from sklearn.decomposition import KernelPCA

    X, t = roll["X"], roll["t"]
    fig = plt.figure(figsize=(14, 4.0), layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")
    ax0.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=SEQ, s=6, linewidths=0, depthshade=False)
    ax0.view_init(10, -70)
    ax0.set_title("Swiss roll (3D)", fontsize=10.5)
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.set_zticklabels([])
    settings = [
        ("linear", None, "Linear kernel\n(= ordinary PCA)"),
        ("rbf", rbf_gamma, f"RBF kernel, gamma = {rbf_gamma:.4f}\n(chosen by CV accuracy)"),
        (
            "sigmoid",
            sigmoid_gamma,
            f"Sigmoid kernel, gamma = {sigmoid_gamma:.4f}\n(best sigmoid by CV accuracy)",
        ),
    ]
    for i, (kernel, gamma, title) in enumerate(settings, start=2):
        Z = KernelPCA(n_components=2, kernel=kernel, gamma=gamma, random_state=seed).fit_transform(
            X
        )
        _scatter_t(fig.add_subplot(1, 4, i), Z, t, title)
    fig.suptitle(
        "Swiss roll reduced to 2D with Kernel PCA (colour = position along the roll)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, path)


def kpca_selection(
    grid, preimage, best_gamma, pre_gamma, best_kernel, pre_kernel, path: Path
) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    col = "cv_preimage_mse" if "cv_preimage_mse" in preimage else "test_preimage_mse"
    for i, kernel in enumerate(["rbf", "sigmoid"]):
        g = grid[grid.kernel == kernel].sort_values("gamma")
        ax1.plot(g.gamma, 100 * g.cv_accuracy, "-o", color=SERIES[i], lw=2, ms=5, label=kernel)
        p = preimage[preimage.kernel == kernel].sort_values("gamma")
        ax2.plot(p.gamma, p[col], "-o", color=SERIES[i], lw=2, ms=5, label=kernel)
    for ax, g, kernel, frac in [
        (ax1, best_gamma, best_kernel, 0.55),
        (ax2, pre_gamma, pre_kernel, 0.5),
    ]:
        ax.axvline(g, color=MUTED, lw=1)
        ylo, yhi = ax.get_ylim()
        xlo, xhi = ax.get_xlim()
        right = g > (xlo + xhi) / 2  # near the right edge: write the note on the left of the line
        ax.text(
            g,
            ylo + frac * (yhi - ylo),
            f"chosen: {kernel},  \ngamma = {g:.4f}  "
            if right
            else f"  chosen: {kernel},\n  gamma = {g:.4f}",
            fontsize=10.5,
            color=INK,
            ha="right" if right else "left",
            va="center",
            path_effects=HALO,
        )
        ax.set_xlabel("gamma")
    ax1.set_ylabel("3-fold CV accuracy (%, higher = better)")
    ax1.set_title("Supervised rule: kPCA -> Logistic Regression accuracy", loc="left")
    ax2.set_ylabel("3-fold CV pre-image MSE (lower = better)")
    ax2.set_title("Unsupervised rule: reconstruction pre-image error", loc="left")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, [f"{lab} kernel" for lab in labels], loc="outside upper center", ncol=2)
    return _save(fig, path)


def manifold_methods(embeddings: dict, t, table, path: Path) -> Path:
    names = list(embeddings)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.6 * nrows + 0.3), layout="constrained")
    fig.get_layout_engine().set(wspace=0.05, hspace=0.06)
    for ax, name in zip(axes.ravel(), names, strict=False):
        r = table.loc[table.method == name].iloc[0]
        secs = "< 0.01 s" if r.seconds < 0.005 else f"{r.seconds:.2f} s"
        _scatter_t(
            ax,
            embeddings[name],
            t,
            f"{name}\naccuracy {r.downstream_accuracy:.1%} · {secs}",
        )
    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    fig.suptitle(
        "The Swiss roll mapped to 2D by six methods (colour = position t along the roll, "
        "purple = start, yellow = end)\n"
        "Accuracy: 3-fold CV of a linear classifier telling t \u2264 6.9 from t > 6.9 on the 2D points",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, path)


# ------------------------------------------------------------------------------------------
# Part D - MNIST maps
# ------------------------------------------------------------------------------------------
def _seg_point_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0, 1))
    return float(np.hypot(*(p - (a + t * ab))))


def _place_labels(med: np.ndarray, min_dist: float = 0.075) -> np.ndarray:
    """Label positions (axis-fraction units) for the class medians ``med``.

    A label stays on its median when nothing else is close. Otherwise it is moved outwards,
    away from the crowd, to the first candidate position that keeps a clear distance from every
    other label and median and whose leader line does not pass under another label.
    """
    n = len(med)
    placed = np.full((n, 2), np.nan)
    crowd = [
        int(sum(np.hypot(*(med[j] - med[i])) < min_dist for j in range(n) if j != i))
        for i in range(n)
    ]
    for i in sorted(range(n), key=lambda k: crowd[k]):
        others = [j for j in range(n) if j != i]
        if crowd[i] == 0:
            placed[i] = med[i]
            continue
        near = [j for j in others if np.hypot(*(med[j] - med[i])) < 2 * min_dist]
        centre = np.mean(med[[i, *near]], axis=0)
        away = med[i] - centre
        base = np.arctan2(away[1], away[0]) if np.hypot(*away) > 1e-9 else 0.0
        best = None
        for radius in (0.07, 0.1, 0.13, 0.17, 0.22):
            for k in range(24):
                ang = base + (k // 2 + 1) * (np.pi / 12) * (-1 if k % 2 else 1) - np.pi / 12
                cand = np.clip(med[i] + radius * np.array([np.cos(ang), np.sin(ang)]), 0.04, 0.96)
                ok = all(np.hypot(*(cand - med[j])) >= min_dist * 0.8 for j in others)
                ok = ok and all(
                    np.hypot(*(cand - placed[j])) >= min_dist
                    for j in others
                    if not np.isnan(placed[j, 0])
                )
                ok = ok and all(
                    _seg_point_dist(placed[j], med[i], cand) >= 0.035
                    for j in others
                    if not np.isnan(placed[j, 0])
                )
                if ok:
                    best = cand
                    break
            if best is not None:
                break
        placed[i] = best if best is not None else med[i]
    return placed


def _digit_map(ax, Z, y, title, s=4, label_size=13):
    for d in range(10):
        m = y == d
        ax.scatter(Z[m, 0], Z[m, 1], s=s, color=DIGIT_COLORS[d], linewidths=0, alpha=0.8)
    # the digit is written at its class median; overlapping labels are spread apart and joined to
    # their median by a thin line, so no label hides another
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    pad = 0.03 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    med = np.array([np.median(Z[y == d], axis=0) for d in range(10)])
    frac = np.c_[(med[:, 0] - x0) / (x1 - x0), (med[:, 1] - y0) / (y1 - y0)]
    placed = _place_labels(frac)
    for d in range(10):
        lx, ly = x0 + placed[d, 0] * (x1 - x0), y0 + placed[d, 1] * (y1 - y0)
        if np.hypot(*(placed[d] - frac[d])) > 1e-6:
            halo = [patheffects.withStroke(linewidth=2.6, foreground=SURFACE)]
            ax.plot(
                [med[d, 0], lx], [med[d, 1], ly], color=INK, lw=0.9, zorder=4, path_effects=halo
            )
            ax.plot(med[d, 0], med[d, 1], "o", ms=3.5, color=INK, mec=SURFACE, mew=0.8, zorder=4)
        ax.text(
            lx,
            ly,
            str(d),
            fontsize=label_size,
            fontweight="bold",
            color=INK,
            ha="center",
            va="center",
            zorder=5,
            path_effects=[patheffects.withStroke(linewidth=4, foreground=SURFACE)],
        )
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _digit_legend(where, title=None):
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", ms=8, color=DIGIT_COLORS[d]) for d in range(10)
    ]
    title = title or (
        "colour = true digit; the digit is written at its class median "
        "(black dot + line = label moved to avoid a collision)"
    )
    kw = (
        {"loc": "outside lower center"}
        if hasattr(where, "get_layout_engine")
        else {"loc": "upper center", "bbox_to_anchor": (0.5, -0.02)}
    )
    where.legend(
        handles, [str(d) for d in range(10)], ncol=10, title=title, title_fontsize=10, **kw
    )


def tsne_map(Z, y, images, path: Path, min_dist: float = 0.052, max_thumbs: int = 400) -> Path:
    """Left: coloured map with the digit written at each cluster centre.
    Right: scaled-down digit images, one wherever no other image is already close (book's idea)."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.4), layout="constrained")
    _digit_map(ax1, Z, y, "t-SNE map, coloured by true digit", s=5, label_size=16)
    Zn = (Z - Z.min(0)) / (Z.max(0) - Z.min(0))
    ax2.scatter(Zn[:, 0], Zn[:, 1], s=2, color=GRID, linewidths=0)
    shown = np.empty((0, 2))
    for i in range(len(Zn)):
        if len(shown) >= max_thumbs:
            break
        if len(shown) and np.min(np.sum((shown - Zn[i]) ** 2, axis=1)) < min_dist**2:
            continue
        shown = np.vstack([shown, Zn[i]])
        im = images[i].reshape(28, 28).astype(float)
        im = im / max(float(im.max()), 1.0)  # same darkness for faint and heavy strokes
        img = OffsetImage(im, cmap="gray_r", zoom=0.55)
        img.get_children()[0].set_clim(0, 1)
        ax2.add_artist(AnnotationBbox(img, Zn[i], frameon=False, pad=0))
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("The same map with the digit images (one per neighbourhood)", fontsize=10.5)
    _digit_legend(ax1, "colour = true digit (digit written at each class median)")
    fig.suptitle(
        f"t-SNE of {len(Z):,} MNIST training digits (PCA to 95% variance first)",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    return _save(fig, path)


def maps_comparison(embeddings: dict, y, table, path: Path) -> Path:
    names = [n for n in embeddings if n != "t-SNE (raw pixels)"]
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.3 * nrows + 0.6), layout="constrained")
    for ax, name in zip(axes.ravel(), names, strict=False):
        r = table.loc[table.method == name].iloc[0]
        _digit_map(
            ax,
            embeddings[name],
            y,
            f"{name}\nkNN accuracy in 2D {r.knn_accuracy_2d:.1%} · {r.seconds:.1f} s",
        )
    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    _digit_legend(fig)
    fig.suptitle(
        f"{len(y):,} MNIST digits mapped to 2D: higher kNN accuracy = digits better separated",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    return _save(fig, path)


# ------------------------------------------------------------------------------------------
# Part E - classification with and without PCA
# ------------------------------------------------------------------------------------------
def classification(table, path: Path) -> Path:
    table = table[table.features.isin(["raw", "pca"])]
    clfs = list(dict.fromkeys(table.classifier))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), layout="constrained")
    x = np.arange(len(clfs))
    w = 0.36
    panels = [
        ("fit_seconds", "training time (s)", "Training time", "{:.1f}"),
        ("predict_seconds_10k", "time to predict 10,000 images (s)", "Prediction time", "{:.1f}"),
        (
            "test_accuracy",
            "test accuracy (%, axis starts at 90)",
            "Test accuracy, 95% CI",
            "{:.1f}",
        ),
    ]
    handles = []
    for ax, (col, ylabel, title, fmt) in zip(axes, panels, strict=True):
        for j, (rep, label) in enumerate(
            [("raw", "784 raw pixels"), ("pca", "PCA features (95% of the variance)")]
        ):
            sel = [table[(table.classifier == c) & (table.features == rep)].iloc[0] for c in clfs]
            vals = np.array([float(r[col]) for r in sel])
            err = None
            if col == "test_accuracy":
                vals = 100 * vals
                if "accuracy_ci_low" in table:
                    lo = 100 * np.array([float(r.accuracy_ci_low) for r in sel])
                    hi = 100 * np.array([float(r.accuracy_ci_high) for r in sel])
                    err = np.vstack([vals - lo, hi - vals])
            bars = ax.bar(
                x + (j - 0.5) * w,
                vals,
                width=w - 0.03,
                color=SERIES[j],
                label=label,
                yerr=err,
                error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3},
            )
            if ax is axes[0]:
                handles.append(bars)
            for b, v, k in zip(bars, vals, range(len(vals)), strict=True):
                top = b.get_height() + (err[1][k] if err is not None else 0)
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    top,
                    fmt.format(v) if v >= 1 else f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color=INK,
                    path_effects=HALO,
                )
        ax.set_xticks(x, [c.replace(" (", "\n(") for c in clfs], fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        if col == "test_accuracy":
            ax.set_ylim(90, max(99, float(100 * table.test_accuracy.max()) + 1.2))
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.legend(handles, [h.get_label() for h in handles], loc="outside upper center", ncol=2)
    return _save(fig, path)


# ------------------------------------------------------------------------------------------
# Part F - unusual-input detector
# ------------------------------------------------------------------------------------------
def error_histograms(errors: dict, threshold: float, table, path: Path) -> Path:
    kinds = [k for k in errors if k != "clean"]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.6), layout="constrained", sharex=True)
    bins = np.logspace(
        np.log10(5), np.log10(max(float(np.max(v)) for v in errors.values()) * 1.1), 60
    )
    for ax, kind in zip(axes.ravel(), kinds, strict=True):
        ax.hist(errors["clean"], bins=bins, color=MUTED, alpha=0.55, label="clean test digits")
        ax.hist(errors[kind], bins=bins, color=SERIES[0], alpha=0.8, label=kind)
        ax.axvline(threshold, color=SERIES[7], lw=1.5)
        pct = float(table.loc[table.input == kind, "flagged_pct"].iloc[0])
        ax.set_title(f"{kind}: {pct:.0f}% flagged", fontsize=10.5, loc="left")
        ax.set_xscale("log")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("reconstruction error (MSE, log scale)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.55),
        plt.Rectangle((0, 0), 1, 1, color=SERIES[0], alpha=0.8),
        plt.Line2D([0], [0], color=SERIES[7], lw=1.5),
    ]
    fig.legend(
        handles,
        ["clean test digits", "corrupted copies", f"threshold = {threshold:.0f}"],
        loc="outside lower center",
        ncol=3,
    )
    fig.suptitle(
        "Reconstruction error of clean vs corrupted test digits (PCA, 154 components; "
        "each panel scaled to its own peak)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, path)


def corruption_examples(X_test, detector, path: Path, index: int = 0, seed: int = 0) -> Path:
    from .data import CORRUPTIONS, corrupt_digits

    x = X_test[index : index + 1]
    inputs = [("clean", x)] + [
        (k, corrupt_digits(x, k, seed + i)) for i, k in enumerate(CORRUPTIONS)
    ]
    fig, axes = plt.subplots(
        2, len(inputs), figsize=(1.45 * len(inputs), 3.9), layout="constrained"
    )
    for j, (name, img) in enumerate(inputs):
        rec = np.clip(
            detector.pca_.inverse_transform(detector.pca_.transform(img.astype(np.float64))), 0, 255
        )
        err = float(detector.reconstruction_error(img)[0])
        flag = err > detector.threshold_
        axes[0, j].imshow(img.reshape(28, 28), cmap="gray_r", vmin=0, vmax=255)
        axes[1, j].imshow(rec.reshape(28, 28), cmap="gray_r", vmin=0, vmax=255)
        axes[0, j].set_title(name, fontsize=10)
        axes[1, j].set_xlabel(
            f"error {err:,.0f}\n{'UNUSUAL' if flag else 'normal'}",
            fontsize=9,
            color=INK if flag else INK_2,
            fontweight="bold" if flag else "normal",
        )
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(GRID)
                sp.set_linewidth(1.0)
    axes[0, 0].set_ylabel("input", fontsize=10)
    axes[1, 0].set_ylabel("rebuilt from\n154 PCs", fontsize=10)
    fig.suptitle(
        f"Inputs (top) and their PCA reconstructions (bottom); flagged above {detector.threshold_:,.0f}",
        x=0.01,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    return _save(fig, path)


def detection_rates(table, threshold_sweep, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1.1, 1]}
    )
    df = table.iloc[::-1]
    y = np.arange(len(df))
    colors = [MUTED if n == "clean test digits" else SERIES[0] for n in df.input]
    ax1.barh(y, df.flagged_pct, height=0.62, color=colors)
    ax1.set_yticks(
        y,
        ["clean test digits\n(false alarms)" if n == "clean test digits" else n for n in df.input],
    )
    for yi, v in zip(y, df.flagged_pct, strict=True):
        ax1.text(v, yi, f"  {v:.1f}%", va="center", fontsize=9.5, color=INK, path_effects=HALO)
    ax1.set_xlim(0, 118)
    ax1.set_xlabel("flagged as unusual (%)")
    ax1.set_title("Flag rate (threshold set for 4% on held-out training digits)", loc="left")
    ax1.grid(axis="y", visible=False)

    ax2.plot(
        threshold_sweep.clean_false_alarm_pct,
        threshold_sweep.mean_detection_pct,
        "-o",
        color=SERIES[0],
        lw=2,
        ms=6,
        markeredgecolor=SURFACE,
    )
    for r in threshold_sweep.itertuples():
        ax2.annotate(
            f"{r.flag_rate_pct:g}%",
            (r.clean_false_alarm_pct, r.mean_detection_pct),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color=INK,
            path_effects=HALO,
        )
    ax2.set_xlabel("false alarms on clean test digits (%)")
    ax2.set_ylabel("mean detection over 8 corruptions (%)")
    ax2.set_title("Trade-off as the threshold moves (label = target rate)", loc="left")
    ylo, yhi = threshold_sweep.mean_detection_pct.min(), threshold_sweep.mean_detection_pct.max()
    ax2.set_ylim(ylo - 2, yhi + 3)
    return _save(fig, path)
