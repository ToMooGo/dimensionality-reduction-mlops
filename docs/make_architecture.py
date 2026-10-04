"""Draw docs/images/architecture.png (run: python docs/make_architecture.py)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK_2, MUTED = "#0b0b0b", "#3d3c39", "#8a8984"
BLUE, BLUE_SOFT = "#2a78d6", "#e3eefb"
ORANGE, ORANGE_SOFT = "#eb6834", "#fdebe3"
AQUA, AQUA_SOFT = "#1baf7a", "#e3f6ef"
GREY_SOFT = "#f0efec"


def box(ax, x, y, w, h, title, lines=(), edge=BLUE, fill=BLUE_SOFT, title_size=13.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", lw=1.6, ec=edge, fc=fill
        )
    )
    ax.text(
        x + 0.18, y + h - 0.2, title, fontsize=title_size, fontweight="bold", color=INK, va="top"
    )
    for i, line in enumerate(lines):
        ax.text(x + 0.18, y + h - 0.66 - 0.4 * i, line, fontsize=11.2, color=INK_2, va="top")


def arrow(ax, p, q, text=None, offset=(0, 0.14), color=INK_2, rad=0.0, ha="center"):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.4,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )
    if text:
        mx, my = (p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1]
        ax.text(
            mx,
            my,
            text,
            fontsize=10.5,
            color=INK,
            ha=ha,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"},
        )


def main(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(16, 9.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(
        0.2,
        9.0,
        "Dimensionality Reduction MLOps: one config, one command (docker compose up)",
        fontsize=15,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.2,
        8.55,
        "Every box is a Docker Compose service, except the data source, the reports folder and the browser.",
        fontsize=10.5,
        color=INK_2,
        va="top",
    )

    # data source
    box(
        ax,
        0.2,
        5.75,
        3.0,
        2.2,
        "MNIST (70,000 x 784)",
        [
            "fetch_openml('mnist_784')",
            "as in the book; fallback:",
            "official files from a mirror,",
            "MD5-checked; cached",
        ],
        edge=MUTED,
        fill=GREY_SOFT,
    )

    # pipeline
    box(
        ax,
        4.0,
        4.45,
        5.2,
        3.5,
        "pipeline  (Prefect flows)",
        ["train-flow: one task per part"],
    )
    parts = [
        ("A  compression", "B  scaling PCA"),
        ("C  Kernel PCA", "D  2D maps (t-SNE ...)"),
        ("E  SVM raw vs PCA", "F  unusual-input flag"),
    ]
    for i, (left, right) in enumerate(parts):
        yy = 7.95 - 1.06 - 0.4 * i
        ax.text(4.5, yy, left, fontsize=11.2, color=INK_2, va="top")
        ax.text(6.65, yy, right, fontsize=11.2, color=INK_2, va="top")
    for i, line in enumerate(
        [
            "eval-flow: quality gates + champion check",
            "deploy-flow: set @champion, POST /reload",
            "monitor-flow: hourly drift check",
        ]
    ):
        ax.text(4.18, 7.95 - 2.26 - 0.4 * i, line, fontsize=11.2, color=INK_2, va="top")
    # report outputs
    box(
        ax,
        4.0,
        2.75,
        5.2,
        1.05,
        "reports/  (written into the repo)",
        ["14 figures, RESULTS.md, metrics.json"],
        edge=MUTED,
        fill=GREY_SOFT,
    )

    # mlflow and prefect
    box(
        ax,
        10.1,
        6.05,
        2.75,
        1.9,
        "MLflow :5050",
        ["tracking + registry", "5 models @champion", "skops serialisation"],
        edge=ORANGE,
        fill=ORANGE_SOFT,
    )
    box(
        ax,
        13.15,
        6.05,
        2.65,
        1.9,
        "Prefect :4200",
        ["server + UI", "flow and task runs", "schedules"],
        edge=ORANGE,
        fill=ORANGE_SOFT,
    )

    # api
    box(
        ax,
        10.1,
        2.6,
        3.75,
        2.75,
        "api  (FastAPI) :8000",
        [
            "POST /compress   POST /predict",
            "GET /maps  /samples  /monitoring",
            "web UI: PCA Lab",
            "polls the registry for a new",
            "@champion; logs every prediction",
        ],
        edge=AQUA,
        fill=AQUA_SOFT,
    )
    box(
        ax,
        14.6,
        3.3,
        1.3,
        1.55,
        "browser",
        ["PCA Lab", "web page"],
        edge=MUTED,
        fill=GREY_SOFT,
        title_size=11.5,
    )

    # postgres
    box(
        ax,
        10.1,
        0.35,
        5.8,
        1.5,
        "PostgreSQL :5433",
        [
            "databases: mlflow (runs, registry) and prefect (flow state),",
            "app (one row per prediction: input, both SVMs, flag)",
        ],
        edge=MUTED,
        fill=GREY_SOFT,
    )

    # arrows
    arrow(ax, (3.2, 6.85), (4.0, 6.85), "load", offset=(0, 0.1))
    arrow(ax, (6.6, 4.45), (6.6, 3.8), "write", offset=(0.45, -0.14))
    arrow(ax, (9.2, 7.2), (10.1, 7.2), "log runs,\nregister", offset=(0, 0.12))
    # pipeline -> Prefect server, routed above the MLflow box
    ax.plot([8.6, 8.6, 14.5], [7.95, 8.3, 8.3], color=INK_2, lw=1.4)
    arrow(ax, (14.5, 8.3), (14.5, 7.95))
    ax.text(
        11.6, 8.36, "report flow and task runs", fontsize=10.5, color=INK, ha="center", va="bottom"
    )
    arrow(ax, (9.2, 4.8), (10.1, 4.8), "POST\n/reload", offset=(0, 0.1))
    arrow(ax, (11.4, 6.05), (11.4, 5.35), "load @champion", offset=(1.0, -0.17))
    # browser <-> api (requests in, pages and JSON out)
    ax.add_patch(
        FancyArrowPatch(
            (13.85, 4.05), (14.6, 4.05), arrowstyle="<|-|>", mutation_scale=14, lw=1.4, color=INK_2
        )
    )
    ax.text(14.22, 4.15, "HTTP", fontsize=10.5, color=INK, ha="center", va="bottom")
    arrow(ax, (12.0, 2.6), (12.0, 1.85), "log predictions", offset=(1.0, -0.17))
    # monitor-flow reads the prediction log: PostgreSQL -> pipeline, routed left of reports/
    ax.plot([10.1, 3.55, 3.55], [1.1, 1.1, 4.9], color=INK_2, lw=1.4)
    arrow(ax, (3.55, 4.9), (4.0, 4.9))
    ax.text(
        6.8,
        1.2,
        "monitor-flow reads the prediction log (drift alerts are logged to MLflow)",
        fontsize=10.5,
        color=INK,
        ha="center",
        va="bottom",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main(Path(__file__).resolve().parent / "images" / "architecture.png"))
