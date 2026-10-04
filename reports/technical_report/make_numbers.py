"""Write numbers.tex (inline values) and tables.tex (result tables) from ../metrics.json.

The report quotes no number by hand: ``python make_numbers.py`` after every training run, then
``pdflatex technical_report.tex`` twice.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = json.loads((HERE.parent / "metrics.json").read_text())


def f(v, nd=1):
    return f"{v:,.{nd}f}".replace(",", "{,}")


def pct(v, nd=1):  # fraction -> percent string
    return f"{100 * v:.{nd}f}"


def signed(v, nd=2):  # signed value with a typographic minus/plus sign
    text = f"{v:+.{nd}f}"
    return ("$-$" if text[0] == "-" else "$+$") + text[1:]


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


a, b, c = M["A_compression"], M["B_scaling"], M["C_kernel_pca_manifolds"]
d, e, fdet = M["D_mnist_maps"], M["E_classification"], M["F_unusual_inputs"]
am, bm, cm, dm, em, fm = (
    a["metrics"],
    b["metrics"],
    c["metrics"],
    d["metrics"],
    e["metrics"],
    fdet["metrics"],
)
env = M["environment"]


def solver(name, col):
    return next(r[col] for r in b["solvers"] if r["solver"] == name)


def method(rows, name, col, key="method"):
    return next(r[col] for r in rows if r[key] == name)


def clf(kind, rep, col):
    return next(r[col] for r in e["table"] if r["kind"] == kind and r["features"] == rep)


def corr(kind):
    return next(r["flagged_pct"] for r in fdet["corruptions"] if r["input"] == kind)


vals = {
    # environment
    "env.cpus": str(env["cpus"]),
    "env.python": env["python"],
    "env.sklearn": env["scikit_learn"],
    "env.numpy": env["numpy"],
    # A
    "A.d95": str(am["n_components_95"]),
    "A.d50": str(am["n_components_50"]),
    "A.d99": str(am["n_components_99"]),
    "A.pctfeat": f(am["pct_of_features_95"]),
    "A.trainEV": pct(am["train_explained_variance_95"], 2),
    "A.testEV": pct(am["test_explained_variance_95"], 2),
    "A.testMSE": f(am["test_mse_95"]),
    "A.trainMSE": f(am["train_mse_95"], 2),
    "A.dropped": f(am["dropped_variance_per_pixel_95"], 2),
    "A.rmse": f(am["test_rmse_pixels_95"]),
    "A.pcOne": pct(am["explained_variance_pc1"]),
    "A.pcTwo": pct(am["explained_variance_pc2"]),
    # B
    "B.dataMB": f(bm["data_MB"], 0),
    "B.full": f(bm["full_svd_seconds"], 2),
    "B.auto": f(bm["auto_seconds"], 2),
    "B.rand": f(bm["randomized_seconds"], 2),
    "B.ipca": f(bm["ipca_seconds"], 1),
    "B.mmfitpeak": f(bm["memmap_fit_peak_MB"], 0),
    "B.mmpfpeak": f(bm["memmap_partial_fit_peak_MB"], 0),
    "B.mmpfsec": f(bm["memmap_partial_fit_seconds"], 1),
    "B.fullRAM": f(bm["full_svd_total_RAM_MB"], 0),
    "B.randgap": f"{bm['randomized_mse_gap_pct']:.2f}",
    "B.ipcagap": f"{bm['ipca_mse_gap_pct']:.2f}",
    "B.saving": f(bm["memory_saving_factor"], 0),
    "B.autosolver": tex_escape(b["params"]["auto_solver_chosen"]),
    "B.randRAM": f(solver("Randomized PCA", "total_RAM_MB"), 0),
    "B.autoRAM": f(solver("PCA, solver='auto'", "total_RAM_MB"), 0),
    # C
    "C.gamma": f"{cm['best_gamma']:.4f}",
    "C.kernel": c["params"]["best_kernel"],
    "C.cv": pct(cm["best_cv_accuracy"]),
    "C.testsup": pct(cm["test_accuracy_supervised_choice"]),
    "C.testunsup": pct(cm["test_accuracy_unsupervised_choice"]),
    "C.raw": pct(cm["test_accuracy_raw_3d"]),
    "C.linear": pct(cm["test_accuracy_linear_pca_2d"]),
    "C.pregamma": f"{cm['preimage_best_gamma']:.4f}",
    "C.prekernel": c["params"]["preimage_best_kernel"],
    "C.premse": f(cm["preimage_best_cv_mse"], 2),
    "C.cvstd": pct(cm["best_cv_std"]),
    "C.nwithin": str(cm["n_settings_within_one_std"]),
    "C.book": f"{cm['book_setting_preimage_mse_all_points']:.4f}",
    "C.nsettings": str(cm["n_settings"]),
    "C.lle": pct(method(c["methods"], "LLE", "downstream_accuracy")),
    "C.isomap": pct(method(c["methods"], "Isomap", "downstream_accuracy")),
    "C.tsne": pct(method(c["methods"], "t-SNE", "downstream_accuracy")),
    "C.mds": pct(method(c["methods"], "MDS", "downstream_accuracy")),
    "C.pca": pct(method(c["methods"], "PCA", "downstream_accuracy")),
    "C.lleCorr": f"{method(c['methods'], 'LLE', 'max_abs_corr_with_t'):.3f}",
    # D
    "D.n": f(dm["n_samples"], 0),
    "D.tsne": pct(dm["tsne_knn_accuracy_2d"]),
    "D.tsneraw": pct(dm["tsne_raw_knn_accuracy_2d"]),
    "D.pca": pct(dm["pca_knn_accuracy_2d"]),
    "D.tsnesec": f(dm["tsne_seconds"]),
    "D.tsnerawsec": f(dm["tsne_raw_seconds"]),
    "D.lle": pct(method(d["methods"], "LLE", "knn_accuracy_2d")),
    "D.isomap": pct(method(d["methods"], "Isomap", "knn_accuracy_2d")),
    "D.mds": pct(method(d["methods"], "MDS", "knn_accuracy_2d")),
    "D.kpca": pct(method(d["methods"], "Kernel PCA", "knn_accuracy_2d")),
    "D.mdssec": f(method(d["methods"], "MDS", "seconds"), 0),
    # E
    "E.svmraw": pct(em["svm_raw_accuracy"], 2),
    "E.svmpca": pct(em["svm_pca_accuracy"], 2),
    "E.svmfit": f(em["svm_fit_speedup"]),
    "E.svmpred": f(em["svm_predict_speedup"]),
    "E.svmn": f(e["params"]["train_sizes"]["svm"], 0),
    "E.svmd": str(clf("svm", "pca", "n_features")),
    "E.svmrawMB": f(clf("svm", "raw", "model_MB")),
    "E.svmpcaMB": f(clf("svm", "pca", "model_MB")),
    "E.rfraw": pct(em["random_forest_raw_accuracy"], 2),
    "E.rfpca": pct(em["random_forest_pca_accuracy"], 2),
    "E.rffit": f(em["random_forest_fit_speedup"], 2),
    "E.rfrawsec": f(clf("random_forest", "raw", "fit_seconds")),
    "E.rfrawMB": f(clf("random_forest", "raw", "model_MB"), 0),
    "E.rfpcaMB": f(clf("random_forest", "pca", "model_MB"), 0),
    "E.rfpcasec": f(clf("random_forest", "pca", "fit_seconds")),
    "E.smraw": pct(em["softmax_raw_accuracy"], 2),
    "E.smpca": pct(em["softmax_pca_accuracy"], 2),
    "E.smfit": f(em["softmax_fit_speedup"]),
    "E.svmdiff": signed(em["svm_accuracy_change_pp"]),
    "E.svmdifflo": signed(em["svm_accuracy_change_ci_low_pp"]),
    "E.svmdiffhi": signed(em["svm_accuracy_change_ci_high_pp"]),
    "E.rfdiff": signed(em["random_forest_accuracy_change_pp"]),
    "E.rfdifflo": signed(em["random_forest_accuracy_change_ci_low_pp"]),
    "E.rfdiffhi": signed(em["random_forest_accuracy_change_ci_high_pp"]),
    "E.smdiff": signed(em["softmax_accuracy_change_pp"]),
    "E.smdifflo": signed(em["softmax_accuracy_change_ci_low_pp"]),
    "E.smdiffhi": signed(em["softmax_accuracy_change_ci_high_pp"]),
    "E.same": pct(em["svm_pca_same_gamma_accuracy"], 2),
    "E.samediff": signed(em["svm_same_gamma_change_pp"]),
    "E.samedifflo": signed(em["svm_same_gamma_change_ci_low_pp"]),
    "E.samediffhi": signed(em["svm_same_gamma_change_ci_high_pp"]),
    "E.repeats": str(e["params"]["timing_repeats"]),
    "E.svmgain": f"{em['svm_accuracy_change_pp']:.2f}",
    "E.samegain": f"{em['svm_same_gamma_change_pp']:.2f}",
    "D.knn784": pct(dm["knn_accuracy_784d"]),
    "D.knn784std": pct(dm["knn_std_784d"]),
    # F
    "F.falo": f(fm["clean_false_alarm_ci_low_pct"], 2),
    "F.fahi": f(fm["clean_false_alarm_ci_high_pct"], 2),
    "F.d": str(fm["n_components"]),
    "F.thr": f(fm["threshold"], 0),
    "F.trainthr": f(fm["train_threshold"], 0),
    "F.fa": f(fm["clean_false_alarm_pct"], 2),
    "F.fatrain": f(fm["clean_false_alarm_pct_train_threshold"], 2),
    "F.mean": f(fm["mean_detection_pct"]),
    "F.auc": f"{fm['roc_auc']:.3f}",
    "F.rotated": f(corr("rotated")),
    "F.mirrored": f(corr("mirrored")),
    "F.shifted": f(corr("shifted")),
    "F.dimmed": f(corr("dimmed")),
    "F.errflag": f(fdet["classifier_error_by_flag"]["error_rate_flagged_pct"]),
    "F.errunflag": f(fdet["classifier_error_by_flag"]["error_rate_unflagged_pct"]),
    "F.nflag": f(fdet["classifier_error_by_flag"]["n_flagged_clean"], 0),
    "F.nerrflag": str(fdet["classifier_error_by_flag"]["n_errors_flagged"]),
    "F.nerr": str(fdet["classifier_error_by_flag"]["n_errors_total"]),
    "F.bestvarAUC": f"{max(r['roc_auc'] for r in fdet['variance_sweep']):.3f}",
    "F.bestvar": pct(max(fdet["variance_sweep"], key=lambda r: r["roc_auc"])["variance"], 0),
    "F.bestvard": str(max(fdet["variance_sweep"], key=lambda r: r["roc_auc"])["n_components"]),
}

lines = ["% generated by make_numbers.py from ../metrics.json - do not edit"]
for k, v in vals.items():
    lines.append(rf"\expandafter\def\csname num@{k}\endcsname{{{v}}}")
(HERE / "numbers.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- tables
def row(*cells):
    return " & ".join(cells) + r" \\"


T = ["% generated by make_numbers.py from ../metrics.json - do not edit"]


def table(name: str, spec: str, header: tuple, rows: list) -> None:
    T.extend([rf"\newcommand{{\{name}}}{{", rf"\begin{{tabular}}{{{spec}}}", r"\toprule"])
    T.append(row(*header))
    T.append(r"\midrule")
    T.extend(row(*r) for r in rows)
    T.extend([r"\bottomrule", r"\end{tabular}}"])


def n(v, fmt="{:,.0f}"):
    return fmt.format(v).replace(",", "{,}")


table(
    "TableVariance",
    "rrrrrr",
    ("target", "$d$", "features", "train var.", "test var.", "test RMSE"),
    [
        (
            f"{100 * r['variance_target']:.0f}\\%",
            str(r["n_components"]),
            f"{r['pct_of_features']:.1f}\\%",
            f"{100 * r['train_explained_variance']:.2f}\\%",
            f"{100 * r['test_explained_variance']:.2f}\\%",
            f"{r['test_rmse_pixels']:.1f}",
        )
        for r in a["variance_table"]
    ],
)

SHORT = {"raw pixels, uint8 (how MNIST is shipped)": "raw pixels, uint8 (as shipped)"}
table(
    "TableStorage",
    "lrrr",
    ("representation", "bytes/image", "vs float32 pixels", "vs uint8 pixels"),
    [
        (
            tex_escape(SHORT.get(r["representation"], r["representation"])),
            n(r["bytes_per_image"]),
            f"{r['pct_of_float32_pixels']:.1f}\\%",
            f"{r['pct_of_uint8_pixels']:.1f}\\%",
        )
        for r in a["storage"]
    ],
)

table(
    "TableSolvers",
    "llrrrr",
    ("solver", "data in", "fit (s)", "RAM (MB)", "variance", r"$\Delta$ test MSE"),
    [
        (
            tex_escape(r["solver"]).replace("'", r"\textquotesingle{}"),
            "RAM" if r["data"] == "in RAM" else "disk",
            f"{r['fit_seconds']:.2f}",
            f"{r['total_RAM_MB']:.0f}",
            f"{r['explained_variance']:.4f}",
            signed(r["mse_vs_exact_pct"]) + "\\%",
        )
        for r in b["solvers"]
    ],
)

table(
    "TableManifold",
    "lrrr",
    ("method", "time (s)", "downstream accuracy", r"$\max_j |\mathrm{corr}(z_j, t)|$"),
    [
        (
            r["method"],
            "$<$0.01" if r["seconds"] < 0.005 else f"{r['seconds']:.2f}",
            f"{100 * r['downstream_accuracy']:.1f}\\%",
            f"{r['max_abs_corr_with_t']:.3f}",
        )
        for r in c["methods"]
    ],
)

table(
    "TableMaps",
    "lrr",
    ("method", "time (s)", r"kNN accuracy ($\pm$ std over folds)"),
    [
        (
            r["method"],
            f"{r['seconds']:.1f}",
            f"{100 * r['knn_accuracy_2d']:.1f}\\% $\\pm$ {100 * r['knn_std']:.1f}",
        )
        for r in d["methods"]
    ]
    + [
        (
            r"\emph{no reduction: all 784 pixels}",
            "--",
            f"{100 * dm['knn_accuracy_784d']:.1f}\\% $\\pm$ {100 * dm['knn_std_784d']:.1f}",
        )
    ],
)

FEATS = {"raw": "784 pixels", "pca": "PCA", "pca_same_gamma": r"PCA, raw $\gamma$"}
table(
    "TableClassifiers",
    "llrrrrr",
    ("classifier", "features", "train", "fit (s)", "predict (s)", r"accuracy (95\% CI)", "MB"),
    [
        (
            r["classifier"].replace(" (RBF kernel)", ""),
            FEATS[r["features"]] + ("" if r["features"] == "raw" else f" ({r['n_features']})"),
            n(r["n_train"]),
            f"{r['fit_seconds']:.1f}",
            f"{r['predict_seconds_10k']:.2f}",
            f"{100 * r['test_accuracy']:.2f} ({100 * r['accuracy_ci_low']:.2f}--{100 * r['accuracy_ci_high']:.2f})",
            f"{r['model_MB']:.1f}",
        )
        for r in e["table"]
    ],
)

table(
    "TableDetector",
    "lrr",
    ("input (10{,}000 test images)", "flagged", "median error"),
    [(r["input"], f"{r['flagged_pct']:.1f}\\%", n(r["median_error"])) for r in fdet["corruptions"]],
)

table(
    "TableVarSweep",
    "rrrrr",
    ("variance", "$d$", "false alarms", "mean detection", "ROC-AUC"),
    [
        (
            f"{100 * r['variance']:.0f}\\%",
            str(r["n_components"]),
            f"{r['clean_false_alarm_pct']:.2f}\\%",
            f"{r['mean_detection_pct']:.1f}\\%",
            f"{r['roc_auc']:.3f}",
        )
        for r in fdet["variance_sweep"]
    ],
)

(HERE / "tables.tex").write_text("\n".join(T) + "\n")
print(f"wrote {len(vals)} numbers and 8 tables")
