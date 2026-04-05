import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODEL_A_COLOR = "#7C83FD"  # periwinkle
MODEL_B_COLOR = "#2CB67D"  # teal green
PARTS = ["pt1", "pt2", "pt3"]

MODEL_LABELS = {
    "xvect": "X-vector",
    "ecapa": "ECAPA-TDNN",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Root results directory")
    parser.add_argument("--out-dir", default="results", help="Output directory for plots")
    parser.add_argument(
        "--models",
        nargs=2,
        default=["xvect", "ecapa"],
        help="Two model folder names to compare (default: xvect ecapa)",
    )
    return parser.parse_args()


def load_metric(results_root, part, model, metric):
    metrics_path = Path(results_root) / part / model / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        data = json.load(f)
    if metric not in data:
        return None
    return float(data[metric])


def model_label(model_name):
    return MODEL_LABELS.get(model_name, model_name.upper())


def style_axes(ax, ylabel):
    ax.set_xlabel("NSC Part")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_value_labels(ax, bars, y_offset):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_offset,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def set_percent_axis(ax, values, factor=1.25):
    vmax = max(values) if values else 0.0
    upper = vmax * factor if vmax > 0 else 1.0
    ax.set_ylim(0, upper)


def collect_metric_series(results_root, model_a, model_b, metric):
    a_vals = []
    b_vals = []
    kept_parts = []
    for part in PARTS:
        a = load_metric(results_root, part, model_a, metric)
        b = load_metric(results_root, part, model_b, metric)
        if a is None or b is None:
            continue
        kept_parts.append(part.upper())
        a_vals.append(a * 100.0)
        b_vals.append(b * 100.0)
    return kept_parts, a_vals, b_vals


def _render_bar_subplot(ax, parts, vals_a, vals_b, model_a, model_b, ylabel, y_offset, shared_vals=None):
    """Render a single grouped bar subplot. Title and figure layout are the caller's responsibility."""
    x = np.arange(len(parts))
    width = 0.36

    bars_a = ax.bar(x - width / 2, vals_a, width=width, label=model_label(model_a), color=MODEL_A_COLOR)
    bars_b = ax.bar(x + width / 2, vals_b, width=width, label=model_label(model_b), color=MODEL_B_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(parts)
    style_axes(ax, ylabel)
    set_percent_axis(ax, (shared_vals if shared_vals is not None else vals_a + vals_b), factor=1.18)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    add_value_labels(ax, bars_a, y_offset=y_offset)
    add_value_labels(ax, bars_b, y_offset=y_offset)

    return bars_a, bars_b


def plot_grouped_metric(
    results_root,
    out_dir,
    model_a,
    model_b,
    metric_key,
    title,
    ylabel,
    out_filename,
    y_offset,
):
    parts, vals_a, vals_b = collect_metric_series(results_root, model_a, model_b, metric_key)
    if not parts:
        raise RuntimeError(f"No parts with {metric_key} for both models.")

    fig, ax = plt.subplots(figsize=(9, 5))
    _render_bar_subplot(ax, parts, vals_a, vals_b, model_a, model_b, ylabel, y_offset)

    ax.set_title(title, pad=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(out_dir) / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fpr_fnr(results_root, out_dir, model_a, model_b):
    parts_fpr, fpr_a, fpr_b = collect_metric_series(results_root, model_a, model_b, "FPR")
    parts_fnr, fnr_a, fnr_b = collect_metric_series(results_root, model_a, model_b, "FNR")
    if not parts_fpr or not parts_fnr:
        raise RuntimeError("No parts with FPR/FNR for both models.")

    parts = [p.upper() for p in PARTS if p.upper() in parts_fpr and p.upper() in parts_fnr]

    idx_map_fpr = {p: i for i, p in enumerate(parts_fpr)}
    idx_map_fnr = {p: i for i, p in enumerate(parts_fnr)}
    fpr_a = [fpr_a[idx_map_fpr[p]] for p in parts]
    fpr_b = [fpr_b[idx_map_fpr[p]] for p in parts]
    fnr_a = [fnr_a[idx_map_fnr[p]] for p in parts]
    fnr_b = [fnr_b[idx_map_fnr[p]] for p in parts]

    shared_vals = fpr_a + fpr_b + fnr_a + fnr_b

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    _render_bar_subplot(
        axes[0], parts, fpr_a, fpr_b, model_a, model_b,
        ylabel="Rate (%)", y_offset=0.08, shared_vals=shared_vals,
    )
    _render_bar_subplot(
        axes[1], parts, fnr_a, fnr_b, model_a, model_b,
        ylabel="Rate (%)", y_offset=0.08, shared_vals=shared_vals,
    )

    axes[0].set_title("FPR by NSC Part", pad=8)
    axes[1].set_title("FNR by NSC Part", pad=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(out_dir) / "fpr_fnr_bar.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    model_a, model_b = args.models

    plot_grouped_metric(
        args.results_root,
        args.out_dir,
        model_a,
        model_b,
        metric_key="EER",
        title="EER by NSC Part",
        ylabel="EER (%)",
        out_filename="eer_grouped_bar.png",
        y_offset=0.08,
    )
    plot_grouped_metric(
        args.results_root,
        args.out_dir,
        model_a,
        model_b,
        metric_key="accuracy",
        title="Accuracy by NSC Part",
        ylabel="Accuracy (%)",
        out_filename="accuracy_grouped_bar.png",
        y_offset=0.06,
    )
    plot_fpr_fnr(args.results_root, args.out_dir, model_a, model_b)


if __name__ == "__main__":
    main()