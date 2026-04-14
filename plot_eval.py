import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODEL_A_COLOR = "#7C83FD"  # periwinkle
MODEL_B_COLOR = "#2CB67D"  # teal green
HUMAN_COLOR = "#F4A261"    # warm orange
PARTS = ["pt1", "pt2", "pt3"]

MODEL_LABELS = {
    "xvect": "X-vector",
    "ecapa": "ECAPA-TDNN",
    "human": "Human",
}
HUMAN_PART_METRICS = {
    "FPR": {"pt1": 0.125, "pt2": 0.1522, "pt3": 0.1842},
    "FNR": {"pt1": 0.125, "pt2": 0.0417, "pt3": 0.0833},
    "Accuracy": {"pt1": 0.875, "pt2": 0.904, "pt3": 0.872},
    "EER": {"pt1": 0.1310, "pt2": 0.0966, "pt3": 0.1635}
}
ETHNICITIES = ["CHINESE", "INDIAN", "MALAY"]
ETHNICITY_COLORS = {
    "CHINESE": "#A7C7E7",  # pastel blue
    "INDIAN": "#FFCF99",   # pastel orange
    "MALAY": "#B7E4C7",    # pastel green
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Root results directory")
    parser.add_argument("--trials-root", default="trials", help="Root trials directory")
    parser.add_argument("--out-dir", default="results", help="Output directory for plots")
    parser.add_argument(
        "--models",
        nargs=2,
        default=["xvect", "ecapa"],
        help="Two model folder names to compare (default: xvect ecapa)",
    )
    parser.add_argument(
        "--ethnicity-model",
        default="ecapa",
        choices=["ecapa", "xvect"],
        help="Model used for FP/FN by ethnicity plot.",
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


def collect_metric_by_part(results_root, models, metric):
    by_part = {}
    for part in PARTS:
        vals = []
        for model in models:
            v = load_metric(results_root, part, model, metric)
            if v is None:
                vals = None
                break
            vals.append(v * 100.0)
        if vals is not None:
            by_part[part.upper()] = vals
    return by_part


def load_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_ethnicity(row):
    eth = row.get("ethnicity")
    if eth:
        return eth.strip().upper()
    stratum = row.get("stratum", "")
    if "." in stratum:
        return stratum.split(".")[-1].strip().upper()
    return ""


def plot_grouped_series(ax, parts, series, ylabel, y_offset, shared_vals=None, legend_y=1.02):
    """
    Render grouped bar series on one axis.
    series: list of (label, values, color)
    """
    x = np.arange(len(parts))
    n_series = max(1, len(series))
    width = 0.8 / n_series
    start = -width * (n_series - 1) / 2.0
    all_vals = []

    for i, (label, vals, color) in enumerate(series):
        bars = ax.bar(x + start + i * width, vals, width=width, label=label, color=color)
        add_value_labels(ax, bars, y_offset=y_offset)
        all_vals.extend(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(parts)
    style_axes(ax, ylabel)
    set_percent_axis(ax, (shared_vals if shared_vals is not None else all_vals), factor=1.18)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=len(series))


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
    include_human=False,
    human_metric_key=None,
):
    by_part = collect_metric_by_part(results_root, [model_a, model_b], metric_key)
    parts = [p.upper() for p in PARTS if p.upper() in by_part]
    if not parts:
        raise RuntimeError(f"No parts with {metric_key} for both models.")
    vals_a = [by_part[p][0] for p in parts]
    vals_b = [by_part[p][1] for p in parts]

    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        (model_label(model_a), vals_a, MODEL_A_COLOR),
        (model_label(model_b), vals_b, MODEL_B_COLOR),
    ]
    if include_human:
        if not human_metric_key or human_metric_key not in HUMAN_PART_METRICS:
            raise RuntimeError(f"Missing human metric key for {metric_key}.")
        vals_h = [HUMAN_PART_METRICS[human_metric_key][p.lower()] * 100.0 for p in parts]
        series.append((model_label("human"), vals_h, HUMAN_COLOR))

    plot_grouped_series(ax, parts, series, ylabel=ylabel, y_offset=y_offset)
    ax.set_title(title, pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(out_dir) / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fpr_fnr(results_root, out_dir, model_a, model_b):
    fpr_by_part = collect_metric_by_part(results_root, [model_a, model_b], "FPR")
    fnr_by_part = collect_metric_by_part(results_root, [model_a, model_b], "FNR")
    if not fpr_by_part or not fnr_by_part:
        raise RuntimeError("No parts with FPR/FNR for both models.")

    parts = [p.upper() for p in PARTS if p.upper() in fpr_by_part and p.upper() in fnr_by_part]
    fpr_a = [fpr_by_part[p][0] for p in parts]
    fpr_b = [fpr_by_part[p][1] for p in parts]
    fnr_a = [fnr_by_part[p][0] for p in parts]
    fnr_b = [fnr_by_part[p][1] for p in parts]

    fpr_h = [HUMAN_PART_METRICS["FPR"][p.lower()] * 100.0 for p in parts]
    fnr_h = [HUMAN_PART_METRICS["FNR"][p.lower()] * 100.0 for p in parts]
    shared_vals = fpr_a + fpr_b + fpr_h + fnr_a + fnr_b + fnr_h

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fpr_series = [
        (model_label(model_a), fpr_a, MODEL_A_COLOR),
        (model_label(model_b), fpr_b, MODEL_B_COLOR),
        (model_label("human"), fpr_h, HUMAN_COLOR),
    ]
    fnr_series = [
        (model_label(model_a), fnr_a, MODEL_A_COLOR),
        (model_label(model_b), fnr_b, MODEL_B_COLOR),
        (model_label("human"), fnr_h, HUMAN_COLOR),
    ]
    plot_grouped_series(axes[0], parts, fpr_series, ylabel="Rate (%)", y_offset=0.08, shared_vals=shared_vals)
    plot_grouped_series(axes[1], parts, fnr_series, ylabel="Rate (%)", y_offset=0.08, shared_vals=shared_vals)

    axes[0].set_title("FPR by NSC Part", pad=8)
    axes[1].set_title("FNR by NSC Part", pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(out_dir) / "fpr_fnr_bar.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fp_fn_by_ethnicity(results_root, trials_root, out_dir, model):
    # Denominators from all trials (per label, per ethnicity).
    pos_den = {part: {eth: 0 for eth in ETHNICITIES} for part in PARTS}
    neg_den = {part: {eth: 0 for eth in ETHNICITIES} for part in PARTS}
    # Numerators from mispredictions.
    fp_num = {part: {eth: 0 for eth in ETHNICITIES} for part in PARTS}
    fn_num = {part: {eth: 0 for eth in ETHNICITIES} for part in PARTS}

    for part in PARTS:
        trials_path = Path(trials_root) / part / "trials.csv"
        mispred_path = Path(results_root) / part / model / "mispredicted_pairs.csv"
        if not trials_path.exists():
            raise FileNotFoundError(f"Missing trials file: {trials_path}")
        if not mispred_path.exists():
            raise FileNotFoundError(f"Missing mispredictions file: {mispred_path}")

        for row in load_csv_rows(trials_path):
            eth = parse_ethnicity(row)
            if eth not in ETHNICITIES:
                continue
            if int(row["label"]) == 1:
                pos_den[part][eth] += 1
            else:
                neg_den[part][eth] += 1

        for row in load_csv_rows(mispred_path):
            eth = parse_ethnicity(row)
            if eth not in ETHNICITIES:
                continue
            err = row.get("error_type", "").strip().lower()
            if err == "false_positive":
                fp_num[part][eth] += 1
            elif err == "false_negative":
                fn_num[part][eth] += 1

    # Convert to rates in percent.
    fp_rates = {
        eth: [
            (100.0 * fp_num[part][eth] / neg_den[part][eth]) if neg_den[part][eth] > 0 else 0.0
            for part in PARTS
        ]
        for eth in ETHNICITIES
    }
    fn_rates = {
        eth: [
            (100.0 * fn_num[part][eth] / pos_den[part][eth]) if pos_den[part][eth] > 0 else 0.0
            for part in PARTS
        ]
        for eth in ETHNICITIES
    }

    x = np.arange(len(PARTS))
    width = 0.24
    offsets = [-width, 0.0, width]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for idx, eth in enumerate(ETHNICITIES):
        axes[0].bar(
            x + offsets[idx],
            fp_rates[eth],
            width=width,
            label=eth.title(),
            color=ETHNICITY_COLORS[eth],
        )
        axes[1].bar(
            x + offsets[idx],
            fn_rates[eth],
            width=width,
            label=eth.title(),
            color=ETHNICITY_COLORS[eth],
        )

    for ax, title in zip(axes, ["False Positive Rate by Ethnicity", "False Negative Rate by Ethnicity"]):
        ax.set_xticks(x)
        ax.set_xticklabels([p.upper() for p in PARTS])
        ax.set_xlabel("NSC Part")
        ax.set_ylabel("Rate (%)")
        ax.set_title(f"{title} ({model_label(model)})")
        ax.grid(axis="y", alpha=0.22, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper right")

    ymax = max([v for vals in fp_rates.values() for v in vals] + [v for vals in fn_rates.values() for v in vals] + [1.0])
    for ax in axes:
        ax.set_ylim(0, min(100.0, ymax * 1.2))

    out_path = Path(out_dir) / f"fp_fn_by_ethnicity_{model}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
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
        include_human=True,
        human_metric_key="EER",
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
        include_human=True,
        human_metric_key="Accuracy",
    )
    plot_fpr_fnr(args.results_root, args.out_dir, model_a, model_b)
    plot_fp_fn_by_ethnicity(
        args.results_root,
        args.trials_root,
        args.out_dir,
        args.ethnicity_model,
    )


if __name__ == "__main__":
    main()
