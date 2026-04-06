import argparse
import csv
import shutil
from pathlib import Path


PARTS = ["pt1", "pt2", "pt3"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Root results directory")
    parser.add_argument("--audio-root", default="audio", help="Root audio directory")
    parser.add_argument(
        "--model",
        choices=["ecapa", "xvect"],
        default="ecapa",
        help="Model folder to read from (default: ecapa)",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional output CSV path. Default: results/closest_errors_<model>.csv",
    )
    parser.add_argument(
        "--export-audio-dir",
        default=None,
        help=(
            "Optional directory to copy selected error audio files. "
            "Default: results/closest_errors_<model>_audio"
        ),
    )
    parser.add_argument(
        "--allow-utt-overlap",
        action="store_true",
        help="Allow utt overlap between selected FP/FN in the same part.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def pair_key(row):
    return tuple(sorted((row["utt1"], row["utt2"])))


def utt_set(row):
    return {row["utt1"], row["utt2"]}


def select_part_errors(rows, allow_utt_overlap=False):
    """Select closest FP/FN for one part with optional no-overlap constraint."""
    false_pos = [r for r in rows if r.get("error_type") == "false_positive"]
    false_neg = [r for r in rows if r.get("error_type") == "false_negative"]

    false_pos.sort(key=lambda r: r["score"], reverse=True)  # highest score first
    false_neg.sort(key=lambda r: r["score"])                # lowest score first

    selected_fp = false_pos[0] if false_pos else None
    selected_fn = false_neg[0] if false_neg else None
    had_overlap_fallback = False

    if (
        selected_fp is not None
        and selected_fn is not None
        and not allow_utt_overlap
        and not utt_set(selected_fp).isdisjoint(utt_set(selected_fn))
    ):
        # Try to find the best non-overlapping combination.
        found = None
        for fp in false_pos:
            fp_utts = utt_set(fp)
            for fn in false_neg:
                if fp_utts.isdisjoint(utt_set(fn)):
                    found = (fp, fn)
                    break
            if found:
                break

        if found:
            selected_fp, selected_fn = found
        else:
            # Keep the extrema but report fallback.
            had_overlap_fallback = True

    return selected_fp, selected_fn, had_overlap_fallback


def build_utt_index(part_audio_root: Path):
    """Map utterance filename -> full path for one NSC part."""
    index = {}
    for wav_path in part_audio_root.rglob("*.wav"):
        name = wav_path.name
        if name in index:
            raise RuntimeError(
                f"Duplicate filename in {part_audio_root}: {name}\n"
                f" - {index[name]}\n"
                f" - {wav_path}"
            )
        index[name] = wav_path
    return index


def copy_selected_audio(selected, audio_root: Path, export_audio_dir: Path):
    if export_audio_dir.exists():
        shutil.rmtree(export_audio_dir)
    export_audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_fields = [
        "part",
        "selection_rule",
        "error_type",
        "score",
        "utt_role",
        "utt_name",
        "source_path",
        "copied_path",
    ]
    manifest_rows = []

    part_index_cache = {}

    for row in selected:
        part = row["part"]
        part_audio_root = audio_root / f"nsc_{part}_strata"
        if part not in part_index_cache:
            if not part_audio_root.exists():
                raise FileNotFoundError(f"Missing audio folder: {part_audio_root}")
            part_index_cache[part] = build_utt_index(part_audio_root)
        utt_index = part_index_cache[part]

        pair_dir = export_audio_dir / part / row["selection_rule"]
        pair_dir.mkdir(parents=True, exist_ok=True)

        for role in ("utt1", "utt2"):
            utt_name = row[role]
            if utt_name not in utt_index:
                raise FileNotFoundError(
                    f"Could not find {utt_name} under {part_audio_root}"
                )
            src = utt_index[utt_name]
            dst = pair_dir / f"{role}_{utt_name}"
            shutil.copy2(src, dst)
            manifest_rows.append(
                {
                    "part": part,
                    "selection_rule": row["selection_rule"],
                    "error_type": row["error_type"],
                    "score": row["score"],
                    "utt_role": role,
                    "utt_name": utt_name,
                    "source_path": str(src),
                    "copied_path": str(dst),
                }
            )

    manifest_path = export_audio_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path, len(manifest_rows)


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    audio_root = Path(args.audio_root)
    out_csv = Path(args.out_csv) if args.out_csv else results_root / f"closest_errors_{args.model}.csv"
    export_audio_dir = (
        Path(args.export_audio_dir)
        if args.export_audio_dir
        else results_root / f"closest_errors_{args.model}_audio"
    )

    selected = []
    missing_parts = []
    overlap_fallback_parts = []
    seen_pair_keys = set()

    for part in PARTS:
        mispred_path = results_root / part / args.model / "mispredicted_pairs.csv"
        if not mispred_path.exists():
            missing_parts.append(part)
            continue

        rows = load_rows(mispred_path)
        if not rows:
            continue

        for row in rows:
            row["score"] = float(row["score"])

        best_fp, best_fn, had_overlap_fallback = select_part_errors(
            rows, allow_utt_overlap=args.allow_utt_overlap
        )
        if had_overlap_fallback:
            overlap_fallback_parts.append(part)

        if best_fp:
            pk = (part, pair_key(best_fp))
            if pk not in seen_pair_keys:
                best_fp["part"] = part
                best_fp["selection_rule"] = "false_positive_max_score"
                selected.append(best_fp)
                seen_pair_keys.add(pk)

        if best_fn:
            pk = (part, pair_key(best_fn))
            if pk not in seen_pair_keys:
                best_fn["part"] = part
                best_fn["selection_rule"] = "false_negative_min_score"
                selected.append(best_fn)
                seen_pair_keys.add(pk)

    if not selected:
        raise RuntimeError("No selected errors found. Check results path/model.")

    field_order = [
        "part",
        "selection_rule",
        "error_type",
        "score",
        "threshold",
        "label",
        "prediction",
        "utt1",
        "utt2",
        "stratum",
        "gender",
        "age_bin",
        "ethnicity",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Saved {len(selected)} rows to {out_csv}")

    manifest_path, copied_count = copy_selected_audio(selected, audio_root, export_audio_dir)
    print(f"Copied {copied_count} audio files to {export_audio_dir}")
    print(f"Saved audio manifest to {manifest_path}")

    if missing_parts:
        print(f"Skipped parts (missing file): {', '.join(missing_parts)}")
    if overlap_fallback_parts:
        print(
            "Warning: Could not avoid utt overlap in parts: "
            + ", ".join(overlap_fallback_parts)
            + " (used strict extrema)."
        )

    for row in selected:
        print(
            f"{row['part']} | {row['selection_rule']} | score={row['score']:.6f} | "
            f"{row['utt1']} vs {row['utt2']}"
        )


if __name__ == "__main__":
    main()
