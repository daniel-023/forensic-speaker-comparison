"""
Select one correctly-predicted same-speaker pair and one correctly-predicted
different-speaker pair per NSC part (pt1, pt2, pt3), then populate demo/samples/.

A pair is "correctly predicted" if it does NOT appear in the mispredicted_pairs.csv
for EITHER the ecapa OR xvect model.

Run from the project root:
    python demo/select_pairs.py
"""
import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_ROOT = PROJECT_ROOT / "audio"
TRIALS_ROOT = PROJECT_ROOT / "trials"
RESULTS_ROOT = PROJECT_ROOT / "results"
SAMPLES_ROOT = Path(__file__).resolve().parent / "samples"

PARTS = ["pt1", "pt2", "pt3"]
MODELS = ["ecapa", "xvect"]
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}

# Target strata chosen for demographic diversity across the 6 demo pairs.
# Balance: M×3 / F×3, MALAY×2 / INDIAN×2 / CHINESE×2, varied age groups.
# Falls back to first available correct pair if the target stratum has none.
TARGET_STRATA = {
    ("pt1", "same"):      "nsc_pt1.M.(24,31].MALAY",
    ("pt1", "different"): "nsc_pt1.F.(31,44].INDIAN",
    ("pt2", "same"):      "nsc_pt2.F.(32,43].CHINESE",
    ("pt2", "different"): "nsc_pt2.M.(24,32].MALAY",
    ("pt3", "same"):      "nsc_pt3.F.35_45.INDIAN",
    ("pt3", "different"): "nsc_pt3.M.18_25.CHINESE",
}


def parse_stratum(stratum: str) -> dict:
    """Parse 'nsc_pt1.F.(21,24].CHINESE' → {gender, age_bin, ethnicity}."""
    parts = stratum.strip('"').split(".")
    # parts: ['nsc_pt1', 'F', '(21,24]', 'CHINESE']
    return {"gender": parts[1], "age_bin": parts[2], "ethnicity": parts[3]}


def audio_path(part: str, stratum: str, utt: str) -> Path:
    folder = AUDIO_ROOT / f"nsc_{part}_strata" / stratum
    return folder / utt


def build_exclusion_set(part: str) -> set:
    excluded = set()
    for model in MODELS:
        mp_path = RESULTS_ROOT / part / model / "mispredicted_pairs.csv"
        if not mp_path.exists():
            print(f"  WARNING: {mp_path} not found, skipping")
            continue
        mp = pd.read_csv(mp_path)
        for _, row in mp.iterrows():
            excluded.add(frozenset([row["utt1"], row["utt2"]]))
    return excluded


def select_and_copy():
    for part in PARTS:
        print(f"\n=== {part} ===")

        trials_path = TRIALS_ROOT / part / "trials.csv"
        if not trials_path.exists():
            print(f"  ERROR: {trials_path} not found")
            continue

        trials = pd.read_csv(trials_path)
        excluded = build_exclusion_set(part)
        print(f"  Excluded pairs (mispredicted by either model): {len(excluded)}")

        mask = ~trials.apply(
            lambda r: frozenset([r["utt1"], r["utt2"]]) in excluded, axis=1
        )
        good = trials[mask]
        print(f"  Good pairs available: {len(good)} / {len(trials)}")

        for label_val, label_name in [(1, "same"), (0, "different")]:
            candidates = good[good["label"] == label_val]
            if candidates.empty:
                print(f"  WARNING: No good candidates for {label_name}")
                continue

            target = TARGET_STRATA.get((part, label_name))
            if target:
                targeted = candidates[candidates["stratum"].str.strip('"') == target]
                if not targeted.empty:
                    candidates = targeted
                else:
                    print(f"  WARNING: No correct pairs in target stratum {target}, using fallback")

            row = candidates.iloc[0]
            stratum = row["stratum"].strip('"')
            utt1, utt2 = row["utt1"], row["utt2"]

            src_a = audio_path(part, stratum, utt1)
            src_b = audio_path(part, stratum, utt2)

            if not src_a.exists():
                print(f"  WARNING: Audio not found: {src_a}")
                continue
            if not src_b.exists():
                print(f"  WARNING: Audio not found: {src_b}")
                continue

            dest_dir = SAMPLES_ROOT / f"{part}_{label_name}"
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Remove existing audio files before copying new ones
            for f in dest_dir.iterdir():
                if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                    f.unlink()

            shutil.copy2(src_a, dest_dir / utt1)
            shutil.copy2(src_b, dest_dir / utt2)

            demographics = parse_stratum(stratum)
            metadata = {
                "part": part,
                "label": label_name,
                "stratum": stratum,
                "utt_a": utt1,
                "utt_b": utt2,
                **demographics,
            }
            (dest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

            print(f"  {label_name}: {utt1} vs {utt2}  [{stratum}]")


if __name__ == "__main__":
    select_and_copy()
    print("\nDone.")
