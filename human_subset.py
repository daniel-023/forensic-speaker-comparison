import argparse
import csv
import os
import random


# Fixed design
NUM_PAIRS = 8
STRATA_COUNT = 4
REQUIRED_ETHNICITIES = ["CHINESE", "INDIAN", "MALAY"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsc-part", default=None, help="Single NSC part, e.g. pt1")
    parser.add_argument("--parts", nargs="+", default=["pt1", "pt2", "pt3"])
    parser.add_argument("--trials-root", default="trials")
    parser.add_argument("--out-dir", default="human subsets")
    parser.add_argument("--out-name", default="human.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ethnicity(stratum):
    return stratum.split(".")[-1]


def choose_strata(shared_strata, rng):
    # 1) Guarantee one stratum per required ethnicity.
    chosen = []
    for eth in REQUIRED_ETHNICITIES:
        candidates = [s for s in shared_strata if ethnicity(s) == eth]
        if not candidates:
            raise RuntimeError(f"No shared stratum available for ethnicity {eth}.")
        chosen.append(rng.choice(candidates))
    chosen = list(dict.fromkeys(chosen))  # keep order, drop duplicates if any

    # 2) Fill to target size at random.
    remaining = [s for s in shared_strata if s not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, STRATA_COUNT - len(chosen))])

    if len(chosen) < STRATA_COUNT:
        raise RuntimeError(
            f"Need at least {STRATA_COUNT} distinct shared strata, got {len(chosen)}."
        )
    return chosen[:STRATA_COUNT]


def sample_part(part, trials_root, out_dir, out_name, rng):
    in_path = os.path.join(trials_root, part, "trials.csv")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing trials file: {in_path}")

    with open(in_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {in_path}")

    # Build per-stratum pools for each label.
    pos = {}
    neg = {}
    for r in rows:
        s = r.get("stratum", "")
        if not s:
            continue
        if int(r["label"]) == 1:
            pos.setdefault(s, []).append(r)
        else:
            neg.setdefault(s, []).append(r)

    shared = sorted(set(pos) & set(neg))
    if len(shared) < STRATA_COUNT:
        raise RuntimeError(
            f"{part}: need >= {STRATA_COUNT} shared strata, found {len(shared)}."
        )

    strata = choose_strata(shared, rng)
    subset = []
    used = set()

    # One positive + one negative pair per chosen stratum.
    for s in strata:
        p = pos[s][:]
        n = neg[s][:]
        rng.shuffle(p)
        rng.shuffle(n)

        p_row = next((r for r in p if (r["utt1"], r["utt2"], r["label"]) not in used), None)
        n_row = next((r for r in n if (r["utt1"], r["utt2"], r["label"]) not in used), None)
        if p_row is None or n_row is None:
            raise RuntimeError(f"{part}: insufficient unique pairs in selected stratum {s}.")

        subset.append(p_row)
        subset.append(n_row)
        used.add((p_row["utt1"], p_row["utt2"], p_row["label"]))
        used.add((n_row["utt1"], n_row["utt2"], n_row["label"]))

    if len(subset) != NUM_PAIRS:
        raise RuntimeError(f"{part}: expected {NUM_PAIRS} pairs, got {len(subset)}.")

    rng.shuffle(subset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{part}_{out_name}")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(subset)

    covered = len(set(r.get("stratum", "") for r in subset))
    return out_path, covered


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    parts = [args.nsc_part] if args.nsc_part else args.parts

    for part in parts:
        out_path, covered = sample_part(part, args.trials_root, args.out_dir, args.out_name, rng)
        print(f"{part}: saved {NUM_PAIRS} pairs, {covered} strata covered -> {out_path}")


if __name__ == "__main__":
    main()
