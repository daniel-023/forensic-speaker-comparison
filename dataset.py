import argparse
import csv
import os
import random
from itertools import combinations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsc-part", default="pt1", help="NSC part tag, e.g. pt1, pt2")
    parser.add_argument("--audio-root", default="audio", help="Root folder containing NSC audio")
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Directory containing stratum subfolders. Defaults to audio/nsc_<part>_strata",
    )
    parser.add_argument("--strata-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", default="trials", help="Root output dir for trials")
    parser.add_argument("--impostors-per-utt", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    corpus_dir = (
        args.corpus_dir
        or args.strata_root
        or os.path.join(args.audio_root, f"nsc_{args.nsc_part}_strata")
    )
    impostors_per_utt = args.impostors_per_utt

    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    all_trials = []
    total_target = 0
    total_nontarget = 0

    strata_dirs = [
        d
        for d in sorted(os.listdir(corpus_dir))
        if os.path.isdir(os.path.join(corpus_dir, d))
    ]
    if not strata_dirs:
        raise RuntimeError(f"No stratum directories found under {corpus_dir}")

    for stratum in strata_dirs:
        stratum_dir = os.path.join(corpus_dir, stratum)
        utterances = {}

        for wav in sorted(os.listdir(stratum_dir)):
            if wav.endswith(".wav"):
                speaker_id = wav.split("_")[0]
                utterances.setdefault(speaker_id, []).append(wav)

        if not utterances:
            continue

        print(f"\nStratum: {stratum}")
        for spk, utts in utterances.items():
            print(f"Speaker {spk} has {len(utts)} utterances.")

        target_trials = []
        for _, utts in utterances.items():
            for utt1, utt2 in combinations(utts, 2):
                target_trials.append((utt1, utt2, 1, stratum))

        nontarget_trials = []
        speakers = list(utterances.keys())
        for spk in speakers:
            for utt in utterances[spk]:
                candidate_impostors = [s for s in speakers if s != spk]
                k = min(impostors_per_utt, len(candidate_impostors))
                for imp in random.sample(candidate_impostors, k):
                    imp_utt = random.choice(utterances[imp])
                    nontarget_trials.append((utt, imp_utt, 0, stratum))

        total_target += len(target_trials)
        total_nontarget += len(nontarget_trials)
        all_trials.extend(target_trials)
        all_trials.extend(nontarget_trials)

        print(
            f"Generated {len(target_trials)} target and {len(nontarget_trials)} nontarget "
            f"trials for {stratum}."
        )

    print(
        f"\nGenerated {total_target} target trials and {total_nontarget} nontarget trials "
        f"(total={len(all_trials)})."
    )

    out_dir = os.path.join(args.output_root, args.nsc_part)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "trials.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utt1", "utt2", "label", "stratum"])
        writer.writerows(all_trials)

    print(f"Saved combined strata trials to {out_path}")


if __name__ == "__main__":
    main()
