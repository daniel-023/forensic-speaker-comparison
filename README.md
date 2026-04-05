# Forensic Speaker Comparison

Lightweight pipeline for speaker-pair generation, embedding extraction, evaluation outputs, and model comparison plots across NSC parts (`pt1`-`pt3`).

## Pipeline

![Speaker comparison pipeline](figures/speaker_comparison.png)

## Project Layout

- `audio/nsc_pt{1,2,3}_strata/`: stratified audio folders
- `trials/<part>/trials.csv`: generated target/non-target pairs
- `embeddings/<part>/<model>/embeddings.pt`: extracted speaker embeddings
- `results/<part>/<model>/metrics.json`: per-model metrics by NSC part
- `results/*.png`: cross-part comparison plots

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

1. Generate trials for one part:

```bash
python dataset.py --nsc-part pt1 --audio-root audio
```

2. Extract embeddings (both models):

```bash
python inference.py --nsc-part pt1 --audio-root audio --model all
```

3. Plot model comparison from existing `results/*/*/metrics.json`:

```bash
python plot_eval.py --results-root results --out-dir results --models xvect ecapa
```

4. Build human listening subset CSVs (default: all 3 parts):

```bash
python human_subset.py --trials-root trials --out-dir "human subsets"
```

## Notes

- Parts are fixed to `pt1`, `pt2`, `pt3` in plotting scripts.
- Model keys: `ecapa` and `xvect`.
- `test.py` is a quick cosine-similarity sanity check on sample files.

## References

- https://huggingface.co/blog/norwooodsystems/ecapa-vs-xvector-speaker-recognition-comparison
