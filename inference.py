import os
import argparse
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from pathlib import Path

MODEL_SOURCES = {
    "ecapa": "speechbrain/spkrec-ecapa-voxceleb",
    "xvect": "speechbrain/spkrec-xvect-voxceleb",
}

_CLASSIFIER_CACHE = {}


def get_classifier(model_name, savedir_root="pretrained_models"):
    if model_name not in MODEL_SOURCES:
        raise ValueError(f"Unsupported model: {model_name}")
    if model_name not in _CLASSIFIER_CACHE:
        source = MODEL_SOURCES[model_name]
        savedir = os.path.join(savedir_root, source.split("/")[-1])
        _CLASSIFIER_CACHE[model_name] = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
        )
    return _CLASSIFIER_CACHE[model_name]


def load_and_prepare_audio(wav_path, target_sr=16000):
    signal, fs = torchaudio.load(wav_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if fs != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sr)
        signal = resampler(signal)
    return signal


def run_inference(wav_path, model_name="ecapa", savedir_root="pretrained_models"):
    signal = load_and_prepare_audio(wav_path)
    classifier = get_classifier(model_name, savedir_root=savedir_root)
    with torch.no_grad():
        emb = classifier.encode_batch(signal)  # type: ignore
    return emb.squeeze()


def ecapa_inference(wav_path):
    return run_inference(wav_path, model_name="ecapa")


def xvect_inference(wav_path):
    return run_inference(wav_path, model_name="xvect")


def build_embeddings(
    corpus_dir,
    nsc_part,
    model_name,
    output_root="embeddings",
    savedir_root="pretrained_models",
):
    embeddings = {}
    for wav in sorted(Path(corpus_dir).rglob("*.wav")):
        utt_name = wav.name
        if utt_name in embeddings:
            raise ValueError(f"Duplicate utterance filename found across strata: {utt_name}")
        embeddings[utt_name] = run_inference(
            str(wav), model_name=model_name, savedir_root=savedir_root
        )
        print(f"[{model_name}] Processed {utt_name}")

    out_dir = os.path.join(output_root, nsc_part, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "embeddings.pt")
    torch.save(embeddings, out_path)
    print(f"Saved {len(embeddings)} embeddings to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["ecapa", "xvect", "all"],
        default="ecapa",
        help="Which inference model to run.",
    )
    parser.add_argument("--nsc-part", default="pt1", help="NSC part tag, e.g. pt1, pt2")
    parser.add_argument("--audio-root", default="audio", help="Root folder containing NSC audio")
    # Backward-compatible alias for older commands.
    parser.add_argument("--part", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Directory containing .wav files. Defaults to nsc_<part>_strata",
    )
    parser.add_argument(
        "--output-root",
        default="embeddings",
        help="Root directory where model embeddings are saved.",
    )
    parser.add_argument(
        "--savedir-root",
        default="pretrained_models",
        help="Root directory for downloaded SpeechBrain models.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    nsc_part = args.nsc_part or args.part or "pt1"
    corpus_dir = args.corpus_dir or os.path.join(args.audio_root, f"nsc_{nsc_part}_strata")
    models = ["ecapa", "xvect"] if args.model == "all" else [args.model]
    for model_name in models:
        build_embeddings(
            corpus_dir=corpus_dir,
            nsc_part=nsc_part,
            model_name=model_name,
            output_root=args.output_root,
            savedir_root=args.savedir_root,
        )


if __name__ == "__main__":
    main()
