import json
import random
import re
from pathlib import Path
import streamlit as st
import torch
from inference import run_inference



RESULTS_ROOT = Path("./results")
PRETRAINED_ROOT = Path("./pretrained_models")
SAMPLES_ROOT = Path("./demo/samples")

PARTS = ["pt1", "pt2", "pt3"]
MODELS = ["ecapa", "xvect"]
DEFAULT_THRESHOLDS = {"ecapa": 0.45, "xvect": 0.95}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}
MODEL_LABELS = {"ecapa": "ECAPA-TDNN", "xvect": "X-Vector"}
PART_LABELS = {"pt1": "Part 1", "pt2": "Part 2", "pt3": "Part 3"}


def load_threshold(model: str, part: str, results_root: Path = RESULTS_ROOT):
    metrics_path = results_root / part / model / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text()).get("EER_threshold")
    except Exception:
        return None


def score_audio_paths(path_a: Path, path_b: Path, model: str):
    emb_a = run_inference(str(path_a), model_name=model, savedir_root=str(PRETRAINED_ROOT))
    emb_b = run_inference(str(path_b), model_name=model, savedir_root=str(PRETRAINED_ROOT))
    return torch.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()


def load_sample_pairs(samples_root: Path = SAMPLES_ROOT):
    pairs = []
    if not samples_root.exists():
        return pairs
    for d in sorted(samples_root.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
        if len(files) < 2:
            continue
        folder_name = d.name.lower()
        label = "same" if folder_name.endswith("_same") else ("different" if folder_name.endswith("_different") else "")
        try:
            metadata = json.loads((d / "metadata.json").read_text())
        except Exception:
            metadata = {}
        part_from_name = folder_name.split("_")[0] if "_" in folder_name else ""
        pairs.append({
            "pair_id": d.name,
            "file_a": files[0],
            "file_b": files[1],
            "label": label,
            "part": metadata.get("part", part_from_name),
            "gender": metadata.get("gender", ""),
            "age_bin": metadata.get("age_bin", ""),
            "ethnicity": metadata.get("ethnicity", ""),
        })
    return pairs


def render_result(score: float, threshold: float, true_label: str = ""):
    is_match = score >= threshold
    c1, c2, c3 = st.columns(3)
    c1.metric("Similarity score", f"{score:.4f}")
    c2.metric("Threshold", f"{threshold:.4f}")
    c3.metric("Margin", f"{abs(score - threshold):.4f}")
    if is_match:
        st.success("Decision: Same speaker")
    else:
        st.error("Decision: Different speakers")
    if true_label:
        st.caption(f"Ground truth: **{true_label.title()} speaker**")
    st.caption("Rule used: predict same-speaker if score ≥ threshold, otherwise different-speaker.")


def main():
    st.set_page_config(page_title="Speaker Comparison Demo", page_icon="🎙️", layout="centered")
    st.title("🎙️ Speaker Comparison Demo")
    st.caption("Automatic forensic speaker comparison using ECAPA-TDNN and X-Vector models evaluated on the National Speech Corpus.")

    with st.sidebar:
        st.subheader("Settings")
        model = st.radio("Model", MODELS, format_func=lambda x: MODEL_LABELS[x], horizontal=True)
        part = st.radio("NSC Part", PARTS, format_func=lambda x: PART_LABELS[x], horizontal=True)

        auto_threshold = load_threshold(model, part)
        use_auto = st.toggle(
            "Use EER threshold from results",
            value=True,
            disabled=(auto_threshold is None),
            help="Reads results/<part>/<model>/metrics.json",
        )
        if auto_threshold is None:
            st.warning("No metrics threshold found. Using manual threshold.")
        threshold_default = auto_threshold if (use_auto and auto_threshold is not None) else DEFAULT_THRESHOLDS[model]
        threshold = st.slider("Decision threshold", min_value=-1.0, max_value=1.0,
                              value=float(threshold_default), step=0.001)

        st.divider()
        hide_labels = st.toggle("Hide pair labels", value=True,
                                help="Shows 'Pair 1/2' instead of Same/Different — for live demos.")

    all_pairs = load_sample_pairs()
    sample_pairs = [p for p in all_pairs if p.get("part") == part]
    if not sample_pairs:
        st.info(f"No sample pairs found for {part}. Run demo/select_pairs.py to populate demo/samples.")
        return

    # Shuffle order once per part so position doesn't reveal the label.
    shuffle_key = f"pair_order_{part}"
    if shuffle_key not in st.session_state:
        order = list(range(len(sample_pairs)))
        random.shuffle(order)
        st.session_state[shuffle_key] = order
    display_pairs = [sample_pairs[i] for i in st.session_state[shuffle_key]]

    idx = st.radio(
        "Sample pair",
        options=list(range(len(display_pairs))),
        format_func=lambda i: f"Pair {i + 1}" if hide_labels else (display_pairs[i]["label"].title() or display_pairs[i]["pair_id"]),
        horizontal=True,
    )
    selected = display_pairs[idx]

    if selected.get("gender"):
        st.caption("Speaker demographics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Gender", selected["gender"])
        c2.metric("Age group", selected["age_bin"])
        c3.metric("Ethnicity", selected["ethnicity"].title())

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Speaker A")
        st.audio(str(selected["file_a"]))
    with col2:
        st.caption("Speaker B")
        st.audio(str(selected["file_b"]))

    if st.button("Compare", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Running {MODEL_LABELS[model]} inference..."):
                score = score_audio_paths(selected["file_a"], selected["file_b"], model)
            render_result(score, threshold, true_label=selected["label"] if hide_labels else "")
        except Exception as exc:
            st.error(f"Inference failed: {exc}")


if __name__ == "__main__":
    main()
