"""Small Gradio interface for windowed Filipino speech deepfake detection."""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

from improvedCQCC_config import HOP_SAMPLES, SAMPLE_RATE, WIN_SAMPLES
from improvedCQCC_augmentation import peak_normalize
from improvedCQCC_features import extract_cqcc, extract_f0, extract_logmel
from improvedCQCC_train import CNNGRUOnly
from improvedCQCC_cqcc_train import CQCCMLP
from improvedCQCC_f0_train import F0MLP, F0_DIM

MAX_BYTES = 10 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent
PROPOSED_MODEL = "CNN–GRU + CQCC + F0 (Prediction-Level Fusion + Causal Temporal Correction)"
MODEL_CHOICES = [PROPOSED_MODEL, "CNN–GRU baseline", "CQCC expert", "F0 expert"]
SINGLE_MODELS = {
    "CNN–GRU baseline": (CNNGRUOnly, "logmel", PROJECT_ROOT / "CNN-GRU" / "final_model.pt"),
    "CQCC expert": (CQCCMLP, "cqcc", PROJECT_ROOT / "CQCC-Akinci-Ozbay-DEV" / "final_model.pt"),
    "F0 expert": (F0MLP, "f0", PROJECT_ROOT / "F0-v25-AUC" / "final_model.pt"),
}
PROPOSED_ARTIFACTS = {
    "CNN–GRU branch": PROJECT_ROOT / "CNN-GRU" / "final_model.pt",
    "CQCC branch": PROJECT_ROOT / "CQCC-Akinci-Ozbay-DEV" / "final_model.pt",
    "F0 branch": PROJECT_ROOT / "F0-v25-AUC" / "final_model.pt",
    "Prediction-level fusion": PROJECT_ROOT / "Prediction-Fusion" / "CNN-GRU-CQCC-F0-F-Akinci-Temporal-EXP" / "fusion_model_akinci_temporal.npy",
    "Causal temporal correction": PROJECT_ROOT / "Prediction-Fusion" / "CNN-GRU-CQCC-F0-F-Akinci-Temporal-EXP" / "temporal_final_akinci.pt",
}
LIVE_HISTORY_LIMIT = 60
LIVE_MIN_RMS = 0.002
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
_LIVE_SESSION_LOCK = threading.Lock()
_LIVE_SESSION_ACTIVE = {}


def get_audio_path(value):
    if isinstance(value, dict):
        return value.get("path")
    return value


def decode_audio_value(audio_value):
    """Decode either Gradio's numpy audio tuple or a filesystem audio path."""
    if isinstance(audio_value, tuple) and len(audio_value) == 2:
        sample_rate, audio = audio_value
        audio = np.asarray(audio)
    else:
        path = get_audio_path(audio_value)
        if not path:
            raise ValueError("Upload an audio file or record with the microphone.")
        path = Path(path)
        if path.suffix.lower() not in {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au"}:
            raise ValueError("Use a supported audio file: WAV, FLAC, OGG, AIFF, or AU.")
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Audio files must be 10 MB or smaller.")
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("The audio is empty or invalid.")
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        raise ValueError("The audio contains invalid samples.")
    if int(sample_rate) != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def _load_single_model(model_name):
    cls, feature, path = SINGLE_MODELS[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"Required {model_name} checkpoint is missing: {path}")
    model = cls(F0_DIM) if feature == "f0" else (cls(40) if feature == "cqcc" else cls(dropout=0.0))
    try:
        weights = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        model.load_state_dict(weights, strict=True)
    except Exception as exc:
        raise ValueError(f"Built-in {model_name} checkpoint is incompatible: {exc}") from exc
    model.to(DEVICE).eval()
    return model, feature, str(path)


def _load_proposed_model():
    """Load the exact finalized five-artifact temporal fusion architecture."""
    missing = [f"{name}: {path}" for name, path in PROPOSED_ARTIFACTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing proposed-model artifact(s): " + " | ".join(missing))
    import improvedCQCC_config as config
    from improvedCQCC_latency_test import _get_latency_module

    # The published artifacts are intentionally beside this app, rather than
    # under config.MODELS_DIR/models. Compile the frozen temporal runner with
    # this project root as its artifact root without changing global config.
    original_models_dir = config.MODELS_DIR
    config.MODELS_DIR = str(PROJECT_ROOT)
    try:
        module = _get_latency_module("improvedCQCC_latency_v28")
        # The fusion metadata was generated in E:\Thesis and records that
        # absolute OOF path. The copied D:\New\Thesis OOF has the same SHA-256;
        # permit this relocation only when its recorded content hash still
        # matches exactly. Any actual cache modification remains an error.
        from improvedCQCC_cqcc_f0_fusion import _get_fusion
        _, fusion_modules = _get_fusion()
        v28 = fusion_modules["improvedCQCC_v28_temporal_evidence_fusion"]
        strict_load_artifact = v28.load_artifact

        def load_relocated_artifact():
            try:
                return strict_load_artifact()
            except RuntimeError as exc:
                if "OOF cache changed since v28 fitting" not in str(exc):
                    raise
                artifact = np.load(v28.META_PATH, allow_pickle=True).item()
                current_hash = v28.sha256(v28.OOF_PATH)
                if artifact.get("source_sha256") != current_hash:
                    raise
                return artifact

        v28.load_artifact = load_relocated_artifact
        runner = module.Runner()
    finally:
        config.MODELS_DIR = original_models_dir
    return runner, "proposed", runner.paths


def load_active_model(model_name):
    """Load built-in models only; user checkpoint uploads are intentionally unsupported."""
    with _MODEL_CACHE_LOCK:
        if model_name not in _MODEL_CACHE:
            if model_name == PROPOSED_MODEL:
                _MODEL_CACHE[model_name] = _load_proposed_model()
            elif model_name in SINGLE_MODELS:
                model, feature, path = _load_single_model(model_name)
                _MODEL_CACHE[model_name] = (model, feature, {"checkpoint": path})
            else:
                raise ValueError(f"Unknown model selection: {model_name}")
        return _MODEL_CACHE[model_name]


def model_readiness(model_name):
    """Return a user-facing readiness state after artifact validation/loading."""
    try:
        load_active_model(model_name)
        return True, ""
    except Exception as exc:
        return False, f"Model not ready — {exc}"


DEFAULT_MODEL_READY, DEFAULT_MODEL_STATUS = model_readiness(PROPOSED_MODEL)


def change_active_model(model_name):
    ready, status = model_readiness(model_name)
    return (
        model_name,
        model_name,
        gr.Textbox(value=status, visible=not ready),
        gr.Audio(interactive=ready),
    )


def predict_window(window, model, feature, fusion_history=()):
    """Return one window score and processing latency (feature extraction included)."""
    started = time.perf_counter()
    if feature == "proposed":
        probability, label, feature_row, _ = model.process_window(window, previous_rows=fusion_history)
        return probability, (time.perf_counter() - started) * 1000, label, feature_row
    if feature == "logmel":
        x = torch.from_numpy(extract_logmel(window)[None, None]).float().to(DEVICE)
        with torch.inference_mode():
            logit = model(x)
    else:
        vector = extract_cqcc(window) if feature == "cqcc" else extract_f0(window)
        x = torch.from_numpy(vector[None]).float().to(DEVICE)
        with torch.inference_mode():
            logit = model(x)
    probability = float(torch.sigmoid(logit.reshape(-1)[0]).item())
    return probability, (time.perf_counter() - started) * 1000, ("AI" if probability >= 0.5 else "Real"), None


def normalize_for_model(audio):
    """Match the zero-mean, unit-variance, peak-normalized training waveform path."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Audio is empty.")
    audio = audio - float(np.mean(audio))
    audio = audio / (float(np.std(audio)) + 1e-8)
    return peak_normalize(audio)


def analyze(audio_value, model_name):
    try:
        audio = normalize_for_model(decode_audio_value(audio_value))
    except Exception as exc:
        raise gr.Error(f"Could not decode this audio file: {exc}") from exc

    model, feature, _ = load_active_model(model_name)
    windows = [audio[start:start + WIN_SAMPLES] for start in range(0, max(1, len(audio) - WIN_SAMPLES + 1), HOP_SAMPLES)]
    if len(audio) < WIN_SAMPLES:
        windows = [np.pad(audio, (0, WIN_SAMPLES - len(audio)))]
    elif len(audio) > WIN_SAMPLES and (len(audio) - WIN_SAMPLES) % HOP_SAMPLES:
        tail = audio[-WIN_SAMPLES:]
        if not np.array_equal(tail, windows[-1]):
            windows.append(tail)

    rows = []
    probabilities = []
    fusion_history = []
    started = time.perf_counter()
    for index, window in enumerate(windows, start=1):
        p_ai, window_ms, label, feature_row = predict_window(window, model, feature, fusion_history)
        if feature_row is not None:
            fusion_history.append(np.asarray(feature_row, dtype=np.float32))
        probabilities.append(p_ai)
        start_seconds = (index - 1) * HOP_SAMPLES / SAMPLE_RATE
        rows.append([index, f"{start_seconds:.2f}–{start_seconds + WIN_SAMPLES / SAMPLE_RATE:.2f}s",
                     f"{p_ai * 100:.1f}%", label, f"{window_ms:.1f} ms"])
    total_ms = (time.perf_counter() - started) * 1000
    final_ai = float(np.mean(probabilities))
    final_label = "AI" if final_ai >= 0.5 else "Real"
    final_percent = f"{final_ai * 100:.1f}%"
    return rows, final_percent, final_label, f"{total_ms:.1f} ms", f"{final_percent} ({final_label}, {len(probabilities)} windows)", "Offline clip analyzed."


def _empty_live_state():
    return {
        "buffer": np.empty(0, dtype=np.float32),
        "window_count": 0,
        "rows": [],
        "probabilities": [],
        "fusion_history": [],
        "session_token": None,
    }


def _empty_result_state():
    return {"probability": "", "label": "", "running": "", "latency": ""}


def _set_session_active(token, active):
    with _LIVE_SESSION_LOCK:
        _LIVE_SESSION_ACTIVE[token] = active


def _is_session_active(token):
    with _LIVE_SESSION_LOCK:
        return bool(token) and _LIVE_SESSION_ACTIVE.get(token, False)


def _live_chunk_to_mono(audio_value):
    """Convert Gradio's (sample_rate, samples) microphone chunk to 16-kHz mono."""
    return decode_audio_value(audio_value)


def reset_live_session():
    token = uuid.uuid4().hex
    _set_session_active(token, True)
    state = _empty_live_state()
    state["session_token"] = token
    return state, _empty_result_state(), [], "Recording started. Waiting for the first full window…", "", "", "", ""


def stop_live_session(state):
    """Mark the session inactive without touching the displayed prediction values."""
    state = state or _empty_live_state()
    _set_session_active(state.get("session_token"), False)
    return state


def analyze_live_chunk(audio_value, model_name, state, result_state):
    """Process 1-second windows every 0.5 s from streaming microphone chunks."""
    state = state or _empty_live_state()
    result_state = result_state or _empty_result_state()
    token = state.get("session_token")
    if not _is_session_active(token):
        # A queued callback from a stopped/older recording must not overwrite the UI.
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
    try:
        chunk = _live_chunk_to_mono(audio_value)
        # Loading happens outside the displayed per-window latency measurement.
        model, feature, _ = load_active_model(model_name)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return (state["rows"], result_state["probability"], result_state["label"],
                result_state["running"], result_state["latency"], f"Live input error: {exc}",
                state, result_state)

    buffer = np.concatenate((np.asarray(state["buffer"], dtype=np.float32), chunk))
    rows = list(state["rows"])
    count = int(state["window_count"])
    probabilities = list(state.get("probabilities", []))
    fusion_history = [np.asarray(row, dtype=np.float32) for row in state.get("fusion_history", [])]
    newest_probability = newest_label = newest_latency = ""
    skipped_windows = 0

    while len(buffer) >= WIN_SAMPLES:
        raw_window = buffer[:WIN_SAMPLES]
        count += 1
        start_seconds = (count - 1) * HOP_SAMPLES / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(raw_window * raw_window)))
        if rms < LIVE_MIN_RMS:
            skipped_windows += 1
            row = [count, f"{start_seconds:.2f}–{start_seconds + WIN_SAMPLES / SAMPLE_RATE:.2f}s",
                   "—", "No speech", f"input RMS {rms:.4f}"]
            rows.append(row)
            rows = rows[-LIVE_HISTORY_LIMIT:]
            buffer = buffer[HOP_SAMPLES:]
            continue
        probability, latency_ms, label, feature_row = predict_window(
            normalize_for_model(raw_window), model, feature, fusion_history
        )
        if feature_row is not None:
            fusion_history.append(np.asarray(feature_row, dtype=np.float32))
        probabilities.append(probability)
        row = [count, f"{start_seconds:.2f}–{start_seconds + WIN_SAMPLES / SAMPLE_RATE:.2f}s",
               f"{probability * 100:.1f}%", label, f"{latency_ms:.1f} ms"]
        rows.append(row)
        rows = rows[-LIVE_HISTORY_LIMIT:]
        newest_probability = f"{probability * 100:.1f}%"
        newest_label = label
        newest_latency = f"{latency_ms:.1f} ms"
        buffer = buffer[HOP_SAMPLES:]

    new_state = {
        "buffer": buffer, "window_count": count, "rows": rows,
        "probabilities": probabilities, "fusion_history": fusion_history, "session_token": token,
    }
    running_probability = ""
    if probabilities:
        mean_probability = float(np.mean(probabilities))
        running_probability = f"{mean_probability * 100:.1f}% ({'AI' if mean_probability >= 0.5 else 'Real'}, {len(probabilities)} speech windows)"
    if not _is_session_active(token):
        # Stop may have occurred while this callback was extracting/inferencing.
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
    if not newest_probability:
        remaining = max(0.0, (WIN_SAMPLES - len(buffer)) / SAMPLE_RATE)
        detail = "No speech detected in the latest window." if skipped_windows else f"Listening… {remaining:.2f}s until the first window."
        return (rows, result_state["probability"], result_state["label"],
                result_state["running"], result_state["latency"], detail,
                new_state, result_state)
    new_result_state = {
        "probability": newest_probability,
        "label": newest_label,
        "running": running_probability,
        "latency": newest_latency,
    }
    return (rows, newest_probability, newest_label, running_probability, newest_latency,
            "Live analysis running.", new_state, new_result_state)


with gr.Blocks(
    title="Speech Authenticity",
    css="""
    #audio-input { position: relative; }
    #audio-input[data-uploaded-filename]:not([data-uploaded-filename=""])::after {
        content: attr(data-uploaded-filename);
        position: absolute;
        top: 50%; left: 50%; transform: translate(-50%, -50%);
        max-width: 82%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        padding: 0.45rem 0.8rem; border-radius: 0.45rem;
        background: rgba(255, 255, 255, 0.92); color: #374151;
        font-weight: 600; pointer-events: none; z-index: 2;
    }
    """,
) as demo:
    gr.Markdown("# Speech Authenticity\nReal-time Filipino speech deepfake detection")
    active_model = gr.Textbox(label="Active Model", value=PROPOSED_MODEL, interactive=False)
    model_status = gr.Textbox(
        label="Model status", value=DEFAULT_MODEL_STATUS,
        interactive=False, visible=not DEFAULT_MODEL_READY,
    )
    selected_model = gr.State(PROPOSED_MODEL)
    with gr.Accordion("Want to use another model?", open=False):
        model_select = gr.Dropdown(MODEL_CHOICES, value=PROPOSED_MODEL, label="Change Model")

    audio_input = gr.Audio(
        label="Audio clip", sources=["upload", "microphone"], type="filepath", format="wav",
        streaming=True, interactive=DEFAULT_MODEL_READY, elem_id="audio-input",
    )
    upload_marker = gr.State("")
    with gr.Row():
        final_percent = gr.Textbox(label="Latest window / clip AI probability", interactive=False)
        final_label = gr.Textbox(label="Latest window / clip label", interactive=False)
        total_latency = gr.Textbox(label="Latest window / clip latency", interactive=False)
    live_running = gr.Textbox(label="Running live-clip AI probability", interactive=False)
    live_status = gr.Textbox(
        label="Analysis status",
        value="Ready to record or upload audio." if DEFAULT_MODEL_READY else "The active model must finish loading before analysis can start.",
        interactive=False,
    )
    live_state = gr.State(_empty_live_state())
    result_state = gr.State(_empty_result_state())
    per_window = gr.Dataframe(headers=["Window", "Audio time", "AI probability", "Label", "Latency"], label="Per-window results", interactive=False)

    model_select.change(change_active_model, [model_select], [selected_model, active_model, model_status, audio_input])
    file_uploaded = audio_input.upload(
        lambda _: "uploaded", [audio_input], [upload_marker],
        js="""(audioValue) => {
            const audio = document.getElementById('audio-input');
            if (!audio) return [audioValue];
            const filename = audioValue?.orig_name || audioValue?.name ||
                (audioValue?.path ? audioValue.path.split(/[\\/]/).pop() : '');
            audio.dataset.uploadedFilename = filename;
            const syncUploadBox = () => {
                const hasFilename = Boolean(audio.dataset.uploadedFilename);
                const uploadPrompt = new Set(['Drop Audio Here', '- or -', 'Click to Upload']);
                audio.querySelectorAll('*').forEach((element) => {
                    if (uploadPrompt.has(element.textContent.trim())) {
                        element.dataset.audioUploadPrompt = 'true';
                        element.style.display = hasFilename ? 'none' : '';
                    }
                });
            };
            if (!audio._uploadBoxObserver) {
                audio._uploadBoxObserver = new MutationObserver(syncUploadBox);
                audio._uploadBoxObserver.observe(audio, { childList: true, subtree: true, characterData: true });
            }
            syncUploadBox();
            return [audioValue];
        }""",
    )
    file_uploaded.then(
        analyze, [audio_input, selected_model],
        [per_window, final_percent, final_label, total_latency, live_running, live_status],
    )
    audio_input.clear(
        lambda: "", outputs=[upload_marker],
        js="""() => {
            const audio = document.getElementById('audio-input');
            if (!audio) return [];
            audio.dataset.uploadedFilename = '';
            audio.querySelectorAll('[data-audio-upload-prompt="true"]').forEach((element) => {
                element.style.display = '';
                delete element.dataset.audioUploadPrompt;
            });
            return [];
        }""",
    )
    audio_input.start_recording(reset_live_session, outputs=[live_state, result_state, per_window, live_status, final_percent, final_label, live_running, total_latency])
    audio_input.stop_recording(stop_live_session, [live_state], [live_state])
    audio_input.stream(
        analyze_live_chunk,
        [audio_input, selected_model, live_state, result_state],
        [per_window, final_percent, final_label, live_running, total_latency, live_status, live_state, result_state],
        stream_every=0.5,
    )


if __name__ == "__main__":
    demo.launch(max_file_size="10mb")
