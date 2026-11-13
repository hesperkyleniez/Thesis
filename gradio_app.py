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
MODELS = {
    "CNN GRU": (CNNGRUOnly, "logmel"),
    "CQCC expert": (CQCCMLP, "cqcc"),
    "F0 expert": (F0MLP, "f0"),
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


def load_checkpoint(model_name, checkpoint_file, checkpoint_path):
    path = get_audio_path(checkpoint_file) if checkpoint_file else (checkpoint_path or "").strip()
    if not path:
        raise ValueError("Select or upload a .pt checkpoint for the chosen model.")
    if not path or Path(path).suffix.lower() != ".pt":
        raise ValueError("The model file must use the .pt extension.")
    if os.path.getsize(path) > MAX_BYTES:
        raise ValueError("The checkpoint must be 10 MB or smaller.")
    cls, feature = MODELS[model_name]
    model = cls(F0_DIM) if feature == "f0" else (cls(40) if feature == "cqcc" else cls(dropout=0.0))
    try:
        weights = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        model.load_state_dict(weights, strict=True)
    except Exception as exc:
        raise ValueError(f"This checkpoint does not match {model_name}: {exc}") from exc
    model.to(DEVICE).eval()
    return model, feature


def load_cached_checkpoint(model_name, checkpoint_file, checkpoint_path):
    """Load once per selected checkpoint; live callbacks must not reload weights."""
    path = get_audio_path(checkpoint_file) if checkpoint_file else (checkpoint_path or "").strip()
    if not path:
        raise ValueError("Select or upload a .pt checkpoint before starting the live microphone.")
    path = str(Path(path).resolve())
    if not os.path.isfile(path):
        raise ValueError(f"Checkpoint not found: {path}")
    key = (model_name, path, os.path.getmtime(path))
    with _MODEL_CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE.clear()
            _MODEL_CACHE[key] = load_checkpoint(model_name, None, path)
        return _MODEL_CACHE[key]


def predict_window(window, model, feature):
    """Return one window score and processing latency (feature extraction included)."""
    started = time.perf_counter()
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
    return probability, (time.perf_counter() - started) * 1000


def normalize_for_model(audio):
    """Match the zero-mean, unit-variance, peak-normalized training waveform path."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Audio is empty.")
    audio = audio - float(np.mean(audio))
    audio = audio / (float(np.std(audio)) + 1e-8)
    return peak_normalize(audio)


def analyze(audio_value, model_name, checkpoint_file, checkpoint_path):
    try:
        audio = normalize_for_model(decode_audio_value(audio_value))
    except Exception as exc:
        raise gr.Error(f"Could not decode this audio file: {exc}") from exc

    model, feature = load_checkpoint(model_name, checkpoint_file, checkpoint_path)
    windows = [audio[start:start + WIN_SAMPLES] for start in range(0, max(1, len(audio) - WIN_SAMPLES + 1), HOP_SAMPLES)]
    if len(audio) < WIN_SAMPLES:
        windows = [np.pad(audio, (0, WIN_SAMPLES - len(audio)))]
    elif len(audio) > WIN_SAMPLES and (len(audio) - WIN_SAMPLES) % HOP_SAMPLES:
        tail = audio[-WIN_SAMPLES:]
        if not np.array_equal(tail, windows[-1]):
            windows.append(tail)

    rows = []
    probabilities = []
    started = time.perf_counter()
    for index, window in enumerate(windows, start=1):
        p_ai, window_ms = predict_window(window, model, feature)
        probabilities.append(p_ai)
        label = "AI" if p_ai >= 0.5 else "Real"
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
    if not isinstance(audio_value, tuple):
        raise ValueError("No microphone samples received yet.")
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


def warm_live_model(model_name, checkpoint_file, checkpoint_path):
    try:
        _, feature = load_cached_checkpoint(model_name, checkpoint_file, checkpoint_path)
        return f"Model ready ({feature}). Click Record below; first result appears after 1.0 second."
    except ValueError as exc:
        return f"Model not ready: {exc}"


def analyze_live_chunk(audio_value, model_name, checkpoint_file, checkpoint_path, state, result_state):
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
        model, feature = load_cached_checkpoint(model_name, checkpoint_file, checkpoint_path)
    except ValueError as exc:
        return (state["rows"], result_state["probability"], result_state["label"],
                result_state["running"], result_state["latency"], f"Live input error: {exc}",
                state, result_state)

    buffer = np.concatenate((np.asarray(state["buffer"], dtype=np.float32), chunk))
    rows = list(state["rows"])
    count = int(state["window_count"])
    probabilities = list(state.get("probabilities", []))
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
        probability, latency_ms = predict_window(normalize_for_model(raw_window), model, feature)
        probabilities.append(probability)
        label = "AI" if probability >= 0.5 else "Real"
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
        "probabilities": probabilities, "session_token": token,
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


with gr.Blocks(title="Speech Authenticity") as demo:
    gr.Markdown("# Speech Authenticity\nWindow level AI speech detection")
    with gr.Row():
        model_select = gr.Dropdown(list(MODELS), value="CNN GRU", label="Model")
        checkpoint = gr.File(label="Model checkpoint (.pt)", file_types=[".pt"], type="filepath")
    checkpoint_path = gr.Textbox(label="Or enter a local .pt file path", placeholder=r"D:\New\Thesis\models\checkpoint.pt")
    audio_input = gr.Audio(label="Audio clip", sources=["upload", "microphone"], type="numpy", format="wav", streaming=True)
    run_button = gr.Button("Analyze uploaded/recorded clip", variant="primary")
    with gr.Row():
        final_percent = gr.Textbox(label="Latest window / clip AI probability", interactive=False)
        final_label = gr.Textbox(label="Latest window / clip label", interactive=False)
        total_latency = gr.Textbox(label="Latest window / clip latency", interactive=False)
    live_running = gr.Textbox(label="Running live-clip AI probability", interactive=False)
    live_status = gr.Textbox(label="Status", value="Upload a file then Analyze, or click the microphone icon for live detection.", interactive=False)
    live_state = gr.State(_empty_live_state())
    result_state = gr.State(_empty_result_state())
    per_window = gr.Dataframe(headers=["Window", "Audio time", "AI probability", "Label", "Latency"], label="Per-window results", interactive=False)

    run_button.click(analyze, [audio_input, model_select, checkpoint, checkpoint_path], [per_window, final_percent, final_label, total_latency, live_running, live_status])
    audio_input.start_recording(reset_live_session, outputs=[live_state, result_state, per_window, live_status, final_percent, final_label, live_running, total_latency])
    audio_input.stop_recording(stop_live_session, [live_state], [live_state])
    audio_input.stream(
        analyze_live_chunk,
        [audio_input, model_select, checkpoint, checkpoint_path, live_state, result_state],
        [per_window, final_percent, final_label, live_running, total_latency, live_status, live_state, result_state],
        stream_every=0.5,
    )


if __name__ == "__main__":
    demo.launch(max_file_size="10mb")
