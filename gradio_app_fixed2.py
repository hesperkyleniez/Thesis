"""Small Gradio interface for windowed Filipino speech deepfake detection."""
from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from gradio.routes import mount_gradio_app
import uvicorn

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
# Tab/system audio is often quieter than a close mic; avoid skipping real speech.
LIVE_MIN_PEAK = 0.004
MAX_LIVE_WINDOWS_PER_CHUNK = 2
DECISION_THRESHOLD = 0.50
# Keep temporal context short so mixed Real/AI speech can flip quickly.
LIVE_FUSION_HISTORY = 6
LIVE_RECENT_WINDOW_COUNT = 10

def decision_label(probability):
    """Map AI probability to AI/Real using a fixed 0.50 threshold."""
    return "AI" if float(probability) >= DECISION_THRESHOLD else "Real"

def _trim_fusion_history(fusion_history):
    if len(fusion_history) > LIVE_FUSION_HISTORY:
        return fusion_history[-LIVE_FUSION_HISTORY:]
    return fusion_history

def _mixed_running_summary(probabilities):
    """Summarize recent windows so mixed Real/AI content stays visible."""
    if not probabilities:
        return ""
    recent = list(probabilities[-LIVE_RECENT_WINDOW_COUNT:])
    mean_probability = float(np.mean(recent))
    ai_n = sum(1 for p in recent if float(p) >= DECISION_THRESHOLD)
    real_n = len(recent) - ai_n
    latest = decision_label(recent[-1])
    return (
        f"latest {latest} ({recent[-1] * 100:.1f}%) | "
        f"last {len(recent)}: {ai_n} AI / {real_n} Real "
        f"(mean {mean_probability * 100:.1f}%)"
    )
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
_LIVE_SESSION_LOCK = threading.Lock()
_LIVE_SESSION_ACTIVE = {}
_CAPTURE_LIVE_LOCK = threading.Lock()
_CAPTURE_LIVE_STATE = {}
_CAPTURE_UI_LOCK = threading.Lock()
_CAPTURE_UI = {
    "session_id": None,
    "snapshot": None,
    "final": None,
    "seq": 0,
    "last_seen_seq": 0,
}

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
        if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".au"}:
            raise ValueError("Use a supported audio file: WAV, MP3, FLAC, OGG, AIFF, or AU.")
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Audio files must be 10 MB or smaller.")
        if path.suffix.lower() == ".mp3":
            audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
        else:
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

    original_models_dir = config.MODELS_DIR
    config.MODELS_DIR = str(PROJECT_ROOT)
    try:
        module = _get_latency_module("improvedCQCC_latency_v28")
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
        # GUI uses a fixed 0.50 decision threshold instead of the tuned artifact cutoff.
        runner.threshold = DECISION_THRESHOLD
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
        probability, _, feature_row, _ = model.process_window(window, previous_rows=fusion_history)
        return (
            float(probability),
            (time.perf_counter() - started) * 1000,
            decision_label(probability),
            feature_row,
        )
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
    return probability, (time.perf_counter() - started) * 1000, decision_label(probability), None

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
    final_label = decision_label(final_ai)
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

def _window_has_speech(raw_window):
    """Gate on peak and RMS so quiet Chrome-tab speech is still scored."""
    peak = float(np.max(np.abs(raw_window)))
    if peak < LIVE_MIN_PEAK:
        return False
    rms = float(np.sqrt(np.mean(raw_window * raw_window)))
    return rms >= LIVE_MIN_RMS or peak >= (LIVE_MIN_PEAK * 2)

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

def finalize_live_session(audio_value, model_name, state, result_state):
    """Re-analyze the completed microphone recording with the offline clip pipeline."""
    state = stop_live_session(state)
    result_state = result_state or _empty_result_state()

    if not audio_value:
        return (
            state["rows"], result_state["probability"], result_state["label"],
            result_state["running"], result_state["latency"],
            "Recording stopped, but no audio was received for final analysis.",
            state, result_state,
        )

    try:
        rows, probability, label, latency, running_summary, _ = analyze(
            audio_value, model_name
        )
    except Exception as exc:
        return (
            state["rows"], result_state["probability"], result_state["label"],
            result_state["running"], result_state["latency"],
            f"Final recording analysis failed: {exc}", state, result_state,
        )

    final_result_state = {
        "probability": probability,
        "label": label,
        "running": running_summary,
        "latency": latency,
    }
    return (
        rows, probability, label, running_summary, latency,
        "Recording complete. Final result is based on the entire recording.",
        state, final_result_state,
    )

def analyze_live_chunk(audio_value, model_name, state, result_state):
    """Process 1-second windows every 0.5 s from streaming microphone chunks."""
    state = state or _empty_live_state()
    result_state = result_state or _empty_result_state()
    token = state.get("session_token")
    if not _is_session_active(token):
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
    try:
        chunk = _live_chunk_to_mono(audio_value)
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
    windows_done = 0

    while len(buffer) >= WIN_SAMPLES and windows_done < MAX_LIVE_WINDOWS_PER_CHUNK:
        raw_window = buffer[:WIN_SAMPLES]
        count += 1
        windows_done += 1
        start_seconds = (count - 1) * HOP_SAMPLES / SAMPLE_RATE
        peak = float(np.max(np.abs(raw_window)))
        rms = float(np.sqrt(np.mean(raw_window * raw_window)))
        if not _window_has_speech(raw_window):
            skipped_windows += 1
            row = [count, f"{start_seconds:.2f}–{start_seconds + WIN_SAMPLES / SAMPLE_RATE:.2f}s",
                   "—", "No speech", f"peak {peak:.4f} / RMS {rms:.4f}"]
            rows.append(row)
            rows = rows[-LIVE_HISTORY_LIMIT:]
            buffer = buffer[HOP_SAMPLES:]
            continue
        prev_label = decision_label(probabilities[-1]) if probabilities else ""
        probability, latency_ms, label, feature_row = predict_window(
            normalize_for_model(raw_window), model, feature, fusion_history
        )
        # Soft-reset temporal memory when Real↔AI flips so mixed clips react faster.
        if prev_label and label != prev_label:
            fusion_history = fusion_history[-2:] if len(fusion_history) > 2 else list(fusion_history)
        if feature_row is not None:
            fusion_history.append(np.asarray(feature_row, dtype=np.float32))
            fusion_history = _trim_fusion_history(fusion_history)
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
    running_probability = _mixed_running_summary(probabilities)
    if not _is_session_active(token):
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
    if not newest_probability:
        remaining = max(0.0, (WIN_SAMPLES - len(buffer)) / SAMPLE_RATE)
        detail = "No speech detected in the latest window." if skipped_windows else f"Listening… {remaining:.2f}s until the first window."
        return (rows, result_state["probability"], result_state["label"],
                running_probability or result_state["running"], result_state["latency"], detail,
                new_state, result_state)
    new_result_state = {
        "probability": newest_probability,
        "label": newest_label,
        "running": running_probability,
        "latency": newest_latency,
    }
    return (rows, newest_probability, newest_label, running_probability, newest_latency,
            "Live analysis running.", new_state, new_result_state)

def _process_capture_live_chunk(chunk, model_name, session_id):
    """Same windowed live pipeline used by mic streaming, for Chrome/screen-share audio."""
    with _CAPTURE_LIVE_LOCK:
        state = _CAPTURE_LIVE_STATE.get(session_id)
        if state is None:
            state = _empty_live_state()
            state["session_token"] = session_id
            _CAPTURE_LIVE_STATE[session_id] = state

    model, feature, _ = load_active_model(model_name)
    buffer = np.concatenate((np.asarray(state["buffer"], dtype=np.float32), chunk))
    rows = list(state["rows"])
    count = int(state["window_count"])
    probabilities = list(state.get("probabilities", []))
    fusion_history = [np.asarray(row, dtype=np.float32) for row in state.get("fusion_history", [])]
    last_probability = state.get("last_probability") or ""
    last_label = state.get("last_label") or ""
    last_latency = state.get("last_latency") or ""
    newest_probability = newest_label = newest_latency = ""
    skipped_windows = 0
    windows_done = 0
    status = "Capturing laptop / Chrome audio…"

    # Cap work per request so the UI stays responsive instead of batching seconds of lag.
    while len(buffer) >= WIN_SAMPLES and windows_done < MAX_LIVE_WINDOWS_PER_CHUNK:
        raw_window = buffer[:WIN_SAMPLES]
        count += 1
        windows_done += 1
        start_seconds = (count - 1) * HOP_SAMPLES / SAMPLE_RATE
        peak = float(np.max(np.abs(raw_window)))
        rms = float(np.sqrt(np.mean(raw_window * raw_window)))
        if not _window_has_speech(raw_window):
            skipped_windows += 1
            row = [count, f"{start_seconds:.2f}–{start_seconds + WIN_SAMPLES / SAMPLE_RATE:.2f}s",
                   "—", "No speech", f"peak {peak:.4f} / RMS {rms:.4f}"]
            rows.append(row)
            rows = rows[-LIVE_HISTORY_LIMIT:]
            buffer = buffer[HOP_SAMPLES:]
            continue
        prev_label = decision_label(probabilities[-1]) if probabilities else (last_label or "")
        probability, latency_ms, label, feature_row = predict_window(
            normalize_for_model(raw_window), model, feature, fusion_history
        )
        # Soft-reset temporal memory when Real↔AI flips so mixed clips react faster.
        if prev_label in {"AI", "Real"} and label != prev_label:
            fusion_history = fusion_history[-2:] if len(fusion_history) > 2 else list(fusion_history)
        if feature_row is not None:
            fusion_history.append(np.asarray(feature_row, dtype=np.float32))
            fusion_history = _trim_fusion_history(fusion_history)
        probabilities.append(probability)
        row = [count, f"{start_seconds:.2f}–{start_seconds + WIN_SAMPLES / SAMPLE_RATE:.2f}s",
               f"{probability * 100:.1f}%", label, f"{latency_ms:.1f} ms"]
        rows.append(row)
        rows = rows[-LIVE_HISTORY_LIMIT:]
        newest_probability = f"{probability * 100:.1f}%"
        newest_label = label
        newest_latency = f"{latency_ms:.1f} ms"
        last_probability, last_label, last_latency = newest_probability, newest_label, newest_latency
        buffer = buffer[HOP_SAMPLES:]

    running_probability = _mixed_running_summary(probabilities)
    display_probability = newest_probability or last_probability
    display_label = newest_label or last_label
    display_latency = newest_latency or last_latency
    if newest_probability:
        status = f"Live latest: {display_label} ({display_probability}). {running_probability}"
    elif skipped_windows and not display_probability:
        status = "No speech detected in the latest window."
    elif skipped_windows:
        status = f"Quiet moment — last score {display_label} ({display_probability})."
    else:
        remaining = max(0.0, (WIN_SAMPLES - len(buffer)) / SAMPLE_RATE)
        status = f"Listening to shared audio… {remaining:.2f}s until the first window."

    new_state = {
        "buffer": buffer,
        "window_count": count,
        "rows": rows,
        "probabilities": probabilities,
        "fusion_history": fusion_history,
        "session_token": session_id,
        "last_probability": last_probability,
        "last_label": last_label,
        "last_latency": last_latency,
    }
    with _CAPTURE_LIVE_LOCK:
        _CAPTURE_LIVE_STATE[session_id] = new_state

    payload = {
        "rows": rows,
        "probability": display_probability,
        "label": display_label,
        "latency": display_latency,
        "running_summary": running_probability,
        "status": status,
    }
    _publish_capture_snapshot(payload)
    return payload

def _reset_capture_live_session(session_id):
    with _CAPTURE_LIVE_LOCK:
        state = _empty_live_state()
        state["session_token"] = session_id
        _CAPTURE_LIVE_STATE[session_id] = state
    with _CAPTURE_UI_LOCK:
        _CAPTURE_UI["session_id"] = session_id
        _CAPTURE_UI["snapshot"] = {
            "rows": [],
            "probability": "",
            "label": "",
            "latency": "",
            "running_summary": "",
            "status": "Listening to shared audio… waiting for the first window.",
        }
        _CAPTURE_UI["final"] = None
        _CAPTURE_UI["seq"] += 1

def _clear_capture_live_session(session_id):
    with _CAPTURE_LIVE_LOCK:
        _CAPTURE_LIVE_STATE.pop(session_id, None)

def _publish_capture_snapshot(payload):
    with _CAPTURE_UI_LOCK:
        _CAPTURE_UI["snapshot"] = dict(payload)
        _CAPTURE_UI["seq"] += 1

def _publish_capture_final(payload):
    with _CAPTURE_UI_LOCK:
        _CAPTURE_UI["final"] = dict(payload)
        _CAPTURE_UI["session_id"] = None
        _CAPTURE_UI["seq"] += 1

def poll_chrome_capture_ui():
    """Mirror Chrome-tab capture results into the shared Gradio outputs/table."""
    with _CAPTURE_UI_LOCK:
        if _CAPTURE_UI["final"] is not None:
            data = _CAPTURE_UI["final"]
            _CAPTURE_UI["final"] = None
            _CAPTURE_UI["last_seen_seq"] = _CAPTURE_UI["seq"]
            return (
                data.get("rows") or [],
                data.get("probability") or "",
                data.get("label") or "",
                data.get("running_summary") or "",
                data.get("latency") or "",
                data.get("status") or "Analysis complete.",
            )
        if not _CAPTURE_UI["session_id"] or _CAPTURE_UI["snapshot"] is None:
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        if _CAPTURE_UI["seq"] == _CAPTURE_UI["last_seen_seq"]:
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        _CAPTURE_UI["last_seen_seq"] = _CAPTURE_UI["seq"]
        data = _CAPTURE_UI["snapshot"]
        return (
            data.get("rows") or [],
            data.get("probability") or "",
            data.get("label") or "",
            data.get("running_summary") or "",
            data.get("latency") or "",
            data.get("status") or "Live screen/tab audio analysis running.",
        )

def switch_input_mode(mode):
    if mode == "Upload":
        return (
            gr.update(visible=True, sources=["upload"], streaming=False),
            gr.update(visible=False),
            "Upload WAV, MP3, FLAC, OGG, AIFF, or AU. Results appear in the table below.",
        )
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        "Click Start Capture once, share a Chrome tab with Share tab audio. Live results use the same table below.",
    )

CHROME_CAPTURE_SCRIPT = r"""
(() => {
  if (window.__chromeTabCaptureInstalled) return;
  window.__chromeTabCaptureInstalled = true;

  let displayStream = null;
  let audioContext = null;
  let processor = null;
  let source = null;
  let silentGain = null;
  let chunks = [];
  let pendingChunks = [];
  let capturing = false;
  let starting = false;
  let stopping = false;
  let liveBusy = false;
  let liveTimer = null;
  let sessionId = null;

  const get = (id) => document.getElementById(id);
  const setStatus = (message) => {
    const node = get('chrome-capture-status');
    if (node) node.textContent = message;
  };
  const setButtons = (isCapturing) => {
    const start = get('chrome-start');
    const stop = get('chrome-stop');
    if (start) start.disabled = isCapturing || starting;
    if (stop) stop.disabled = !isCapturing;
  };

  function cleanup() {
    if (liveTimer) {
      clearInterval(liveTimer);
      liveTimer = null;
    }
    if (processor) {
      processor.disconnect();
      processor.onaudioprocess = null;
    }
    if (source) source.disconnect();
    if (silentGain) silentGain.disconnect();
    if (displayStream) displayStream.getTracks().forEach((track) => track.stop());
    if (audioContext && audioContext.state !== 'closed') audioContext.close();
    displayStream = null;
    audioContext = null;
    processor = null;
    source = null;
    silentGain = null;
    capturing = false;
    starting = false;
    pendingChunks = [];
    setButtons(false);
  }

  function makeWav(parts, sampleRate) {
    const length = parts.reduce((total, part) => total + part.length, 0);
    const samples = new Float32Array(length);
    let offset = 0;
    for (const part of parts) {
      samples.set(part, offset);
      offset += part.length;
    }
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeText = (position, value) => {
      for (let i = 0; i < value.length; i++) view.setUint8(position + i, value.charCodeAt(i));
    };
    writeText(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeText(8, 'WAVE');
    writeText(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeText(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let position = 44;
    for (let i = 0; i < samples.length; i++, position += 2) {
      const value = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(position, value < 0 ? value * 32768 : value * 32767, true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
  }

  function currentModelName() {
    const active = document.querySelector('#active-model textarea, #active-model input');
    if (active && active.value) return active.value;
    return 'CNN–GRU + CQCC + F0 (Prediction-Level Fusion + Causal Temporal Correction)';
  }

  function takePendingAudio(maxSeconds) {
    const rate = audioContext ? audioContext.sampleRate : 48000;
    const maxSamples = Math.floor(maxSeconds * rate);
    let total = 0;
    const parts = [];
    while (pendingChunks.length && total < maxSamples) {
      const part = pendingChunks.shift();
      parts.push(part);
      total += part.length;
    }
    // If we fell far behind, drop excess backlog so scores stay near-real-time.
    const backlogLimit = rate * 1.5;
    let backlog = pendingChunks.reduce((sum, part) => sum + part.length, 0);
    while (backlog > backlogLimit && pendingChunks.length) {
      backlog -= pendingChunks.shift().length;
    }
    return parts;
  }

  async function flushLiveChunk() {
    if (!capturing || liveBusy || !pendingChunks.length || !audioContext) return;
    const parts = takePendingAudio(0.6);
    if (!parts.length) return;
    const audioBlob = makeWav(parts, audioContext.sampleRate);
    if (audioBlob.size < 100) return;
    liveBusy = true;
    try {
      const form = new FormData();
      form.append('file', audioBlob, 'chrome_live_chunk.wav');
      form.append('model_name', currentModelName());
      form.append('session_id', sessionId);
      const response = await fetch('/capture-audio-live', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || 'Live analysis failed.');
      if (data.label && data.probability) {
        setStatus('Live: ' + data.label + ' (' + data.probability + '). ' + (data.running_summary || ''));
      } else if (data.status) {
        setStatus(data.status);
      }
    } catch (error) {
      setStatus('Live analysis error: ' + error.message);
    } finally {
      liveBusy = false;
    }
  }

  async function finishCapture() {
    if (!capturing || stopping) return;
    stopping = true;
    setStatus('Analyzing full capture…');
    const sampleRate = audioContext ? audioContext.sampleRate : 48000;
    const audioBlob = makeWav(chunks, sampleRate);
    const modelName = currentModelName();
    const finishedSession = sessionId;
    cleanup();

    if (audioBlob.size < 1000) {
      setStatus('No audio captured. Choose a Chrome tab and enable Share tab audio, then try again.');
      stopping = false;
      return;
    }
    if (audioBlob.size > 10 * 1024 * 1024) {
      setStatus('Recording is too long. Keep it under approximately 5 minutes and try again.');
      stopping = false;
      return;
    }

    try {
      const form = new FormData();
      form.append('file', audioBlob, 'chrome_tab_capture.wav');
      form.append('model_name', modelName);
      if (finishedSession) form.append('session_id', finishedSession);
      const response = await fetch('/capture-audio', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || 'Analysis failed.');
      setStatus('Analysis complete. See the shared results table below.');
    } catch (error) {
      setStatus('Analysis failed: ' + error.message);
    } finally {
      stopping = false;
    }
  }

  async function startCapture() {
    if (capturing || starting) {
      setStatus('Already capturing. Use Stop & Analyze when you are done. Do not open Share again.');
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      setStatus('Screen/tab audio capture is not supported in this browser. Use Google Chrome.');
      return;
    }

    starting = true;
    setButtons(false);
    try {
      setStatus('Pick a Chrome Tab (not Entire Screen), enable Share tab audio, then click Share.');
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        },
        preferCurrentTab: false,
        systemAudio: 'include'
      });
      displayStream = stream;
      const audioTracks = displayStream.getAudioTracks();
      if (!audioTracks.length) {
        cleanup();
        setStatus('No audio received. Start again, pick a Chrome tab, and check Share tab audio.');
        return;
      }

      sessionId = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now());
      await fetch('/capture-audio-live-reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      }).catch(() => {});

      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      await audioContext.resume();
      source = audioContext.createMediaStreamSource(new MediaStream(audioTracks));
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      chunks = [];
      pendingChunks = [];
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        const copy = new Float32Array(input.length);
        copy.set(input);
        chunks.push(copy);
        pendingChunks.push(copy);
      };
      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(audioContext.destination);

      capturing = true;
      starting = false;
      stopping = false;
      setButtons(true);
      setStatus('Streaming shared audio. Results update live below. Click Stop & Analyze when done.');
      liveTimer = setInterval(flushLiveChunk, 300);
      displayStream.getVideoTracks().forEach((track) => {
        track.addEventListener('ended', () => { if (capturing) finishCapture(); }, { once: true });
      });
      audioTracks.forEach((track) => {
        track.addEventListener('ended', () => { if (capturing) finishCapture(); }, { once: true });
      });
    } catch (error) {
      cleanup();
      setStatus(
        error.name === 'NotAllowedError'
          ? 'Share cancelled. Click Start Capture again when ready, pick one Chrome tab with Share tab audio, then Share once.'
          : 'Could not start capture: ' + error.message
      );
    }
  }

  window.__startChromeCapture = (event) => {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    startCapture();
  };
  window.__stopChromeCapture = (event) => {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    finishCapture();
  };
})();
"""

TABLE_AUTO_FOLLOW_SCRIPT = r"""
(() => {
  if (window.__tableAutoFollowInstalled) return;
  window.__tableAutoFollowInstalled = true;

  const ROOT_ID = 'per-window-results';
  const NEAR_BOTTOM_PX = 56;
  let followBottom = true;
  let boundEl = null;

  function findScrollEl(root) {
    if (!root) return null;
    const nodes = [root, ...root.querySelectorAll('div, table, section')];
    let best = null;
    for (const el of nodes) {
      const style = window.getComputedStyle(el);
      const overflowY = style.overflowY;
      if ((overflowY === 'auto' || overflowY === 'scroll' || overflowY === 'overlay') &&
          el.scrollHeight > el.clientHeight + 2) {
        if (!best || el.scrollHeight > best.scrollHeight) best = el;
      }
    }
    if (best) return best;
    root.classList.add('per-window-scroll');
    return root;
  }

  function isNearBottom(el) {
    return (el.scrollHeight - el.scrollTop - el.clientHeight) <= NEAR_BOTTOM_PX;
  }

  function bindScroll(el) {
    if (!el || el === boundEl) return;
    if (boundEl) boundEl.removeEventListener('scroll', onUserScroll);
    boundEl = el;
    el.addEventListener('scroll', onUserScroll, { passive: true });
  }

  function onUserScroll(event) {
    followBottom = isNearBottom(event.currentTarget);
  }

  function stickIfFollowing() {
    const root = document.getElementById(ROOT_ID);
    if (!root) return;
    const el = findScrollEl(root);
    if (!el) return;
    bindScroll(el);
    if (followBottom) el.scrollTop = el.scrollHeight;
  }

  function watch() {
    const root = document.getElementById(ROOT_ID);
    if (!root || root._autoFollowWatching) return;
    root._autoFollowWatching = true;
    const observer = new MutationObserver(() => {
      requestAnimationFrame(stickIfFollowing);
    });
    observer.observe(root, { childList: true, subtree: true, characterData: true });
    stickIfFollowing();
  }

  const boot = () => watch();
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
  setInterval(watch, 1000);
})();
"""

APP_CSS = """
#audio-input[data-uploaded-filename]:not([data-uploaded-filename=""])::after {
    content: attr(data-uploaded-filename);
    position: absolute;
    top: 50%; left: 50%; transform: translate(-50%, -50%);
    max-width: 82%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    padding: 0.45rem 0.8rem; border-radius: 0.45rem;
    background: rgba(255, 255, 255, 0.92); color: #374151;
    font-weight: 600; pointer-events: none; z-index: 2;
}
#input-mode {
    margin-bottom: 0.35rem;
}
#chrome-capture-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px;
    margin: 0 0 8px 0;
    background: var(--block-background-fill);
}
#per-window-results {
    max-height: 360px;
    overflow-y: auto;
}
"""

CHROME_PANEL_HTML = """
<div id="chrome-capture" style="display:flex;flex-direction:column;gap:12px;">
  <p style="margin:0;font-size:14px;opacity:0.9;">
    Share a <strong>Chrome Tab</strong> and enable <strong>Share tab audio</strong>.
    Live AI/Real scores use the same results table below.
  </p>
  <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
    <button id="chrome-start" type="button" onclick="event.preventDefault();event.stopPropagation();window.__startChromeCapture&amp;&amp;window.__startChromeCapture(event)" style="padding:9px 14px;border:0;border-radius:8px;background:var(--button-primary-background-fill);color:var(--button-primary-text-color);font-weight:600;cursor:pointer;">Start Capture</button>
    <button id="chrome-stop" type="button" disabled onclick="event.preventDefault();event.stopPropagation();window.__stopChromeCapture&amp;&amp;window.__stopChromeCapture(event)" style="padding:9px 14px;border:1px solid var(--border-color-primary);border-radius:8px;background:var(--background-fill-secondary);color:var(--body-text-color);cursor:pointer;">Stop &amp; Analyze</button>
  </div>
  <div id="chrome-capture-status" role="status" style="font-size:14px;">Ready. Click Start Capture, share one Chrome tab with audio, then wait for the picker to close.</div>
</div>
"""

with gr.Blocks(title="Speech Authenticity") as demo:
    gr.Markdown("# Speech Authenticity\nReal-time Filipino speech deepfake detection")

    active_model = gr.Textbox(label="Active Model", value=PROPOSED_MODEL, interactive=False, elem_id="active-model")
    model_status = gr.Textbox(
        label="Model status", value=DEFAULT_MODEL_STATUS,
        interactive=False, visible=not DEFAULT_MODEL_READY,
    )
    selected_model = gr.State(PROPOSED_MODEL)
    with gr.Accordion("Want to use another model?", open=False):
        model_select = gr.Dropdown(MODEL_CHOICES, value=PROPOSED_MODEL, label="Change Model")

    input_mode = gr.Radio(
        ["Upload", "Stream"],
        value="Upload",
        label="Input source",
        elem_id="input-mode",
    )
    audio_input = gr.Audio(
        label="Audio clip (WAV, MP3, FLAC, OGG, AIFF, AU)",
        sources=["upload"],
        type="filepath",
        format="wav",
        streaming=False,
        interactive=DEFAULT_MODEL_READY,
        elem_id="audio-input",
    )
    chrome_panel = gr.HTML(CHROME_PANEL_HTML, visible=False, elem_id="chrome-capture-panel")
    upload_marker = gr.State("")

    with gr.Row():
        final_percent = gr.Textbox(label="Latest window / clip AI probability", interactive=False)
        final_label = gr.Textbox(label="Latest window / clip label", interactive=False)
        total_latency = gr.Textbox(label="Latest window / clip latency", interactive=False)
    live_running = gr.Textbox(label="Running live-clip AI probability", interactive=False)
    live_status = gr.Textbox(
        label="Analysis status",
        value="Choose Upload or Stream." if DEFAULT_MODEL_READY else "The active model must finish loading before analysis can start.",
        interactive=False,
    )
    live_state = gr.State(_empty_live_state())
    result_state = gr.State(_empty_result_state())
    per_window = gr.Dataframe(
        headers=["Window", "Audio time", "AI probability", "Label", "Latency"],
        label="Per-window results",
        interactive=False,
        elem_id="per-window-results",
    )

    model_select.change(change_active_model, [model_select], [selected_model, active_model, model_status, audio_input])
    input_mode.change(switch_input_mode, [input_mode], [audio_input, chrome_panel, live_status])
    capture_timer = gr.Timer(0.25)
    capture_timer.tick(
        poll_chrome_capture_ui,
        outputs=[per_window, final_percent, final_label, live_running, total_latency, live_status],
    )

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
api_app = FastAPI()

@api_app.post("/capture-audio-live-reset")
async def reset_chrome_live(payload: dict):
    session_id = str((payload or {}).get("session_id") or "").strip()
    if not session_id:
        return JSONResponse(status_code=400, content={"detail": "session_id is required."})
    _reset_capture_live_session(session_id)
    return {"ok": True}

@api_app.post("/capture-audio-live")
async def analyze_chrome_live(
    file: UploadFile = File(...),
    model_name: str = Form(PROPOSED_MODEL),
    session_id: str = Form(...),
):
    if not session_id.strip():
        return JSONResponse(status_code=400, content={"detail": "session_id is required."})
    if model_name not in MODEL_CHOICES:
        return JSONResponse(status_code=400, content={"detail": f"Unknown model: {model_name}"})
    payload = await file.read()
    if not payload:
        return JSONResponse(status_code=400, content={"detail": "Empty audio chunk."})
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(payload)
            temp_path = temp_file.name
        chunk = decode_audio_value(temp_path)
        return _process_capture_live_chunk(chunk, model_name, session_id.strip())
    except Exception as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@api_app.post("/capture-audio")
async def analyze_chrome_capture(
    file: UploadFile = File(...),
    model_name: str = Form(PROPOSED_MODEL),
    session_id: str = Form(""),
):
    if file.content_type not in {"audio/wav", "audio/x-wav", "application/octet-stream", None}:
        # Some browsers omit or loosen content_type for Blob uploads.
        pass
    if model_name not in MODEL_CHOICES:
        return JSONResponse(status_code=400, content={"detail": f"Unknown model: {model_name}"})
    payload = await file.read()
    if not payload or len(payload) > MAX_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Captured audio must be non-empty and 10 MB or smaller."})
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(payload)
            temp_path = temp_file.name
        rows, probability, label, latency, running_summary, _ = analyze(temp_path, model_name)
        if session_id:
            _clear_capture_live_session(session_id)
        payload = {
            "rows": rows,
            "probability": probability,
            "label": label,
            "latency": latency,
            "running_summary": running_summary,
            "status": "Analysis complete. Prediction is AI or Real for the shared Chrome audio.",
        }
        _publish_capture_final(payload)
        return payload
    except Exception as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    mounted_app = mount_gradio_app(
        api_app,
        demo,
        path="/",
        max_file_size="10mb",
        css=APP_CSS,
        js=CHROME_CAPTURE_SCRIPT + "\n" + TABLE_AUTO_FOLLOW_SCRIPT,
        head=f"<script>{CHROME_CAPTURE_SCRIPT}</script><script>{TABLE_AUTO_FOLLOW_SCRIPT}</script>",
    )
    uvicorn.run(mounted_app, host="127.0.0.1", port=7860)
