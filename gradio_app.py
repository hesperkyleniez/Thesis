"""Filipino speech detector. Run: python gradio_app.py; install once with --install.

Place this file beside code/ (or code_improved/) and models/. No launcher is needed.
"""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid

APP_DIR = Path(__file__).resolve().parent
RUNTIME_PACKAGES = [
    "gradio==6.32.0", "numpy==2.4.6", "scipy==1.17.1", "librosa==0.11.0",
    "soundfile==0.14.0", "scikit-learn==1.9.0", "numba==0.67.0", "tqdm==4.70.0",
    "fastapi==0.165.0", "uvicorn==0.54.0", "python-multipart==0.0.34",
]


def install_runtime():
    """Install into the Python environment used to run this command."""
    if sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        raise SystemExit("Please use 64-bit Python 3.11, then run this file with --install again.")
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine != "arm64":
        raise SystemExit("On M-series Macs use native Apple Silicon Python 3.11, without Rosetta.")
    if sys.platform == "darwin" and platform.mac_ver()[0] and int(platform.mac_ver()[0].split(".")[0]) < 14:
        raise SystemExit("These pinned audio libraries require macOS 14 or newer.")
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check", "--use-feature=truststore"]
    os.environ.pop("SSLKEYLOGFILE", None)
    print("Installing the app libraries into this Python environment. Internet is needed.", flush=True)
    if sys.platform == "win32":
        subprocess.run(pip + ["install", "torch==2.5.1+cpu", "--index-url",
                              "https://download.pytorch.org/whl/cpu"], check=True)
    else:
        subprocess.run(pip + ["install", "torch==2.5.1"], check=True)
    subprocess.run(pip + ["install"] + RUNTIME_PACKAGES, check=True)
    print("\nInstallation complete. Run this same Python file without --install.", flush=True)


OPTIONS = argparse.Namespace(code_dir=None, models_dir=None, port=7860, check=False)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--install", action="store_true", help="Install the Python libraries once, then exit")
    parser.add_argument("--check", action="store_true", help="Load the saved model and test generated audio, then exit")
    parser.add_argument("--code-dir", help="Optional path to the folder containing improvedCQCC_config.py")
    parser.add_argument("--models-dir", help="Optional path to the saved models folder")
    parser.add_argument("--port", type=int, default=7860)
    OPTIONS = parser.parse_args()
    if OPTIONS.install:
        try:
            install_runtime()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"Installation failed: {exc}. Read the message above and retry --install.")
        raise SystemExit(0)

# All paths are relative to this file, not the terminal's working directory.
code_override = OPTIONS.code_dir or os.environ.get("THESIS_CODE_DIR")
code_candidates = [APP_DIR / "code", APP_DIR / "code_improved", APP_DIR]
SOURCE_DIR = Path(code_override).expanduser().resolve() if code_override else next(
    (p for p in code_candidates if (p / "improvedCQCC_config.py").is_file()), APP_DIR / "code")
models_override = OPTIONS.models_dir or os.environ.get("THESIS_MODELS_DIR")
model_candidates = [APP_DIR / "models", APP_DIR.parent / "models", APP_DIR]
PROJECT_ROOT = Path(models_override).expanduser().resolve() if models_override else next(
    (p for p in model_candidates if (p / "CNN-GRU" / "final_model.pt").is_file()), APP_DIR / "models")
if not (SOURCE_DIR / "improvedCQCC_config.py").is_file():
    raise SystemExit(f"The code folder is missing: {SOURCE_DIR}\nPlace code/ beside this Python file, or use --code-dir PATH.")
os.environ["THESIS_BASE_DIR"] = str(PROJECT_ROOT.parent if PROJECT_ROOT.name == "models" else APP_DIR)
os.environ["THESIS_CODE_DIR"] = str(SOURCE_DIR)
os.environ["THESIS_MODELS_DIR"] = str(PROJECT_ROOT)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.pop("SSLKEYLOGFILE", None)
sys.path.insert(0, str(SOURCE_DIR))
missing = [name for name in ("gradio", "librosa", "numpy", "scipy", "soundfile", "torch", "sklearn",
                             "numba", "tqdm", "fastapi", "uvicorn", "python_multipart")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing libraries: " + ", ".join(missing) + "\nRun this Python file once with --install, then run it normally.")

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from gradio.routes import mount_gradio_app
import uvicorn

from improvedCQCC_config import HOP_SAMPLES, SAMPLE_RATE, WIN_SAMPLES
from improvedCQCC_augmentation import peak_normalize
from improvedCQCC_features import (extract_improved_cqcc as extract_cqcc, extract_f0, extract_logmel,
    paper_cqcc_frames, pool_by_centers, extract_f0_frames, _summarize_f0)
from improvedCQCC_train import CNNGRUOnly
from improvedCQCC_cqcc_train import CQCCMLP
from improvedCQCC_f0_train import F0MLP, F0_DIM

MAX_BYTES = 10 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import improvedCQCC_config as thesis_config
# PROJECT_ROOT was resolved before importing the thesis configuration.
_INFER_LOCK = threading.RLock()
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
WAITING_LABEL = "Low audio level"
LIVE_MIN_PEAK = 0.004
MAX_LIVE_WINDOWS_PER_CHUNK = 4
LIVE_RECENT_WINDOW_COUNT = 10
# Live scoring uses a short sliding mean so mixed Real/AI can flip quickly.
LIVE_SEGMENT_LIMIT = 4
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
_LIVE_SESSION_LOCK = threading.Lock()
_LIVE_SESSION_ACTIVE = {}
_CAPTURE_LIVE_LOCK = threading.RLock()
_CAPTURE_LIVE_STATE = {}
_CAPTURE_UI_LOCK = threading.Lock()
_CAPTURE_UI = {}


def _canonical_label(label):
    if not label or label == WAITING_LABEL:
        return ""
    return "AI" if str(label).startswith("AI") else "Real"


def _mixed_running_summary(probabilities, threshold):
    if not probabilities:
        return ""
    recent = list(probabilities[-LIVE_RECENT_WINDOW_COUNT:])
    mean_probability = float(np.mean(recent))
    ai_n = sum(1 for p in recent if float(p) >= threshold)
    real_n = len(recent) - ai_n
    latest = "AI" if recent[-1] >= threshold else "Real"
    return (
        f"latest {latest} ({recent[-1] * 100:.1f}%) | "
        f"last {len(recent)}: {ai_n} AI / {real_n} Real "
        f"(mean {mean_probability * 100:.1f}%)"
    )


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
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        midpoint = (info.max + 1) / 2 if info.min == 0 else 0
        scale = (info.max + 1) / 2 if info.min == 0 else max(abs(info.min), info.max)
        audio = (audio.astype(np.float32) - midpoint) / scale
    else:
        audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("The audio is empty or invalid.")
    if int(sample_rate) <= 0:
        raise ValueError("Invalid audio sample rate.")
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
    if feature != "logmel":
        stats = np.load(path.parent / "final_norm_stats.npy", allow_pickle=True).item()
        mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
        dim = 40 if feature == "cqcc" else F0_DIM
        if (mean.shape != (dim,) or std.shape != (dim,) or np.any(std <= 0)
                or not np.isfinite(mean).all() or not np.isfinite(std).all()):
            raise ValueError("Invalid expert normalization.")
        model.feature_mean = mean
        model.feature_std = std
    return model, feature, str(path)

class ProposedRunner:
    """Saved thesis experts and fusion, with incremental causal convolutions."""
    def __init__(self):
        missing = [str(p) for p in PROPOSED_ARTIFACTS.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing artifacts: " + " | ".join(missing))
        from improvedCQCC_cqcc_f0_fusion import _get_fusion
        _, modules = _get_fusion()
        self.api = modules["improvedCQCC_v28_temporal_evidence_fusion"]
        self.features_api = modules["improvedCQCC_cqcc_f0_fusion_v22_window_level"]
        self.artifact = np.load(PROPOSED_ARTIFACTS["Prediction-level fusion"], allow_pickle=True).item()
        if self.artifact.get("features") != self.features_api.WINDOW_FEATURE_NAMES:
            raise ValueError("Incompatible fusion feature order.")
        self.cnn = _load_single_model("CNN–GRU baseline")[0]
        self.cqcc = _load_single_model("CQCC expert")[0]
        self.f0 = _load_single_model("F0 expert")[0]
        self.net = self.api.TemporalResidual().to(DEVICE).eval()
        state = torch.load(PROPOSED_ARTIFACTS["Causal temporal correction"], map_location=DEVICE, weights_only=True)
        if state.get("version") != self.artifact.get("version") or state.get("final_epochs") != self.artifact.get("final_epochs"):
            raise ValueError("Fusion and temporal checkpoints do not match.")
        self.net.load_state_dict(state["state_dict"], strict=True)
        self.threshold = float(self.artifact["threshold"])
        self.paths = {k: str(v) for k, v in PROPOSED_ARTIFACTS.items()}

    def feature_row(self, lm, cq, f0):
        def score(model, value):
            x = torch.as_tensor(value, dtype=torch.float32, device=DEVICE)
            return float(torch.sigmoid(model(x)).reshape(-1)[0].item())
        with torch.inference_mode():
            scores = (score(self.cnn, lm[None, None]),
                      score(self.cqcc, ((cq-self.cqcc.feature_mean)/self.cqcc.feature_std)[None]),
                      score(self.f0, ((f0-self.f0.feature_mean)/self.f0.feature_std)[None]))
        record = dict(cnn_probs=np.array([scores[0]]), cqcc_probs=np.array([scores[1]]),
                      f0_probs=np.array([scores[2]]))
        return self.features_api.window_features(record, self.artifact["calibrators"])[0].astype(np.float32)

    def append_row(self, row, previous_rows):
        # Two k=3 convolutions with dilations 1 and 2 have a seven-window receptive field.
        prior = np.asarray(previous_rows, np.float32).reshape(-1, 10)
        context = np.vstack([prior[-6:, :9], row])
        mean, scale = self.artifact["scaler_mean"], self.artifact["scaler_scale"]
        x = torch.as_tensor((context-mean)/scale, dtype=torch.float32, device=DEVICE).T[None]
        with torch.inference_mode():
            h = torch.nn.functional.gelu(self.net.first(torch.nn.functional.pad(x, (2, 0))))
            h = torch.nn.functional.gelu(self.net.second(torch.nn.functional.pad(h, (4, 0))))
            correction = float(self.net.out(h)[0, 0, -1].item())
        stored = np.r_[row, correction].astype(np.float32)
        rows = np.vstack([prior, stored])
        ref = float(((rows[:, :9].mean(axis=0)-mean)/scale) @ self.artifact["reference_coef"]
                    + self.artifact["reference_intercept"])
        bound = self.api.CORRECTION_MAX
        delta = bound * np.tanh(float(rows[:, 9].mean()) / bound)
        probability = float(1/(1+np.exp(-np.clip(ref+delta, -60, 60))))
        return probability, "AI-Generated" if probability >= self.threshold else "Real", stored

    def process_window(self, audio, previous_rows=()):
        row = self.feature_row(extract_logmel(audio), extract_cqcc(audio), extract_f0(audio))
        if isinstance(previous_rows, dict):
            p, label, row = self.append_incremental(row, previous_rows)
        else:
            p, label, row = self.append_row(row, previous_rows)
        return p, label, row, {}

    def append_incremental(self, row, accumulator):
        """Causal score from six prior features plus a reference/residual mean.

        Offline clips keep a full-segment mean. Live capture sets accumulator['live_limit']
        so the mean uses only a short recent window and can follow mixed Real/AI speech.
        """
        row = np.asarray(row, np.float32)
        tail = np.asarray(accumulator.get("tail", np.empty((0, 9), np.float32)), np.float32)
        context = np.vstack((tail, row))
        mean, scale = self.artifact["scaler_mean"], self.artifact["scaler_scale"]
        x = torch.as_tensor((context-mean)/scale, dtype=torch.float32, device=DEVICE).T[None]
        with torch.inference_mode():
            h = torch.nn.functional.gelu(self.net.first(torch.nn.functional.pad(x, (2, 0))))
            h = torch.nn.functional.gelu(self.net.second(torch.nn.functional.pad(h, (4, 0))))
            correction = float(self.net.out(h)[0, 0, -1].item())
        stored = np.r_[row, correction].astype(np.float32)
        live_limit = accumulator.get("live_limit")
        if live_limit:
            recent = list(accumulator.get("recent_rows", []))
            recent.append(stored)
            recent = recent[-int(live_limit):]
            stacked = np.asarray(recent, dtype=np.float64)
            count = len(recent)
            total = stacked[:, :9].sum(axis=0)
            delta_sum = float(stacked[:, 9].sum())
            accumulator.update(
                count=count, sum=total, delta_sum=delta_sum,
                tail=context[-6:].copy(), recent_rows=recent, live_limit=int(live_limit),
            )
        else:
            count = int(accumulator.get("count", 0)) + 1
            total = np.asarray(accumulator.get("sum", np.zeros(9)), np.float64) + row
            delta_sum = float(accumulator.get("delta_sum", 0.0)) + correction
            accumulator.update(count=count, sum=total, delta_sum=delta_sum, tail=context[-6:].copy())
        ref = float(((total/count-mean)/scale) @ self.artifact["reference_coef"]
                    + self.artifact["reference_intercept"])
        bound = self.api.CORRECTION_MAX
        delta = bound * np.tanh((delta_sum/count)/bound)
        probability = float(1/(1+np.exp(-np.clip(ref+delta, -60, 60))))
        return probability, "AI-Generated" if probability >= self.threshold else "Real", stored

def _load_proposed_model():
    runner = ProposedRunner()
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
    started = time.perf_counter()
    with _INFER_LOCK:
        if feature == "proposed":
            p, label, row, _ = model.process_window(window, fusion_history)
            return p, (time.perf_counter()-started)*1000, label, row
        if feature == "logmel":
            vector = extract_logmel(window)[None, None]
        else:
            vector = extract_cqcc(window) if feature == "cqcc" else extract_f0(window)
            vector = ((vector-model.feature_mean)/model.feature_std)[None]
        with torch.inference_mode():
            p = float(torch.sigmoid(model(torch.as_tensor(vector, dtype=torch.float32, device=DEVICE))).item())
    return p, (time.perf_counter()-started)*1000, "AI" if p >= 0.5 else "Real", None


def normalize_for_model(audio):
    """Match the zero-mean, unit-variance, peak-normalized training waveform path."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Audio is empty.")
    audio = audio - float(np.mean(audio))
    audio = audio / (float(np.std(audio)) + 1e-8)
    return peak_normalize(audio)

def has_audible_audio(audio):
    """An amplitude gate before normalization; this is not a learned speech detector."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not audio.size:
        return False
    centered = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(centered)))
    rms = float(np.sqrt(np.mean(centered*centered)))
    return peak >= LIVE_MIN_PEAK and (rms >= LIVE_MIN_RMS or peak >= 2*LIVE_MIN_PEAK)


def analyze(audio_value, model_name, ignore_silence=False):
    started = time.perf_counter()
    raw_audio = decode_audio_value(audio_value)
    if ignore_silence and not any(has_audible_audio(raw_audio[start:start+HOP_SAMPLES])
                                  for start in range(0, len(raw_audio), HOP_SAMPLES)):
        return [], "", WAITING_LABEL, "", "", "No audible speech in this capture; no classification was made."
    audio = normalize_for_model(raw_audio)
    model, feature, _ = load_active_model(model_name)
    starts = np.arange(0, len(audio), HOP_SAMPLES)
    padded = np.pad(audio, (0, int(starts[-1])+WIN_SAMPLES-len(audio)))
    # Match the paper's whole-recording frame extraction and padded tail windows.
    cqcc = pool_by_centers(audio, paper_cqcc_frames) if feature in ("proposed", "cqcc") else None
    f0 = None
    if feature in ("proposed", "f0"):
        values, voiced, centers = extract_f0_frames(padded)
        f0 = np.stack([_summarize_f0(values[(centers>=s)&(centers<s+WIN_SAMPLES)],
                                     voiced[(centers>=s)&(centers<s+WIN_SAMPLES)]) for s in starts])
    rows, probabilities, history = [], [], {}
    for i, start in enumerate(starts):
        if ignore_silence:
            end = min(start + WIN_SAMPLES, len(raw_audio))
            latest = raw_audio[max(start, end - HOP_SAMPLES):end]
            if not has_audible_audio(latest):
                history = {}  # Do not treat windows separated by a pause as adjacent.
                continue
        window = padded[start:start+WIN_SAMPLES]
        tick = time.perf_counter()
        with _INFER_LOCK:
            if feature == "proposed":
                row = model.feature_row(extract_logmel(window), cqcc[i], f0[i])
                p, label, row = model.append_incremental(row, history)
            elif feature in ("cqcc", "f0"):
                vector = cqcc[i] if feature == "cqcc" else f0[i]
                vector = (vector-model.feature_mean)/model.feature_std
                with torch.inference_mode():
                    p = float(torch.sigmoid(model(torch.as_tensor(vector[None], device=DEVICE))).item())
                label = "AI" if p>=0.5 else "Real"
            else:
                p, _, label, _ = predict_window(window, model, feature)
        probabilities.append(p)
        rows.append([i+1, f"{start/SAMPLE_RATE:.2f}–{min(start+WIN_SAMPLES,len(audio))/SAMPLE_RATE:.2f}s",
                     f"{p*100:.1f}%", label, f"{(time.perf_counter()-tick)*1000:.1f} ms"])
    if not probabilities:
        return [], "", WAITING_LABEL, "", "", "No audible speech in this capture; no classification was made."
    p = probabilities[-1] if feature == "proposed" else float(np.mean(probabilities))
    threshold = model.threshold if feature == "proposed" else .5
    label = "AI-Generated" if p>=threshold else "Real"
    percent = f"{p*100:.1f}%"
    status = ("Capture analyzed; quiet windows excluded. Temporal context restarts after pauses."
              if ignore_silence else "Full recording analyzed; final partial windows are zero-padded.")
    return rows, percent, label, f"{(time.perf_counter()-started)*1000:.1f} ms", f"{percent} ({label})", status


def _empty_live_state():
    return {
        "buffer": np.empty(0, dtype=np.float32),
        "window_count": 0,
        "rows": [],
        "score_total": 0.0,
        "score_count": 0,
        "fusion_state": {"live_limit": LIVE_SEGMENT_LIMIT},
        "recent_probs": [],
        "last_scored_label": "",
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
    with _LIVE_SESSION_LOCK:
        _LIVE_SESSION_ACTIVE.pop(state.get("session_token"), None)
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
        rows, probability, label, latency, running_summary, status = analyze(
            audio_value, model_name, ignore_silence=True
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
        status,
        state, final_result_state,
    )

def _advance_live_audio(chunk, model_name, state):
    """Shared mic/tab pipeline; quiet input neither gets a score nor enters fusion."""
    model, feature, _ = load_active_model(model_name)
    threshold = float(getattr(model, "threshold", 0.5)) if feature == "proposed" else 0.5
    buffer = np.concatenate((np.asarray(state["buffer"], dtype=np.float32), chunk))
    # Drop backlog so scores track the current audio instead of lagging behind.
    max_buffer = WIN_SAMPLES + HOP_SAMPLES * MAX_LIVE_WINDOWS_PER_CHUNK
    if len(buffer) > max_buffer:
        buffer = buffer[-max_buffer:]
    rows = list(state["rows"])
    count = int(state["window_count"])
    score_total = float(state.get("score_total", 0.0))
    score_count = int(state.get("score_count", 0))
    fusion_state = dict(state.get("fusion_state") or {})
    fusion_state.setdefault("live_limit", LIVE_SEGMENT_LIMIT)
    if "recent_rows" in fusion_state:
        fusion_state["recent_rows"] = [np.asarray(r, np.float32).copy() for r in fusion_state["recent_rows"]]
    if "tail" in fusion_state:
        fusion_state["tail"] = np.asarray(fusion_state["tail"], np.float32).copy()
    if "sum" in fusion_state:
        fusion_state["sum"] = np.asarray(fusion_state["sum"], np.float64).copy()
    recent_probs = list(state.get("recent_probs", []))
    last_scored_label = state.get("last_scored_label", "")
    latest_score = state.get("latest_score")
    previous = state.get("last_payload", {})
    newest_probability = previous.get("probability", "")
    newest_label = previous.get("label", "")
    newest_latency = previous.get("latency", "")
    waiting = state.get("waiting_for_audio", False)
    processed = False
    windows_done = 0

    while len(buffer) >= WIN_SAMPLES and windows_done < MAX_LIVE_WINDOWS_PER_CHUNK:
        raw_window = buffer[:WIN_SAMPLES]
        count += 1
        windows_done += 1
        start_seconds = (count - 1) * HOP_SAMPLES / SAMPLE_RATE
        buffer = buffer[HOP_SAMPLES:]
        processed = True
        if not has_audible_audio(raw_window[-HOP_SAMPLES:]):
            waiting = True
            newest_probability = newest_latency = ""
            newest_label = WAITING_LABEL
            score_total, score_count, latest_score = 0.0, 0, None
            fusion_state = {"live_limit": LIVE_SEGMENT_LIMIT}
            recent_probs = []
            last_scored_label = ""
            rows.append([count, f"{start_seconds:.2f}–{start_seconds + WIN_SAMPLES/SAMPLE_RATE:.2f}s",
                         "—", WAITING_LABEL, "—"])
            rows = rows[-LIVE_HISTORY_LIMIT:]
            continue
        waiting = False
        audio_window = normalize_for_model(raw_window)
        started = time.perf_counter()
        if feature == "proposed":
            # One feature extract; solo score detects flips, then short live fusion updates.
            with _INFER_LOCK:
                feat = model.feature_row(
                    extract_logmel(audio_window), extract_cqcc(audio_window), extract_f0(audio_window)
                )
                solo_p, solo_label, _ = model.append_incremental(
                    np.asarray(feat, np.float32).copy(), {"live_limit": LIVE_SEGMENT_LIMIT}
                )
                solo_canon = _canonical_label(solo_label)
                if last_scored_label and solo_canon and solo_canon != last_scored_label:
                    fusion_state = {"live_limit": LIVE_SEGMENT_LIMIT}
                    score_total, score_count = 0.0, 0
                probability, label, _ = model.append_incremental(
                    np.asarray(feat, np.float32).copy(), fusion_state
                )
            latency_ms = (time.perf_counter() - started) * 1000
        else:
            probability, latency_ms, label, _ = predict_window(
                audio_window, model, feature, fusion_state
            )
            solo_canon = _canonical_label(label)
            if last_scored_label and solo_canon and solo_canon != last_scored_label:
                fusion_state = {"live_limit": LIVE_SEGMENT_LIMIT}
                score_total, score_count = 0.0, 0
        canon = _canonical_label(label)
        score_total += probability
        score_count += 1
        latest_score = probability
        recent_probs.append(float(probability))
        recent_probs = recent_probs[-LIVE_RECENT_WINDOW_COUNT:]
        last_scored_label = canon or last_scored_label
        row = [count, f"{start_seconds:.2f}–{start_seconds + WIN_SAMPLES / SAMPLE_RATE:.2f}s",
               f"{probability * 100:.1f}%", label, f"{latency_ms:.1f} ms"]
        rows.append(row)
        rows = rows[-LIVE_HISTORY_LIMIT:]
        newest_probability = f"{probability * 100:.1f}%"
        newest_label = label
        newest_latency = f"{latency_ms:.1f} ms"

    new_state = {
        **state, "buffer": buffer, "window_count": count, "rows": rows,
        "score_total": score_total, "score_count": score_count, "fusion_state": fusion_state,
        "recent_probs": recent_probs, "last_scored_label": last_scored_label,
        "latest_score": latest_score,
        "model_name": model_name, "waiting_for_audio": waiting,
    }
    running_probability = _mixed_running_summary(recent_probs, threshold)
    if waiting:
        status = "Low audio level; no classification. Waiting for audible input…"
    elif newest_probability:
        status = f"Live latest: {newest_label} ({newest_probability}). {running_probability}"
    elif processed or score_count:
        status = "Live analysis running."
    else:
        remaining = max(0.0, (WIN_SAMPLES - len(buffer)) / SAMPLE_RATE)
        status = f"Listening… {remaining:.2f}s until the first window."
    payload = {
        "rows": rows, "probability": newest_probability, "label": newest_label,
        "latency": newest_latency, "running_summary": running_probability, "status": status,
    }
    new_state["last_payload"] = payload
    return new_state, payload


def analyze_live_chunk(audio_value, model_name, state, result_state):
    """Process microphone chunks without classifying pauses or amplifying near silence."""
    state = state or _empty_live_state()
    result_state = result_state or _empty_result_state()
    token = state.get("session_token")
    if state.get("model_name", model_name) != model_name:
        state = _empty_live_state()
        state["session_token"] = token
    if not _is_session_active(token):
        return (gr.skip(),)*8
    try:
        new_state, payload = _advance_live_audio(_live_chunk_to_mono(audio_value), model_name, state)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return (state["rows"], result_state["probability"], result_state["label"],
                result_state["running"], result_state["latency"], f"Live input error: {exc}",
                state, result_state)
    if not _is_session_active(token):
        return (gr.skip(),)*8
    new_result = {"probability": payload["probability"], "label": payload["label"],
                  "running": payload["running_summary"], "latency": payload["latency"]}
    return (payload["rows"], payload["probability"], payload["label"], payload["running_summary"],
            payload["latency"], payload["status"], new_state, new_result)


def _process_capture_live_chunk(chunk, model_name, session_id):
    """Same windowed live pipeline used by mic streaming, for Chrome/screen-share audio."""
    with _CAPTURE_LIVE_LOCK:
        state = _CAPTURE_LIVE_STATE.get(session_id)
        if state is None:
            raise ValueError("Capture session ended. Start a new recording.")

    if state.get("model_name", model_name) != model_name:
        raise ValueError("Model changed; start a new capture.")
    new_state, payload = _advance_live_audio(chunk, model_name, state)
    with _CAPTURE_LIVE_LOCK:
        _CAPTURE_LIVE_STATE[session_id] = new_state

    _publish_capture_snapshot(payload, session_id)
    return payload


def _reset_capture_live_session(session_id, owner):
    with _CAPTURE_LIVE_LOCK:
        old = _CAPTURE_UI.get(owner, {}).get("session_id")
        if old:
            _CAPTURE_LIVE_STATE.pop(old, None)
        _CAPTURE_LIVE_STATE[session_id] = _empty_live_state()
        _CAPTURE_UI[owner] = dict(session_id=session_id, snapshot=None, seq=0, seen=-1)


def _clear_capture_live_session(session_id):
    with _CAPTURE_LIVE_LOCK:
        _CAPTURE_LIVE_STATE.pop(session_id, None)


def _apply_capture_chunk(path, model_name, session_id, owner, sequence):
    """Run away from the web event loop and serialize updates to each session."""
    chunk = decode_audio_value(path)
    with _CAPTURE_LIVE_LOCK:
        ui = _CAPTURE_UI.get(owner)
        if not ui or ui["session_id"] != session_id:
            raise ValueError("Capture session ended.")
        state = _CAPTURE_LIVE_STATE.get(session_id)
        if state is None:
            raise ValueError("Capture session ended.")
        previous = state.get("last_sequence", -1)
        if sequence == previous:
            return state["last_response"]
        if sequence != previous + 1:
            raise ValueError("Audio chunk arrived out of order.")
        response = _process_capture_live_chunk(chunk, model_name, session_id)
        state = _CAPTURE_LIVE_STATE[session_id]
        state["last_sequence"] = sequence
        state["last_response"] = response
        return response

def _publish_capture_snapshot(payload, session_id):
    with _CAPTURE_LIVE_LOCK:
        for ui in _CAPTURE_UI.values():
            if ui["session_id"] == session_id:
                ui["snapshot"] = dict(payload)
                ui["seq"] += 1


def _publish_capture_final(payload, owner):
    with _CAPTURE_LIVE_LOCK:
        ui = _CAPTURE_UI.get(owner)
        if ui:
            ui["snapshot"] = dict(payload)
            ui["seq"] += 1


def poll_chrome_capture_ui(request: gr.Request):
    with _CAPTURE_LIVE_LOCK:
        ui = _CAPTURE_UI.get(request.session_hash)
        if not ui or ui["snapshot"] is None or ui["seen"] == ui["seq"]:
            return (gr.skip(),)*6
        ui["seen"] = ui["seq"]
        data = ui["snapshot"]
        return (data.get("rows", []), data.get("probability", ""), data.get("label", ""),
                data.get("running_summary", ""), data.get("latency", ""), data.get("status", ""))


def switch_input_mode(mode):
    if mode == "Upload":
        return (
            gr.update(visible=True, sources=["upload"], streaming=False),
            gr.update(visible=False),
            "Upload WAV, MP3, FLAC, OGG, AIFF, or AU. Results appear in the table below.",
        )
    if mode == "Microphone":
        return (
            gr.update(visible=True, sources=["microphone"], streaming=True),
            gr.update(visible=False),
            "Record with the microphone. Live results appear in the table below.",
        )
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        "Click Start Capture once, share a Chrome tab with Share tab audio. Live results use the same table below.",
    )

def reset_controls(model_name, mode, state, request: gr.Request):
    stop_live_session(state)
    owner = request.session_hash
    with _CAPTURE_LIVE_LOCK:
        ui = _CAPTURE_UI.pop(owner, None)
        if ui:
            _CAPTURE_LIVE_STATE.pop(ui["session_id"], None)
    ready, status = model_readiness(model_name)
    audio, panel, message = switch_input_mode(mode)
    return (model_name, model_name, gr.update(value=status, visible=not ready),
            gr.update(visible=mode!="Stream", sources=["microphone"] if mode=="Microphone" else ["upload"],
                      streaming=mode=="Microphone", interactive=ready, value=None),
            panel, message, _empty_live_state(), _empty_result_state(), [], "", "", "", "")

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
  let retryBlob = null;
  let chunkSequence = 0;
  let capturedSamples = 0;
  const maxCaptureSamples = Math.floor((10 * 1024 * 1024 - 44) / 2);
  let sessionId = null;
  function browserId() {
    const field = document.querySelector('#capture-owner textarea, #capture-owner input');
    return field ? field.value : '';
  }

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
    retryBlob = null;
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

  async function flushLiveChunk() {
    if (!capturing || stopping || liveBusy || (!pendingChunks.length && !retryBlob) || !audioContext) return;
    const audioBlob = retryBlob || makeWav(pendingChunks, audioContext.sampleRate);
    if (!retryBlob) pendingChunks = [];
    retryBlob = audioBlob;
    if (audioBlob.size < 100) return;
    liveBusy = true;
    const sendingSession = sessionId;
    try {
      const form = new FormData();
      form.append('file', audioBlob, 'chrome_live_chunk.wav');
      form.append('chunk_sequence', String(chunkSequence));
      form.append('browser_id', browserId());
      form.append('model_name', currentModelName());
      form.append('session_id', sessionId);
      const response = await fetch('/capture-audio-live', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || 'Live analysis failed.');
      if (sendingSession === sessionId && capturing) {
        retryBlob = null;
        chunkSequence += 1;
        if (data.status) setStatus(data.status);
      }
    } catch (error) {
      setStatus('Live analysis error: ' + error.message + '. Retrying buffered audio.');

    } finally {
      liveBusy = false;
    }
  }

  window.__cancelChromeCapture = () => {
    const old = sessionId;
    sessionId = null;
    cleanup();
    chunks = [];
    stopping = false;
    return old;
  };

  async function finishCapture() {
    if (!capturing || stopping) return;
    stopping = true;
    setStatus('Analyzing full capture…');
    while (liveBusy) await new Promise(resolve => setTimeout(resolve, 30));
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
      setStatus('Recording exceeds the 10 MB limit. Shorten the capture and try again.');
      stopping = false;
      return;
    }

    try {
      const form = new FormData();
      form.append('file', audioBlob, 'chrome_tab_capture.wav');
      form.append('browser_id', browserId());
      form.append('model_name', modelName);
      if (finishedSession) form.append('session_id', finishedSession);
      const response = await fetch('/capture-audio', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || 'Analysis failed.');
      setStatus(data.status || 'Analysis complete. See the shared results table below.');
    } catch (error) {
      setStatus('Analysis failed: ' + error.message);
    } finally {
      stopping = false;
    }
  }

  async function startCapture() {
    if (capturing || starting || stopping) {
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
      const resetResponse = await fetch('/capture-audio-live-reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, browser_id: browserId() })
      });
      if (!resetResponse.ok) throw new Error('Browser session is not ready. Reload the page.');

      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      await audioContext.resume();
      source = audioContext.createMediaStreamSource(new MediaStream(audioTracks));
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      chunks = [];
      pendingChunks = [];
      retryBlob = null;
      chunkSequence = 0;
      capturedSamples = 0;
      processor.onaudioprocess = (event) => {
        if (!capturing || stopping) return;
        const input = event.inputBuffer.getChannelData(0);
        const remaining = maxCaptureSamples - capturedSamples;
        if (remaining <= 0) return;
        const copy = new Float32Array(Math.min(input.length, remaining));
        copy.set(input.subarray(0, copy.length));
        chunks.push(copy);
        pendingChunks.push(copy);
        capturedSamples += copy.length;
        if (capturedSamples >= maxCaptureSamples) {
          setStatus('Capture size limit reached. Finishing analysis…');
          setTimeout(finishCapture, 0);
        }
      };
      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(audioContext.destination);

      capturing = true;
      starting = false;
      stopping = false;
      setButtons(true);
      setStatus('Streaming shared audio. Results update in the table below. Click Stop & Analyze when done.');
      liveTimer = setInterval(flushLiveChunk, 300);
      const table = document.getElementById('per-window-results');
      if (table) table.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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

APP_CSS = r"""#capture-owner { display: none !important; }

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

CHROME_PANEL_HTML = r"""
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

def browser_key(request: gr.Request):
    return request.session_hash


def cleanup_browser(request: gr.Request):
    with _CAPTURE_LIVE_LOCK:
        ui = _CAPTURE_UI.pop(request.session_hash, None)
        if ui:
            _CAPTURE_LIVE_STATE.pop(ui["session_id"], None)

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

with gr.Blocks(title="Speech Authenticity") as demo:
    capture_owner = gr.Textbox(value="", elem_id="capture-owner", interactive=False, container=False)
    demo.load(browser_key, outputs=capture_owner)
    demo.unload(cleanup_browser)
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
        ["Upload", "Microphone", "Stream"],
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
        final_percent = gr.Textbox(label="Updated recording AI score", interactive=False)
        final_label = gr.Textbox(label="Updated recording label", interactive=False)
        total_latency = gr.Textbox(label="Latest window / clip latency", interactive=False)
    live_running = gr.Textbox(label="Recording / current speech segment summary", interactive=False)
    live_status = gr.Textbox(
        label="Analysis status",
        value="Choose Upload, Microphone, or Stream." if DEFAULT_MODEL_READY else "The active model must finish loading before analysis can start.",
        interactive=False,
    )
    live_state = gr.State(_empty_live_state())
    result_state = gr.State(_empty_result_state())
    per_window = gr.Dataframe(
        headers=["Window", "Audio interval", "Updated recording score (proposed)", "Label", "Processing time"],
        label="Per-window results",
        interactive=False,
        elem_id="per-window-results",
    )

    for control in (model_select, input_mode):
        control.change(reset_controls, [model_select, input_mode, live_state],
            [selected_model, active_model, model_status, audio_input, chrome_panel, live_status,
             live_state, result_state, per_window, final_percent, final_label, live_running, total_latency],
            js="(...args) => { window.__cancelChromeCapture?.(); return args; }",
            concurrency_id="audio", concurrency_limit=1)
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
    audio_input.start_recording(reset_live_session, concurrency_id="audio", outputs=[live_state, result_state, per_window, live_status, final_percent, final_label, live_running, total_latency])
    audio_input.stop_recording(
        finalize_live_session,
        [audio_input, selected_model, live_state, result_state],
        [per_window, final_percent, final_label, live_running, total_latency,
         live_status, live_state, result_state], concurrency_id="audio", concurrency_limit=1,
    )
    audio_input.stream(
        analyze_live_chunk,
        [audio_input, selected_model, live_state, result_state],
        [per_window, final_percent, final_label, live_running, total_latency, live_status, live_state, result_state],
        stream_every=0.5, time_limit=600, concurrency_id="audio", concurrency_limit=1,
    )

api_app = FastAPI()

@api_app.post("/capture-audio-live-reset")
async def reset_chrome_live(payload: dict, request: Request):
    session_id = str((payload or {}).get("session_id") or "").strip()
    if not session_id:
        return JSONResponse(status_code=400, content={"detail": "session_id is required."})
    owner = str(payload.get("browser_id", "")).strip()
    if not owner:
        return JSONResponse(status_code=400, content={"detail": "Reload the page to initialize this browser session."})
    _reset_capture_live_session(session_id, owner)
    return {"ok": True}

@api_app.post("/capture-audio-live")
async def analyze_chrome_live(
    request: Request,
    chunk_sequence: int = Form(0),
    browser_id: str = Form(...),
    file: UploadFile = File(...),
    model_name: str = Form(PROPOSED_MODEL),
    session_id: str = Form(...),
):
    if not session_id.strip():
        return JSONResponse(status_code=400, content={"detail": "session_id is required."})
    if model_name not in MODEL_CHOICES:
        return JSONResponse(status_code=400, content={"detail": f"Unknown model: {model_name}"})
    owner = browser_id.strip()
    with _CAPTURE_LIVE_LOCK:
        ui = _CAPTURE_UI.get(owner)
        if not ui or ui["session_id"] != session_id.strip():
            return JSONResponse(status_code=409, content={"detail": "Capture session ended."})
    payload = await file.read(MAX_BYTES + 1)
    if not payload or len(payload) > MAX_BYTES:
        return JSONResponse(status_code=400, content={"detail": "Empty audio chunk."})
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(payload)
            temp_path = temp_file.name
        return await run_in_threadpool(_apply_capture_chunk, temp_path, model_name,
                                       session_id.strip(), owner, chunk_sequence)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@api_app.post("/capture-audio")
async def analyze_chrome_capture(
    request: Request,
    browser_id: str = Form(...),
    file: UploadFile = File(...),
    model_name: str = Form(PROPOSED_MODEL),
    session_id: str = Form(""),
):
    if file.content_type not in {"audio/wav", "audio/x-wav", "application/octet-stream", None}:
        # Some browsers omit or loosen content_type for Blob uploads.
        pass
    if model_name not in MODEL_CHOICES:
        return JSONResponse(status_code=400, content={"detail": f"Unknown model: {model_name}"})
    owner = browser_id.strip()
    with _CAPTURE_LIVE_LOCK:
        ui = _CAPTURE_UI.get(owner)
        if not ui or ui["session_id"] != session_id.strip():
            return JSONResponse(status_code=409, content={"detail": "Capture session ended."})
    payload = await file.read(MAX_BYTES + 1)
    if not payload or len(payload) > MAX_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Captured audio must be non-empty and 10 MB or smaller."})
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(payload)
            temp_path = temp_file.name
        rows, probability, label, latency, running_summary, status = await run_in_threadpool(
            analyze, temp_path, model_name, ignore_silence=True)
        if session_id:
            _clear_capture_live_session(session_id)
        payload = {
            "rows": rows,
            "probability": probability,
            "label": label,
            "latency": latency,
            "running_summary": running_summary,
            "status": status,
        }
        with _CAPTURE_LIVE_LOCK:
            current = _CAPTURE_UI.get(owner)
            if current and current["session_id"] == session_id:
                _publish_capture_final(payload, owner)
        return payload
    except Exception as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    if not DEFAULT_MODEL_READY:
        raise SystemExit(DEFAULT_MODEL_STATUS)
    if OPTIONS.check:
        runner = load_active_model(PROPOSED_MODEL)[0]
        signal = (.2*np.sin(2*np.pi*170*np.arange(20000)/SAMPLE_RATE)).astype(np.float32)
        result = analyze((SAMPLE_RATE, signal), PROPOSED_MODEL)
        if len(result[0]) != 3:
            raise SystemExit("The audio inference check failed.")
        print(f"Model ready. Device: {DEVICE}; saved threshold: {runner.threshold:.3f}")
        print(f"Code: {SOURCE_DIR}\nModels: {PROJECT_ROOT}")
        print("Generated-audio inference passed. No training/test dataset was used.")
        raise SystemExit(0)
    print(f"Open Google Chrome at http://127.0.0.1:{OPTIONS.port}; keep this process running.", flush=True)
    mounted_app = mount_gradio_app(
        api_app,
        demo,
        path="/",
        max_file_size="10mb",
        css=APP_CSS,
        js=CHROME_CAPTURE_SCRIPT + "\n" + TABLE_AUTO_FOLLOW_SCRIPT,
        head=f"<script>{CHROME_CAPTURE_SCRIPT}</script><script>{TABLE_AUTO_FOLLOW_SCRIPT}</script>",
    )
    uvicorn.run(mounted_app, host="127.0.0.1", port=OPTIONS.port)
