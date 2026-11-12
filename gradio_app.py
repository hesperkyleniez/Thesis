"""Small Gradio interface for windowed Filipino speech deepfake detection."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

from improvedCQCC_config import HOP_SAMPLES, SAMPLE_RATE, WIN_SAMPLES
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


def get_audio_path(value):
    if isinstance(value, dict):
        return value.get("path")
    return value


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


def analyze(audio_value, model_name, checkpoint_file, checkpoint_path):
    path = get_audio_path(audio_value)
    if not path:
        raise gr.Error("Upload an audio file or record a clip with the microphone.")
    path = Path(path)
    if path.suffix.lower() not in {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au"}:
        raise gr.Error("Use a supported audio file: WAV, FLAC, OGG, AIFF, or AU.")
    if path.stat().st_size > MAX_BYTES:
        raise gr.Error("Audio files must be 10 MB or smaller.")
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("The audio is empty or invalid.")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
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
        window_start = time.perf_counter()
        if feature == "logmel":
            x = torch.from_numpy(extract_logmel(window)[None, None]).float().to(DEVICE)
            with torch.inference_mode():
                logit = model(x)
        else:
            vector = extract_cqcc(window) if feature == "cqcc" else extract_f0(window)
            x = torch.from_numpy(vector[None]).float().to(DEVICE)
            with torch.inference_mode():
                logit = model(x)
        p_ai = float(torch.sigmoid(logit.reshape(-1)[0]).item())
        probabilities.append(p_ai)
        label = "AI" if p_ai >= 0.5 else "Real"
        rows.append([index, f"{p_ai * 100:.1f}%", label, f"{(time.perf_counter() - window_start) * 1000:.1f} ms"])
    total_ms = (time.perf_counter() - started) * 1000
    final_ai = float(np.mean(probabilities))
    final_label = "AI" if final_ai >= 0.5 else "Real"
    return rows, f"{final_ai * 100:.1f}%", final_label, f"{total_ms:.1f} ms"


with gr.Blocks(title="Speech Authenticity") as demo:
    gr.Markdown("# Speech Authenticity\nWindow level AI speech detection")
    with gr.Row():
        model_select = gr.Dropdown(list(MODELS), value="CNN GRU", label="Model")
        checkpoint = gr.File(label="Model checkpoint (.pt)", file_types=[".pt"], type="filepath")
    checkpoint_path = gr.Textbox(label="Or enter a local .pt file path", placeholder=r"D:\Thesis\models\checkpoint.pt")
    audio_input = gr.Audio(label="Audio clip", sources=["upload", "microphone"], type="filepath", format="wav")
    run_button = gr.Button("Analyze", variant="primary")
    with gr.Row():
        final_percent = gr.Textbox(label="Clip AI probability", interactive=False)
        final_label = gr.Textbox(label="Final label", interactive=False)
        total_latency = gr.Textbox(label="Clip latency", interactive=False)
    per_window = gr.Dataframe(headers=["Window", "AI probability", "Label", "Latency"], label="Per window results", interactive=False)
    run_button.click(analyze, [audio_input, model_select, checkpoint, checkpoint_path], [per_window, final_percent, final_label, total_latency])


if __name__ == "__main__":
    demo.launch(max_file_size="10mb")
