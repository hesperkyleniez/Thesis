# save as E:\Thesis\code\latency_test.py
import torch
import numpy as np
import librosa
import time
from scipy.ndimage import zoom
import sys
sys.path.insert(0, r"D:\Thesis\code")

device = torch.device("cuda")

# Load a real model to test with
from train import CNNGRUFusion
from config import MODELS_DIR, N_MELS, N_MFCC, N_FFT, HOP_LENGTH, TARGET_SHAPE, F0_MIN, F0_MAX, SAMPLE_RATE
import os, glob

model = CNNGRUFusion(dropout=0.0).to(device)

# Load best fold 1 model of CNN-GRU-F if available
model_path = os.path.join(MODELS_DIR, "CNN-GRU-F", "best_fold1.pt")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("✅ Loaded CNN-GRU-F model")
else:
    print("⚠️  Using random weights — model not trained yet")

model.eval()

# Simulate one 1-second audio window (16kHz)
dummy_audio = np.random.randn(16000).astype(np.float32)

# ── Warmup GPU (first inference is always slower) ─────────────────────────────
print("Warming up GPU...")
for _ in range(10):
    with torch.no_grad():
        t = torch.randn(1, 1, 128, 128).to(device)
        m = torch.randn(1, 80).to(device)
        f = torch.randn(1, 2).to(device)
        _ = model(t, m, f)
print("✅ Warmup done\n")

# ── Measure each step separately ─────────────────────────────────────────────
N_RUNS = 100
times = {
    "logmel"     : [],
    "mfcc"       : [],
    "f0"         : [],
    "total_feat" : [],
    "inference"  : [],
    "total"      : []
}

for _ in range(N_RUNS):
    audio = np.random.randn(16000).astype(np.float32)

    total_start = time.perf_counter()

    # Log-Mel
    t0 = time.perf_counter()
    mel     = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, window='hamming', power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape != TARGET_SHAPE:
        factors = (TARGET_SHAPE[0]/log_mel.shape[0],
                   TARGET_SHAPE[1]/log_mel.shape[1])
        log_mel = zoom(log_mel, factors)
    mn, mx  = log_mel.min(), log_mel.max()
    log_mel = ((log_mel - mn) / (mx - mn + 1e-8)).astype(np.float32)
    times["logmel"].append((time.perf_counter() - t0) * 1000)

    # MFCC
    t0 = time.perf_counter()
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc_vec = np.concatenate([mfcc.mean(axis=1),
                               mfcc.std(axis=1)]).astype(np.float32)
    times["mfcc"].append((time.perf_counter() - t0) * 1000)

    # F0
    t0 = time.perf_counter()
    f0 = librosa.yin(
        audio, fmin=F0_MIN, fmax=F0_MAX,
        sr=SAMPLE_RATE, hop_length=4096, frame_length=8192
    )
    voiced = f0[(f0 > F0_MIN) & (f0 < F0_MAX)]
    f0_vec = np.array(
        [np.mean(voiced), np.std(voiced)] if len(voiced) > 0
        else [0.0, 0.0], dtype=np.float32
    )
    times["f0"].append((time.perf_counter() - t0) * 1000)

    feat_time = times["logmel"][-1] + times["mfcc"][-1] + times["f0"][-1]
    times["total_feat"].append(feat_time)

    # GPU Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        logmel_t = torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0).to(device)
        mfcc_t   = torch.FloatTensor(mfcc_vec).unsqueeze(0).to(device)
        f0_t     = torch.FloatTensor(f0_vec).unsqueeze(0).to(device)
        logit    = model(logmel_t, mfcc_t, f0_t)
        prob     = torch.sigmoid(logit).item()
    times["inference"].append((time.perf_counter() - t0) * 1000)

    total_time = (time.perf_counter() - total_start) * 1000
    times["total"].append(total_time)

# ── Results ───────────────────────────────────────────────────────────────────
print("="*55)
print("LATENCY BENCHMARK — 100 runs on RTX 2060")
print("="*55)
print(f"{'Step':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'95th%':>8}")
print(f"{'-'*55}")

for key, label in [
    ("logmel",      "Log-Mel extraction"),
    ("mfcc",        "MFCC extraction"),
    ("f0",          "F0 extraction"),
    ("total_feat",  "Total features"),
    ("inference",   "GPU inference"),
    ("total",       "TOTAL end-to-end"),
]:
    vals = np.array(times[key])
    print(f"{label:<20} "
          f"{np.mean(vals):>7.2f}ms "
          f"{np.std(vals):>7.2f}ms "
          f"{np.min(vals):>7.2f}ms "
          f"{np.percentile(vals, 95):>7.2f}ms")

print(f"\n{'='*55}")
total_mean = np.mean(times["total"])
p95        = np.percentile(times["total"], 95)
print(f"  Mean total latency : {total_mean:.2f}ms")
print(f"  95th percentile    : {p95:.2f}ms")
print(f"  100ms constraint   : {'✅ MET' if p95 < 100 else '❌ NOT MET'}")
print(f"{'='*55}")