# latency_test.py
# End-to-end latency benchmark for the v9 primary CNN-GRU pipeline.
#
# Default mode benchmarks the primary CNN-GRU model, which is the system's
# latency-oriented detector. --mode fusion additionally measures prediction
# fusion with MFCC/CQCC/F0 experts when final checkpoints exist.

import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, r"D:\Thesis\code")
from config import (
    MODELS_DIR, SAMPLE_RATE, N_MELS, N_MFCC, N_FFT, HOP_LENGTH,
    TARGET_SHAPE, F0_MIN, F0_MAX,
)
from features import extract_logmel, extract_mfcc, extract_cqcc, extract_f0
from train import CNNGRUOnly

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def load_cnn():
    model = CNNGRUOnly(dropout=0.0).to(DEVICE)
    path = os.path.join(MODELS_DIR, "CNN-GRU", "final_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def benchmark(n_runs=100):
    model = load_cnn()
    dummy = np.random.randn(SAMPLE_RATE).astype(np.float32)

    for _ in range(10):
        lm = torch.from_numpy(extract_logmel(dummy)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        mc = torch.from_numpy(extract_mfcc(dummy)).unsqueeze(0).to(DEVICE)
        f0 = torch.from_numpy(extract_f0(dummy)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _ = model(lm, mc, f0)
    sync()

    times = {"logmel": [], "mfcc": [], "f0": [], "cnn_inference": [], "total": []}

    for _ in range(n_runs):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        t_total = time.perf_counter()

        t = time.perf_counter()
        logmel = extract_logmel(audio)
        times["logmel"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        mfcc = extract_mfcc(audio)
        times["mfcc"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        f0 = extract_f0(audio)
        times["f0"].append((time.perf_counter() - t) * 1000)

        lm = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)
        mc = torch.from_numpy(mfcc).unsqueeze(0).to(DEVICE)
        f0t = torch.from_numpy(f0).unsqueeze(0).to(DEVICE)

        sync()
        t = time.perf_counter()
        with torch.no_grad():
            _ = model(lm, mc, f0t)
        sync()
        times["cnn_inference"].append((time.perf_counter() - t) * 1000)

        times["total"].append((time.perf_counter() - t_total) * 1000)

    print("=" * 65)
    print("V9 LATENCY BENCHMARK — CNN-GRU PRIMARY DETECTOR")
    print("=" * 65)
    print(f"{'Step':<24}{'Mean':>10}{'Std':>10}{'95th%':>10}")
    print("-" * 65)

    for key, label in [
        ("logmel", "Log-Mel extraction"),
        ("mfcc", "MFCC extraction"),
        ("f0", "F0 extraction"),
        ("cnn_inference", "CNN-GRU inference"),
        ("total", "TOTAL end-to-end"),
    ]:
        vals = np.asarray(times[key])
        print(
            f"{label:<24}"
            f"{np.mean(vals):>9.2f}ms"
            f"{np.std(vals):>9.2f}ms"
            f"{np.percentile(vals,95):>9.2f}ms"
        )

    p95 = np.percentile(times["total"], 95)
    print("-" * 65)
    print(f"95th percentile total: {p95:.2f} ms")
    print(f"100 ms target       : {'MET' if p95 < 100 else 'NOT MET'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    benchmark(args.runs)
