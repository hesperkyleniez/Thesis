"""Per-window end-to-end latency benchmark for the four retained models."""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r"D:\Thesis\code")

from config import MODELS_DIR, SAMPLE_RATE, DROPOUT
from features import extract_logmel, extract_mfcc, extract_cqcc, extract_f0, extract_ffv
from train import CNNOnly, CNNGRUOnly, CNNGRUFusion
from cqcc_train import CQCC_DIM, CQCCMLP, CQCC_VERSION
from ffv_train import FFV_DIM, FFVMLP, FFV_VERSION
from cqcc_ffv_fusion import load_fusion_model, fusion_predict_proba, summarize_probabilities

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CHOICES = ("CNN-only", "CNN-GRU", "CNN-GRU-F", "CNN-GRU-CQCC-FFV-F")


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def load_state(model, folder):
    path = os.path.join(MODELS_DIR, folder, "final_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_cqcc_stats():
    path = os.path.join(MODELS_DIR, "CQCC", "final_norm_stats.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    stats = np.load(path, allow_pickle=True).item()
    if stats.get("training_version") != CQCC_VERSION:
        raise RuntimeError(
            "CQCC final model/stats use an older CQCC definition. "
            "Run cqcc_train.py then cqcc_retrain.py first."
        )
    return (
        np.asarray(stats["mean"], dtype=np.float32),
        np.asarray(stats["std"], dtype=np.float32),
    )


def load_ffv_stats():
    path=os.path.join(MODELS_DIR,"FFV","final_norm_stats.npy")
    stats=np.load(path,allow_pickle=True).item()
    if stats.get("training_version") != FFV_VERSION: raise RuntimeError("FFV final model/stats incompatible.")
    return np.asarray(stats["mean"],np.float32),np.asarray(stats["std"],np.float32)

def build_runner(model_name):
    if model_name == "CNN-only":
        model = load_state(CNNOnly(dropout=0.0).to(DEVICE), "CNN-only")

        def run(audio):
            logmel = extract_logmel(audio)
            lm = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                return torch.sigmoid(model(lm)).item()
        return run

    if model_name == "CNN-GRU":
        model = load_state(CNNGRUOnly(dropout=0.0).to(DEVICE), "CNN-GRU")

        def run(audio):
            logmel = extract_logmel(audio)
            lm = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                return torch.sigmoid(model(lm, None, None)).item()
        return run

    if model_name == "CNN-GRU-F":
        model = load_state(CNNGRUFusion(dropout=0.0).to(DEVICE), "CNN-GRU-F")

        # Use final normalization stats exactly as evaluation does for MFCC.
        norm_path = os.path.join(MODELS_DIR, "CNN-GRU-F", "final_norm_stats.npy")
        stats = np.load(norm_path, allow_pickle=True).item()
        mfcc_mean = np.asarray(stats["mfcc_mean"], dtype=np.float32)
        mfcc_std = np.asarray(stats["mfcc_std"], dtype=np.float32)
        # A single latency window has no speaker-wide F0 context; benchmark the
        # feature extraction/model compute with the raw pooled F0 vector.

        def run(audio):
            logmel = extract_logmel(audio)
            mfcc = (extract_mfcc(audio) - mfcc_mean) / mfcc_std
            f0 = extract_f0(audio)
            lm = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            mc = torch.from_numpy(mfcc.astype(np.float32)).unsqueeze(0).to(DEVICE)
            f0t = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                return torch.sigmoid(model(lm, mc, f0t)).item()
        return run

    if model_name == "CNN-GRU-CQCC-FFV-F":
        cnn = load_state(CNNGRUOnly(dropout=0.0).to(DEVICE), "CNN-GRU")
        cqcc = load_state(CQCCMLP(CQCC_DIM).to(DEVICE), "CQCC")
        cqcc_mean, cqcc_std = load_cqcc_stats()
        ffv = load_state(FFVMLP(FFV_DIM).to(DEVICE), "FFV")
        ffv_mean, ffv_std = load_ffv_stats()
        fusion = load_fusion_model()

        def run(audio):
            logmel = extract_logmel(audio)
            cq = (extract_cqcc(audio) - cqcc_mean) / cqcc_std
            fv = (extract_ffv(audio) - ffv_mean) / ffv_std
            lm = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            cqt = torch.from_numpy(cq.astype(np.float32)).unsqueeze(0).to(DEVICE)
            fvt = torch.from_numpy(fv.astype(np.float32)).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                p_cnn = torch.sigmoid(cnn(lm, None, None)).item()
                p_cqcc = torch.sigmoid(cqcc(cqt)).item()
                p_ffv = torch.sigmoid(ffv(fvt)).item()
            x = summarize_probabilities([p_cnn], [p_cqcc], [p_ffv]).reshape(1, -1)
            return float(fusion_predict_proba(fusion, x)[0])
        return run

    raise ValueError(model_name)


def benchmark_one(model_name, n_runs):
    runner = build_runner(model_name)
    rng = np.random.default_rng(42)

    for _ in range(10):
        runner(rng.normal(0, 0.1, SAMPLE_RATE).astype(np.float32))
    sync()

    times = []
    for _ in range(n_runs):
        audio = rng.normal(0, 0.1, SAMPLE_RATE).astype(np.float32)
        sync()
        t0 = time.perf_counter()
        runner(audio)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    values = np.asarray(times)
    return {
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CHOICES) + ["all"], default="all")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    names = MODEL_CHOICES if args.model == "all" else (args.model,)
    print(f"Device: {DEVICE}")
    print(f"Runs per model: {args.runs}")
    print("100 ms per-window target\n")

    for name in names:
        result = benchmark_one(name, args.runs)
        print(
            f"{name:<18} mean={result['mean_ms']:.2f} ms  "
            f"std={result['std_ms']:.2f} ms  p95={result['p95_ms']:.2f} ms  "
            f"{'MET' if result['p95_ms'] <= 100.0 else 'NOT MET'}"
        )


if __name__ == "__main__":
    main()
