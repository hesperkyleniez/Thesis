# prediction_fusion.py
# Prediction-level late fusion for the experimental matrix.
#
# Base experts:
#   CNN-GRU Log-Mel
#   MFCC MLP
#   CQCC MLP
#   F0 MLP
#
# Fusion weights are learned from validation predictions only, using a
# constrained convex combination and binary log-loss. Test data is never used
# to fit fusion weights.
#
# This script does NOT retrain the base experts. Run train.py, aux_train.py,
# retrain.py, and aux_retrain.py first.

import os
import sys
import glob
import itertools
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

sys.path.insert(0, r"D:\Thesis\code")

from config import (
    FEATURES_DIR, MODELS_DIR, RESULTS_DIR, FOLD_ASSIGNMENTS, SEED,
)
from train import (
    CNNGRUOnly, compute_per_speaker_f0_stats,
    lookup_f0_stats, set_seed,
)
from aux_train import AUX_SPECS, AuxMLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION = "v9_prediction_fusion"


EXPERTS = {
    "CNN-GRU": {"kind": "cnn"},
    "MFCC": {"kind": "aux", "feature": "mfcc"},
    "CQCC": {"kind": "aux", "feature": "cqcc"},
    "F0": {"kind": "aux", "feature": "f0"},
}

FUSION_CONFIGS = {
    "CNN-GRU+MFCC": ["CNN-GRU", "MFCC"],
    "CNN-GRU+CQCC": ["CNN-GRU", "CQCC"],
    "CNN-GRU+F0": ["CNN-GRU", "F0"],
    "CNN-GRU+MFCC+CQCC": ["CNN-GRU", "MFCC", "CQCC"],
    "CNN-GRU+MFCC+F0": ["CNN-GRU", "MFCC", "F0"],
    "CNN-GRU+CQCC+F0": ["CNN-GRU", "CQCC", "F0"],
    "CNN-GRU+MFCC+CQCC+F0": ["CNN-GRU", "MFCC", "CQCC", "F0"],
}


def get_train_files():
    return sorted(
        glob.glob(
            os.path.join(FEATURES_DIR, "train", "**", "*.npz"),
            recursive=True,
        )
    )


def get_split_files(split):
    return sorted(
        glob.glob(
            os.path.join(FEATURES_DIR, split, "**", "*.npz"),
            recursive=True,
        )
    )


def get_fold_files(files, fold):
    train, val = [], []
    for path in files:
        speaker = os.path.basename(os.path.dirname(path))
        assigned = FOLD_ASSIGNMENTS.get(speaker)
        if assigned is None:
            continue
        (val if assigned == fold else train).append(path)
    return sorted(train), sorted(val)


def load_state(model_name, fold=None, final=False):
    model_dir = os.path.join(MODELS_DIR, model_name)

    if final:
        path = os.path.join(model_dir, "final_model.pt")
    else:
        path = os.path.join(model_dir, f"best_fold{fold}.pt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    return torch.load(path, map_location=DEVICE, weights_only=True)


def load_cnn_fold_stats(fold):
    path = os.path.join(MODELS_DIR, "CNN-GRU", "fold_results.npy")
    results = list(np.load(path, allow_pickle=True))
    for r in results:
        if int(r["fold"]) == fold:
            norm = r["norm"]
            return (
                np.asarray(norm["mfcc_mean"], dtype=np.float32),
                np.asarray(norm["mfcc_std"], dtype=np.float32),
                norm["f0_speaker_stats"],
                norm["f0_fallback"],
            )
    raise ValueError(f"CNN-GRU fold {fold} normalization stats not found.")


def load_aux_fold_stats(name, fold):
    path = os.path.join(MODELS_DIR, name, "fold_results.npy")
    results = list(np.load(path, allow_pickle=True))
    for r in results:
        if int(r["fold"]) == fold:
            return (
                np.asarray(r["mean"], dtype=np.float32),
                np.asarray(r["std"], dtype=np.float32),
            )
    raise ValueError(f"{name} fold {fold} stats not found.")


def load_aux_final_stats(name):
    path = os.path.join(MODELS_DIR, name, "final_norm_stats.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True).item()
    return np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32)


def predict_cnn(model, files, mfcc_mean, mfcc_std, f0_stats, f0_fallback):
    model.eval()
    results = {}

    mfcc_mean_t = torch.as_tensor(mfcc_mean, dtype=torch.float32, device=DEVICE)
    mfcc_std_t = torch.as_tensor(mfcc_std, dtype=torch.float32, device=DEVICE)

    for path in tqdm(files, desc="CNN-GRU predictions", leave=False):
        with np.load(path, allow_pickle=True) as d:
            lm = torch.as_tensor(d["logmel"], dtype=torch.float32, device=DEVICE).unsqueeze(1)
            mc = (
                torch.as_tensor(d["mfcc"], dtype=torch.float32, device=DEVICE)
                - mfcc_mean_t
            ) / mfcc_std_t

            f0_mean, f0_std = lookup_f0_stats(path, f0_stats, f0_fallback)
            f0 = (
                torch.as_tensor(d["f0"], dtype=torch.float32, device=DEVICE)
                - torch.as_tensor(f0_mean, dtype=torch.float32, device=DEVICE)
            ) / torch.as_tensor(f0_std, dtype=torch.float32, device=DEVICE)

            label = int(d["label"])

        probs = []
        bs = 256
        with torch.no_grad():
            for i in range(0, len(lm), bs):
                logits = model(lm[i:i+bs], mc[i:i+bs], f0[i:i+bs])
                probs.extend(torch.sigmoid(logits).float().cpu().numpy().reshape(-1))

        results[path] = {"score": float(np.mean(probs)), "label": label}

    return results


def predict_aux(model_name, model, files, mean, std):
    model.eval()
    feature = AUX_SPECS[model_name]["feature"]
    results = {}

    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=DEVICE)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=DEVICE)

    for path in tqdm(files, desc=f"{model_name} predictions", leave=False):
        with np.load(path, allow_pickle=True) as d:
            x = (
                torch.as_tensor(d[feature], dtype=torch.float32, device=DEVICE)
                - mean_t
            ) / std_t
            label = int(d["label"])

        probs = []
        with torch.no_grad():
            for i in range(0, len(x), 512):
                logits = model(x[i:i+512])
                probs.extend(torch.sigmoid(logits).float().cpu().numpy().reshape(-1))

        results[path] = {"score": float(np.mean(probs)), "label": label}

    return results


def collect_expert_predictions(experts, files, fold=None, final=False):
    out = {}

    if "CNN-GRU" in experts:
        model = CNNGRUOnly(dropout=0.0).to(DEVICE)
        model.load_state_dict(load_state("CNN-GRU", fold=fold, final=final))

        if final:
            norm = np.load(
                os.path.join(MODELS_DIR, "CNN-GRU", "final_norm_stats.npy"),
                allow_pickle=True,
            ).item()
            mfcc_mean = norm["mfcc_mean"]
            mfcc_std = norm["mfcc_std"]
            f0_stats = norm["f0_speaker_stats"]
            f0_fallback = norm["f0_fallback"]
        else:
            mfcc_mean, mfcc_std, _, _ = load_cnn_fold_stats(fold)
            # Match train.py validation-time F0 normalization exactly:
            # validation speakers are normalized using statistics computed from
            # their own validation files, without labels.
            val_f0_stats, val_f0_fallback = compute_per_speaker_f0_stats(files)
            f0_stats, f0_fallback = val_f0_stats, val_f0_fallback

        out["CNN-GRU"] = predict_cnn(
            model, files, mfcc_mean, mfcc_std, f0_stats, f0_fallback
        )
        del model

    for name in experts:
        if name == "CNN-GRU":
            continue

        model = AuxMLP(AUX_SPECS[name]["dim"]).to(DEVICE)
        model.load_state_dict(load_state(name, fold=fold, final=final))

        if final:
            mean, std = load_aux_final_stats(name)
        else:
            mean, std = load_aux_fold_stats(name, fold)

        out[name] = predict_aux(name, model, files, mean, std)
        del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return out


def optimize_weights(expert_scores, labels):
    """
    Learn non-negative weights that sum to one by minimizing validation
    binary cross entropy. This avoids fitting a high-capacity meta-classifier.
    """
    scores = np.asarray(expert_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    n = scores.shape[1]

    def softmax(z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def objective(z):
        w = softmax(z)
        p = np.clip(scores @ w, 1e-6, 1 - 1e-6)
        return -np.mean(
            labels * np.log(p) + (1 - labels) * np.log(1 - p)
        )

    x0 = np.zeros(n, dtype=np.float64)
    result = minimize(objective, x0, method="BFGS")
    weights = softmax(result.x)
    return weights.astype(np.float32)


def calculate_metrics(labels, probs):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[i] + fnr[i]) / 2.0)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
        "eer": eer,
    }


def fuse_results(predictions, expert_names, weights):
    paths = sorted(predictions[expert_names[0]].keys())
    scores = []
    labels = []

    for path in paths:
        scores.append([
            predictions[name][path]["score"] for name in expert_names
        ])
        labels.append(predictions[expert_names[0]][path]["label"])

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    fused = scores @ np.asarray(weights, dtype=np.float32)

    return labels, fused


def run_cv_fusion(config_name, expert_names):
    print(f"\n{'='*72}")
    print(f"CV PREDICTION FUSION: {config_name}")
    print(f"Experts: {expert_names}")
    print(f"{'='*72}")

    all_labels = []
    all_probs = []
    fold_metrics = []
    fold_weights = []

    train_files = get_train_files()

    for fold in [1, 2, 3, 4]:
        _, val_files = get_fold_files(train_files, fold)

        preds = collect_expert_predictions(
            expert_names, val_files, fold=fold, final=False
        )

        paths = sorted(preds[expert_names[0]].keys())
        matrix = np.asarray([
            [preds[name][p]["score"] for name in expert_names]
            for p in paths
        ], dtype=np.float64)
        labels = np.asarray([
            preds[expert_names[0]][p]["label"] for p in paths
        ], dtype=int)

        weights = optimize_weights(matrix, labels)
        fused = matrix @ weights
        metrics = calculate_metrics(labels, fused)

        fold_weights.append(weights)
        fold_metrics.append(metrics)
        all_labels.extend(labels.tolist())
        all_probs.extend(fused.tolist())

        print(
            f"Fold {fold}: "
            f"F1={metrics['f1']:.4f} "
            f"Acc={metrics['accuracy']:.4f} "
            f"AUC={metrics['auc']:.4f} "
            f"EER={metrics['eer']:.4f} "
            f"weights={np.round(weights,4)}"
        )

    weights = np.mean(np.stack(fold_weights), axis=0)
    weights = weights / weights.sum()

    summary = {
        "training_version": VERSION,
        "configuration": config_name,
        "experts": expert_names,
        "weights": weights,
        "fold_metrics": fold_metrics,
        "mean_f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "std_f1": float(np.std([m["f1"] for m in fold_metrics])),
    }

    out_path = os.path.join(
        MODELS_DIR,
        "Prediction-Fusion",
        f"{config_name}.npy",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, summary)

    print(
        f"Mean F1={summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}"
    )
    print(f"Learned weights: {dict(zip(expert_names, weights))}")

    return summary


def evaluate_final(config_name, expert_names, weights, split):
    files = get_split_files(split)
    preds = collect_expert_predictions(
        expert_names, files, final=True
    )
    labels, probs = fuse_results(preds, expert_names, weights)
    metrics = calculate_metrics(labels, probs)

    print(
        f"{config_name} | {split}: "
        f"Acc={metrics['accuracy']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"Prec={metrics['precision']:.4f} "
        f"Rec={metrics['recall']:.4f} "
        f"AUC={metrics['auc']:.4f} "
        f"EER={metrics['eer']:.4f}"
    )
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fusion",
        choices=list(FUSION_CONFIGS) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    set_names = (
        list(FUSION_CONFIGS)
        if args.fusion == "all"
        else [args.fusion]
    )

    summaries = []

    for name in set_names:
        experts = FUSION_CONFIGS[name]
        summary = run_cv_fusion(name, experts)

        # The averaged CV weights are frozen before touching held-out speakers.
        weights = np.asarray(summary["weights"], dtype=np.float32)

        for split in ["test", "test_degraded"]:
            evaluate_final(name, experts, weights, split)

        summaries.append(summary)

    np.save(
        os.path.join(RESULTS_DIR, "prediction_fusion_results.npy"),
        summaries,
    )
    print("\nPrediction fusion experiments complete.")


if __name__ == "__main__":
    set_seed(SEED)
    main()
