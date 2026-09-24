# evaluate_aux.py
# Evaluate MFCC, CQCC, and F0 auxiliary models on clean and degraded held-out test.
# Uses final models and development-derived normalization statistics only.

import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

sys.path.insert(0, r"D:\Thesis\code")
from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, SEED
from aux_train import AUX_SPECS, AuxMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_files(split):
    return sorted(
        glob.glob(
            os.path.join(FEATURES_DIR, split, "**", "*.npz"),
            recursive=True,
        )
    )


def load_stats(name):
    d = np.load(
        os.path.join(MODELS_DIR, name, "final_norm_stats.npy"),
        allow_pickle=True,
    ).item()
    return d["mean"], d["std"]


def eer(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def evaluate(name, split):
    feature = AUX_SPECS[name]["feature"]
    mean, std = load_stats(name)

    model = AuxMLP(AUX_SPECS[name]["dim"]).to(DEVICE)
    path = os.path.join(MODELS_DIR, name, "final_model.pt")
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()

    clip_scores = []
    labels = []

    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=DEVICE)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=DEVICE)

    for path in tqdm(get_files(split), desc=f"{name} {split}", leave=False):
        with np.load(path, allow_pickle=True) as d:
            x = (
                torch.as_tensor(d[feature], dtype=torch.float32, device=DEVICE)
                - mean_t
            ) / std_t
            label = int(d["label"])

        with torch.no_grad():
            probs = []
            for i in range(0, len(x), 512):
                logits = model(x[i:i+512])
                probs.extend(
                    torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
                )

        clip_scores.append(float(np.mean(probs)))
        labels.append(label)

    probs = np.asarray(clip_scores)
    labels = np.asarray(labels)
    preds = (probs >= 0.5).astype(int)

    result = {
        "model": name,
        "split": split,
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
        "eer": eer(labels, probs),
        "cm": confusion_matrix(labels, preds),
    }

    print(
        f"{name} | {split}: "
        f"Acc={result['accuracy']:.4f} "
        f"F1={result['f1']:.4f} "
        f"Prec={result['precision']:.4f} "
        f"Rec={result['recall']:.4f} "
        f"AUC={result['auc']:.4f} "
        f"EER={result['eer']:.4f}"
    )
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(AUX_SPECS) + ["all"], default="all")
    args = parser.parse_args()

    names = list(AUX_SPECS) if args.model == "all" else [args.model]
    results = []

    for name in names:
        for split in ["test", "test_degraded"]:
            results.append(evaluate(name, split))

    np.save(
        os.path.join(RESULTS_DIR, "aux_test_results.npy"),
        results,
    )
    print("\nAuxiliary model evaluation complete.")
