# cqcc_fusion.py
# CNN-GRU + CQCC prediction-level late fusion.
#
# Independent experts:
#   Log-Mel -> CNN-GRU -> window probabilities
#   CQCC    -> CQCC MLP -> window probabilities
# The fusion classifier is trained only from speaker-grouped OOF predictions.
# Both clean and degraded evaluation use the thesis decision threshold 0.50.

import os
import sys
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    log_loss,
    confusion_matrix,
)

sys.path.insert(0, r"D:\Thesis\code")

from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, FOLD_ASSIGNMENTS, SEED
from train import CNNGRUOnly, set_seed
from cqcc_train import CQCC_DIM, CQCCMLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION = "v11_cnn_gru_cqcc_oof_late_fusion"
OOF_CACHE_VERSION = VERSION
FUSION_NAME = "CNN-GRU-CQCC-F"
MODEL_DIR = os.path.join(MODELS_DIR, "Prediction-Fusion", FUSION_NAME)
MODEL_PATH = os.path.join(MODEL_DIR, "fusion_model.npy")
RESULT_PATH = os.path.join(RESULTS_DIR, "cnn_gru_cqcc_fusion_results.npy")
OOF_CACHE_PATH = os.path.join(MODEL_DIR, "oof_predictions.npy")

# Small regularization search. Selection is speaker-grouped at the meta level.
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

# Chapter III uses a fixed 0.50 decision threshold for all evaluations.
DECISION_THRESHOLD = 0.50

# Inference batch sizes. These affect speed only, not methodology.
CNN_BATCH_SIZE = 256
CQCC_BATCH_SIZE = 512


# -----------------------------------------------------------------------------
# File handling
# -----------------------------------------------------------------------------
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
    train_files, val_files = [], []
    for path in files:
        speaker = os.path.basename(os.path.dirname(path))
        assigned = FOLD_ASSIGNMENTS.get(speaker)
        if assigned is None:
            continue
        (val_files if assigned == fold else train_files).append(path)
    return sorted(train_files), sorted(val_files)


def load_state(model_name, fold=None, final=False):
    model_dir = os.path.join(MODELS_DIR, model_name)
    if final:
        path = os.path.join(model_dir, "final_model.pt")
    else:
        path = os.path.join(model_dir, f"best_fold{fold}.pt")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing checkpoint: {path}\n"
            "Run the corresponding CV/final retraining step first."
        )
    return torch.load(path, map_location=DEVICE, weights_only=True)


def load_cqcc_fold_stats(fold):
    path = os.path.join(MODELS_DIR, "CQCC", "fold_results.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    results = list(np.load(path, allow_pickle=True))
    for result in results:
        if int(result["fold"]) == int(fold):
            return (
                np.asarray(result["mean"], dtype=np.float32),
                np.asarray(result["std"], dtype=np.float32),
            )
    raise ValueError(f"CQCC fold {fold} normalization stats not found.")


def load_cqcc_final_stats():
    path = os.path.join(MODELS_DIR, "CQCC", "final_norm_stats.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True).item()
    return (
        np.asarray(d["mean"], dtype=np.float32),
        np.asarray(d["std"], dtype=np.float32),
    )


# -----------------------------------------------------------------------------
# Independent expert predictions
# -----------------------------------------------------------------------------
def read_clip_metadata(path):
    with np.load(path, allow_pickle=True) as d:
        label = int(d["label"])
        if "condition" in d.files:
            condition = str(d["condition"].item())
        else:
            condition = "clean"
    speaker = os.path.basename(os.path.dirname(path))
    return label, condition, speaker


def predict_cnn_windows(model, files):
    """Window-level CNN-GRU probabilities. Only Log-Mel is read/used."""
    model.eval()
    out = {}

    with torch.inference_mode():
        for path in tqdm(files, desc="CNN-GRU predictions", leave=False):
            with np.load(path, allow_pickle=True) as d:
                x = torch.as_tensor(
                    d["logmel"], dtype=torch.float32, device=DEVICE
                ).unsqueeze(1)
                label = int(d["label"])
                condition = (
                    str(d["condition"].item())
                    if "condition" in d.files
                    else "clean"
                )

            probs = []
            for i in range(0, len(x), CNN_BATCH_SIZE):
                logits = model(x[i:i + CNN_BATCH_SIZE], None, None)
                probs.extend(
                    torch.sigmoid(logits)
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .tolist()
                )

            out[path] = {
                "window_probs": np.asarray(probs, dtype=np.float32),
                "label": label,
                "condition": condition,
                "speaker": os.path.basename(os.path.dirname(path)),
            }

    return out


def predict_cqcc_windows(model, files, mean, std):
    """Window-level CQCC auxiliary probabilities."""
    model.eval()
    out = {}

    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=DEVICE)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=DEVICE)

    with torch.inference_mode():
        for path in tqdm(files, desc="CQCC predictions", leave=False):
            with np.load(path, allow_pickle=True) as d:
                x = (
                    torch.as_tensor(d["cqcc"], dtype=torch.float32, device=DEVICE)
                    - mean_t
                ) / std_t
                label = int(d["label"])
                condition = (
                    str(d["condition"].item())
                    if "condition" in d.files
                    else "clean"
                )

            probs = []
            for i in range(0, len(x), CQCC_BATCH_SIZE):
                logits = model(x[i:i + CQCC_BATCH_SIZE])
                probs.extend(
                    torch.sigmoid(logits)
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .tolist()
                )

            out[path] = {
                "window_probs": np.asarray(probs, dtype=np.float32),
                "label": label,
                "condition": condition,
                "speaker": os.path.basename(os.path.dirname(path)),
            }

    return out


def collect_expert_predictions(files, fold=None, final=False):
    """
    Load the two independent experts and return window-level probabilities.
    No intermediate neural features are exposed to the fusion model.
    """
    cnn = CNNGRUOnly(dropout=0.0).to(DEVICE)
    cnn.load_state_dict(load_state("CNN-GRU", fold=fold, final=final))

    cqcc = CQCCMLP(CQCC_DIM).to(DEVICE)
    cqcc.load_state_dict(load_state("CQCC", fold=fold, final=final))

    if final:
        cqcc_mean, cqcc_std = load_cqcc_final_stats()
    else:
        cqcc_mean, cqcc_std = load_cqcc_fold_stats(fold)

    cnn_preds = predict_cnn_windows(cnn, files)
    cqcc_preds = predict_cqcc_windows(cqcc, files, cqcc_mean, cqcc_std)

    del cnn, cqcc
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    common = sorted(set(cnn_preds) & set(cqcc_preds))
    if len(common) != len(files):
        missing = len(files) - len(common)
        raise RuntimeError(f"Missing paired expert predictions for {missing} clips.")

    records = []
    for path in common:
        a = cnn_preds[path]
        b = cqcc_preds[path]

        if a["label"] != b["label"]:
            raise RuntimeError(f"Label mismatch between experts: {path}")

        pa = a["window_probs"]
        pb = b["window_probs"]
        if len(pa) != len(pb):
            raise RuntimeError(
                f"Window-count mismatch for {path}: CNN-GRU={len(pa)}, CQCC={len(pb)}"
            )

        records.append({
            "path": path,
            "speaker": a["speaker"],
            "condition": a["condition"],
            "label": int(a["label"]),
            "cnn_probs": pa,
            "cqcc_probs": pb,
        })

    return records


# -----------------------------------------------------------------------------
# Prediction-level late-fusion features
# -----------------------------------------------------------------------------
FEATURE_NAMES = [
    "cnn_mean",
    "cnn_median",
    "cnn_q75",
    "cnn_std",
    "cnn_ai_fraction",
    "cqcc_mean",
    "cqcc_median",
    "cqcc_q75",
    "cqcc_std",
    "cqcc_ai_fraction",
    "mean_abs_disagreement",
    "window_agreement_rate",
    "mean_probability_product",
]


def summarize_probabilities(cnn_probs, cqcc_probs):
    """
    Robust clip-level summary built ONLY from the two experts' predictions.

    This remains prediction-level/decision-level late fusion. No Log-Mel, CQCC,
    CNN embeddings, or auxiliary hidden features are passed into the fusion
    classifier.

    The extra window-summary terms specifically address the degraded-test
    failure mode where a few very low-confidence windows can pull a plain mean
    below 0.5 even when the rest of the clip carries strong AI evidence.
    """
    cnn = np.asarray(cnn_probs, dtype=np.float64)
    cqcc = np.asarray(cqcc_probs, dtype=np.float64)

    if len(cnn) == 0 or len(cqcc) == 0:
        raise ValueError("Cannot fuse an empty clip.")
    if len(cnn) != len(cqcc):
        raise ValueError("CNN-GRU and CQCC window probabilities must align.")

    cnn_mean = float(np.mean(cnn))
    cqcc_mean = float(np.mean(cqcc))

    return np.asarray([
        cnn_mean,
        float(np.median(cnn)),
        float(np.quantile(cnn, 0.75)),
        float(np.std(cnn)),
        float(np.mean(cnn >= 0.5)),
        cqcc_mean,
        float(np.median(cqcc)),
        float(np.quantile(cqcc, 0.75)),
        float(np.std(cqcc)),
        float(np.mean(cqcc >= 0.5)),
        float(np.mean(np.abs(cnn - cqcc))),
        float(np.mean((cnn >= 0.5) == (cqcc >= 0.5))),
        cnn_mean * cqcc_mean,
    ], dtype=np.float64)


def records_to_matrix(records):
    x = np.stack([
        summarize_probabilities(r["cnn_probs"], r["cqcc_probs"])
        for r in records
    ])
    y = np.asarray([r["label"] for r in records], dtype=int)
    groups = np.asarray([r["speaker"] for r in records], dtype=object)
    conditions = np.asarray([r["condition"] for r in records], dtype=object)
    paths = np.asarray([r["path"] for r in records], dtype=object)
    return x, y, groups, conditions, paths


# -----------------------------------------------------------------------------
# OOF fusion fitting
# -----------------------------------------------------------------------------
def collect_oof_predictions(rebuild_cache=False):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(OOF_CACHE_PATH) and not rebuild_cache:
        try:
            cached = np.load(OOF_CACHE_PATH, allow_pickle=True).item()
            if cached.get("version") in {VERSION, OOF_CACHE_VERSION}:
                print(f"Using cached OOF predictions: {OOF_CACHE_PATH}")
                return list(cached["records"])
        except Exception:
            pass

    train_files = get_train_files()
    if not train_files:
        raise RuntimeError(
            f"No development feature files found under {os.path.join(FEATURES_DIR, 'train')}"
        )

    records = []
    for fold in (1, 2, 3, 4):
        _, val_files = get_fold_files(train_files, fold)
        if not val_files:
            raise RuntimeError(f"Fold {fold} has no validation files.")

        print(f"\nOOF base predictions — fold {fold}/4 ({len(val_files):,} clips)")
        fold_records = collect_expert_predictions(
            val_files,
            fold=fold,
            final=False,
        )
        for r in fold_records:
            r["fold"] = fold
        records.extend(fold_records)

    np.save(
        OOF_CACHE_PATH,
        {
            "version": OOF_CACHE_VERSION,
            "records": np.asarray(records, dtype=object),
        },
        allow_pickle=True,
    )
    print(f"Saved OOF prediction cache: {OOF_CACHE_PATH}")
    return records


def choose_regularization(x, y, groups):
    unique_groups = np.unique(groups)
    n_splits = min(4, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("Need at least two speakers to fit the fusion classifier.")

    splitter = GroupKFold(n_splits=n_splits)
    best_c = None
    best_loss = float("inf")

    print("\nSelecting fusion regularization with speaker-grouped meta CV...")
    for c in C_GRID:
        fold_losses = []
        for train_idx, val_idx in splitter.split(x, y, groups):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_val = scaler.transform(x[val_idx])

            clf = LogisticRegression(
                C=c,
                solver="lbfgs",
                max_iter=5000,
                random_state=SEED,
            )
            clf.fit(x_train, y[train_idx])
            p = clf.predict_proba(x_val)[:, 1]
            fold_losses.append(
                log_loss(y[val_idx], np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
            )

        mean_loss = float(np.mean(fold_losses))
        print(f"  C={c:<6g} grouped log-loss={mean_loss:.6f}")
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_c = float(c)

    print(f"Selected C={best_c:g} (grouped log-loss={best_loss:.6f})")
    return best_c


def cross_fitted_meta_probabilities(x, y, groups, c):
    unique_groups = np.unique(groups)
    n_splits = min(4, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)

    probs = np.full(len(y), np.nan, dtype=np.float64)

    for train_idx, val_idx in splitter.split(x, y, groups):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_val = scaler.transform(x[val_idx])

        clf = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=5000,
            random_state=SEED,
        )
        clf.fit(x_train, y[train_idx])
        probs[val_idx] = clf.predict_proba(x_val)[:, 1]

    if np.isnan(probs).any():
        raise RuntimeError("Meta-level cross-fitting did not produce every OOF probability.")
    return probs


def fit_fusion_model(records):
    x, y, groups, conditions, _ = records_to_matrix(records)

    best_c = choose_regularization(x, y, groups)
    meta_oof_probs = cross_fitted_meta_probabilities(x, y, groups, best_c)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=5000,
        random_state=SEED,
    )
    clf.fit(x_scaled, y)

    clean_mask = np.asarray([str(c) == "clean" for c in conditions], dtype=bool)
    degraded_mask = ~clean_mask

    def subset_f1(mask):
        if mask.sum() == 0:
            return None
        return float(
            f1_score(
                y[mask],
                (meta_oof_probs[mask] >= DECISION_THRESHOLD).astype(int),
                zero_division=0,
            )
        )

    artifact = {
        "version": VERSION,
        "fusion_name": FUSION_NAME,
        "feature_names": FEATURE_NAMES,
        "best_c": best_c,
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "coef": clf.coef_.reshape(-1).astype(np.float64),
        "intercept": float(clf.intercept_[0]),
        "threshold": DECISION_THRESHOLD,
        "meta_oof_clean_f1": subset_f1(clean_mask),
        "meta_oof_degraded_f1": subset_f1(degraded_mask),
        "n_oof_clips": int(len(y)),
        "n_oof_clean": int(clean_mask.sum()),
        "n_oof_degraded": int(degraded_mask.sum()),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(MODEL_PATH, artifact, allow_pickle=True)

    print("\nFusion model fitted from OOF development predictions only.")
    print(f"  OOF clips      : {len(y):,}")
    print(f"  OOF clean      : {clean_mask.sum():,}")
    print(f"  OOF degraded   : {degraded_mask.sum():,}")
    print(f"  Threshold      : {DECISION_THRESHOLD:.2f} (fixed)")
    if artifact["meta_oof_clean_f1"] is not None:
        print(f"  OOF clean F1   : {artifact['meta_oof_clean_f1']:.4f}")
    if artifact["meta_oof_degraded_f1"] is not None:
        print(f"  OOF degraded F1: {artifact['meta_oof_degraded_f1']:.4f}")
    print(f"  Saved          : {MODEL_PATH}")
    return artifact


def load_fusion_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing fusion model: {MODEL_PATH}\n"
            "Run: python cqcc_fusion.py --fit-only"
        )
    artifact = np.load(MODEL_PATH, allow_pickle=True).item()
    if artifact.get("version") != VERSION:
        raise RuntimeError(
            "Fusion artifact version mismatch. Re-fit with --fit-only."
        )
    return artifact


def fusion_predict_proba(artifact, x):
    mean = np.asarray(artifact["scaler_mean"], dtype=np.float64)
    scale = np.asarray(artifact["scaler_scale"], dtype=np.float64)
    coef = np.asarray(artifact["coef"], dtype=np.float64)
    intercept = float(artifact["intercept"])

    z = (np.asarray(x, dtype=np.float64) - mean) / scale
    logits = z @ coef + intercept
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def calculate_metrics(labels, probs, threshold):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
        "eer": eer,
        "threshold": float(threshold),
        "cm": confusion_matrix(labels, preds),
    }


def evaluate_split(split, artifact):
    files = get_split_files(split)
    if not files:
        raise RuntimeError(
            f"No files found for split '{split}'. Expected under: "
            f"{os.path.join(FEATURES_DIR, split)}"
        )

    print(f"\nFinal expert predictions — {split} ({len(files):,} clips)")
    records = collect_expert_predictions(files, final=True)
    x, y, _, conditions, _ = records_to_matrix(records)
    probs = fusion_predict_proba(artifact, x)

    threshold = DECISION_THRESHOLD

    metrics = calculate_metrics(y, probs, threshold)

    print(f"\n{FUSION_NAME} | {split}")
    print(f"  Threshold : {threshold:.3f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  ROC-AUC   : {metrics['auc']:.4f}")
    print(f"  EER       : {metrics['eer']:.4f}")
    cm = metrics["cm"]
    print(f"  CM        : TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    result = {
        "model": FUSION_NAME,
        "split": split,
        **metrics,
    }

    if split == "test_degraded":
        # Diagnostics only. They never tune thresholds or alter predictions.
        condition_results = {}
        for cond in sorted(set(map(str, conditions))):
            mask = np.asarray([str(c) == cond for c in conditions], dtype=bool)
            if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                continue
            condition_results[cond] = calculate_metrics(
                y[mask], probs[mask], DECISION_THRESHOLD
            )
        result["condition_results"] = condition_results

        if condition_results:
            print("\n  Condition-wise degraded diagnostics")
            for cond, m in condition_results.items():
                print(
                    f"    {cond:<20} F1={m['f1']:.4f} "
                    f"Acc={m['accuracy']:.4f} AUC={m['auc']:.4f}"
                )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "OOF-trained prediction-level late fusion of CNN-GRU Log-Mel and CQCC."
        )
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Fit/save the OOF fusion model, then stop before held-out evaluation.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load an already fitted fusion model and evaluate held-out splits.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute OOF base predictions even if the current cache exists.",
    )
    args = parser.parse_args()

    if args.fit_only and args.eval_only:
        raise ValueError("Use only one of --fit-only or --eval-only.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"Model : {FUSION_NAME}")
    print(f"Version: {VERSION}")
    print("Architecture: Log-Mel -> CNN-GRU -> probability")
    print("              CQCC    -> Aux MLP -> probability")
    print("              OOF logistic late fusion -> final probability")
    print("Decision threshold: 0.50 for both clean and degraded evaluation.")

    if args.eval_only:
        artifact = load_fusion_model()
    else:
        records = collect_oof_predictions(rebuild_cache=args.rebuild_cache)
        artifact = fit_fusion_model(records)

    if args.fit_only:
        return

    results = [
        evaluate_split("test", artifact),
        evaluate_split("test_degraded", artifact),
    ]

    np.save(RESULT_PATH, results, allow_pickle=True)
    print(f"\nSaved CNN-GRU + CQCC fusion results: {RESULT_PATH}")


if __name__ == "__main__":
    set_seed(SEED)
    main()
