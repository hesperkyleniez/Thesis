"""Proposed CNN-GRU + Akıncı–Özbay CQCC + F0 late prediction fusion.

ONE fusion script, containing the original tested implementations of Platt
calibration, aligned-window logistic reference, and causal temporal evidence
correction. All speaker-grouped metadata checks, checkpoint shapes, input
features, and model artifact names match the documented experiment.

--stage baseline-oof : builds CNN/F0 expert OOF from scratch, using the NEW CQCC
                       expert. No discarded CQCC model is trained.
--stage oof          : rebuilds/aligned OOF by replacing CQCC with paper model.
--stage fit          : fits v28-style temporal fusion from development speakers.
--stage evaluate     : evaluates using frozen model and clean/degraded sidecars.
--stage check        : checks sources; never reads held-out data.
--stage self-test    : synthetic feature/model verification.

When the ORIGINAL saved predecessor OOF cache exists, it is reused unchanged
for bit-identical CNN-GRU/F0 historical inputs. Otherwise a new, compatible
OOF reference is generated with the same trained CNN-GRU and AUC-selected F0,
and the paper CQCC scores. Fresh training is NOT guaranteed byte identical.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from improvedCQCC_f0_train import _get_auc_expert

def _compile_internal(name, source):
    """Install one readable internal legacy implementation (no extra files)."""
    import types, sys
    cached=sys.modules.get(name)
    if cached is not None:
        return cached
    mod=types.ModuleType(name)
    mod.__file__=f'<embedded source in {__file__}: {name}>'
    sys.modules[name]=mod
    try:
        exec(compile(source,mod.__file__,'exec'),mod.__dict__)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return mod

_SOURCE_improvedCQCC_cqcc_f0_fusion_v19_calibrated = r'''
# cqcc_f0_fusion.py
# CNN-GRU + CQCC + F0 prediction-level late fusion.
#
# Independent experts:
#   Log-Mel -> CNN-GRU -> window probabilities
#   CQCC    -> CQCC MLP -> window probabilities
#   F0      -> F0 MLP -> window probabilities
# The confidence-aware Elastic-Net fusion classifier is trained only from speaker-grouped OOF predictions.
# One global threshold is selected from cross-fitted development predictions only, then locked for held-out evaluation.

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
from improvedCQCC_robustness import condition_balanced_weights, choose_condition_balanced_threshold
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

# The running script directory is already on sys.path; do not inject historical code paths.
from improvedCQCC_config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, FOLD_ASSIGNMENTS, SEED
from improvedCQCC_train import CNNGRUOnly, set_seed
from improvedCQCC_cqcc_train import CQCCMLP, VERSION as CQCC_VERSION
CQCC_DIM = 40
from improvedCQCC_f0_train import F0_DIM, F0MLP, F0_VERSION


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION = "v19_platt_calibrated_cnn_gru_cqcc_f0"
# Base-expert predictions are unchanged from v16, so the expensive OOF cache can be reused.
OOF_CACHE_VERSION = "v16_robust_all_cnn_gru_cqcc_f0"
FUSION_NAME = "CNN-GRU-CQCC-F0-F"
MODEL_DIR = os.path.join(MODELS_DIR, "Prediction-Fusion", FUSION_NAME)
MODEL_PATH = os.path.join(MODEL_DIR, "fusion_model_v19_platt_calibrated.npy")
RESULT_PATH = os.path.join(RESULTS_DIR, "cnn_gru_cqcc_f0_fusion_v19_platt_calibrated_results.npy")
OOF_CACHE_PATH = os.path.join(MODEL_DIR, "oof_predictions.npy")

# Leakage-safe Platt-calibrated + confidence-aware Elastic-Net meta-fusion.
# Calibration and fusion hyperparameters are fitted only from development speakers.
# Selection is speaker-grouped at the meta level using condition-balanced log-loss.
C_GRID = (0.01, 0.1, 1.0, 10.0)
L1_RATIO_GRID = (0.25, 0.50, 0.75, 1.0)

# A window is treated as high-confidence when its probability is <= 0.20 or >= 0.80.
CONFIDENCE_LOW = 0.20
CONFIDENCE_HIGH = 0.80

# Threshold is selected once from cross-fitted development predictions only.
# The held-out test set never participates in threshold selection.
DECISION_THRESHOLD = 0.50

# Inference batch sizes. These affect speed only, not methodology.
CNN_BATCH_SIZE = 256
CQCC_BATCH_SIZE = 512
F0_BATCH_SIZE = 512


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
    if not results or not all(r.get("training_version") == CQCC_VERSION for r in results):
        raise RuntimeError(
            "CQCC fold results are from an older feature definition. "
            "Run: python cqcc_train.py"
        )
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
    if d.get("training_version") != CQCC_VERSION:
        raise RuntimeError(
            "CQCC final model/stats are from an older feature definition. "
            "Run: python cqcc_retrain.py"
        )
    return (
        np.asarray(d["mean"], dtype=np.float32),
        np.asarray(d["std"], dtype=np.float32),
    )



def load_f0_fold_stats(fold):
    path = os.path.join(MODELS_DIR, "F0", "fold_results.npy")
    results = list(np.load(path, allow_pickle=True))
    if not results or not all(r.get("training_version") == F0_VERSION for r in results):
        raise RuntimeError("F0 fold results are incompatible. Run: python f0_train.py")
    for r in results:
        if int(r["fold"]) == int(fold):
            return np.asarray(r["mean"], np.float32), np.asarray(r["std"], np.float32)
    raise ValueError(f"F0 fold {fold} normalization stats not found.")

def load_f0_final_stats():
    path=os.path.join(MODELS_DIR,"F0","final_norm_stats.npy")
    d=np.load(path,allow_pickle=True).item()
    if d.get("training_version") != F0_VERSION:
        raise RuntimeError("F0 final model/stats are incompatible. Run: python f0_retrain.py")
    return np.asarray(d["mean"],np.float32),np.asarray(d["std"],np.float32)

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

def predict_f0_windows(model, files, mean, std):
    """Window-level F0 auxiliary probabilities."""
    model.eval()
    out = {}

    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=DEVICE)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=DEVICE)

    with torch.inference_mode():
        for path in tqdm(files, desc="F0 predictions", leave=False):
            with np.load(path, allow_pickle=True) as d:
                x = (
                    torch.as_tensor(d["f0"], dtype=torch.float32, device=DEVICE)
                    - mean_t
                ) / std_t
                label = int(d["label"])
                condition = (
                    str(d["condition"].item())
                    if "condition" in d.files
                    else "clean"
                )

            probs = []
            for i in range(0, len(x), F0_BATCH_SIZE):
                logits = model(x[i:i + F0_BATCH_SIZE])
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
    """Independent CNN-GRU, CQCC, and F0 probabilities for late fusion."""
    cnn=CNNGRUOnly(dropout=0.0).to(DEVICE); cnn.load_state_dict(load_state("CNN-GRU",fold=fold,final=final))
    cqcc=CQCCMLP(CQCC_DIM).to(DEVICE); cqcc.load_state_dict(load_state("CQCC",fold=fold,final=final))
    f0=F0MLP(F0_DIM).to(DEVICE); f0.load_state_dict(load_state("F0",fold=fold,final=final))
    if final:
        cm,cs=load_cqcc_final_stats(); fm,fs=load_f0_final_stats()
    else:
        cm,cs=load_cqcc_fold_stats(fold); fm,fs=load_f0_fold_stats(fold)
    a=predict_cnn_windows(cnn,files); b=predict_cqcc_windows(cqcc,files,cm,cs); c=predict_f0_windows(f0,files,fm,fs)
    del cnn,cqcc,f0
    common=sorted(set(a)&set(b)&set(c)); records=[]
    if len(common)!=len(files): raise RuntimeError("Missing paired expert predictions.")
    for path in common:
        pa,pb,pc=a[path]["window_probs"],b[path]["window_probs"],c[path]["window_probs"]
        if not (len(pa)==len(pb)==len(pc)): raise RuntimeError(f"Window-count mismatch: {path}")
        records.append({"path":path,"speaker":a[path]["speaker"],"condition":a[path]["condition"],"label":int(a[path]["label"]),"cnn_probs":pa,"cqcc_probs":pb,"f0_probs":pc})
    return records


# -----------------------------------------------------------------------------
# Prediction-level late-fusion features
# -----------------------------------------------------------------------------
FEATURE_NAMES = [
    # Existing per-expert probability summaries.
    "cnn_mean","cnn_median","cnn_q75","cnn_std","cnn_ai_fraction",
    "cqcc_mean","cqcc_median","cqcc_q75","cqcc_std","cqcc_ai_fraction",
    "f0_mean","f0_median","f0_q75","f0_std","f0_ai_fraction",

    # New confidence summaries. These let the meta-classifier learn when an
    # otherwise weak expert (especially F0) is unusually decisive.
    "cnn_mean_confidence","cnn_mean_entropy","cnn_high_conf_fraction",
    "cqcc_mean_confidence","cqcc_mean_entropy","cqcc_high_conf_fraction",
    "f0_mean_confidence","f0_mean_entropy","f0_high_conf_fraction",

    # Cross-expert complementarity features.
    "cnn_cqcc_disagreement","cnn_f0_disagreement","cqcc_f0_disagreement",
    "three_way_agreement","mean_expert_probability"
]


def _confidence_summary(probs):
    """Confidence descriptors from window probabilities.

    mean_confidence:
        0 when all predictions are 0.5; approaches 1 near 0 or 1.
    mean_entropy:
        Binary entropy normalized to [0, 1]; lower means more confident.
    high_conf_fraction:
        Fraction of windows with p <= 0.20 or p >= 0.80.
    """
    p = np.asarray(probs, dtype=np.float64)
    eps = 1e-8
    clipped = np.clip(p, eps, 1.0 - eps)

    mean_confidence = float(np.mean(2.0 * np.abs(clipped - 0.5)))

    entropy = -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    ) / np.log(2.0)
    mean_entropy = float(np.mean(entropy))

    high_conf_fraction = float(
        np.mean((clipped <= CONFIDENCE_LOW) | (clipped >= CONFIDENCE_HIGH))
    )

    return [mean_confidence, mean_entropy, high_conf_fraction]


def summarize_probabilities(cnn_probs, cqcc_probs, f0_probs):
    arr = [
        np.asarray(x, dtype=np.float64)
        for x in (cnn_probs, cqcc_probs, f0_probs)
    ]

    if any(len(x) == 0 for x in arr) or len({len(x) for x in arr}) != 1:
        raise ValueError("Expert windows must align.")

    feats = []

    # Keep the original five summary features for each expert unchanged.
    for x in arr:
        feats += [
            float(np.mean(x)),
            float(np.median(x)),
            float(np.quantile(x, 0.75)),
            float(np.std(x)),
            float(np.mean(x >= 0.5)),
        ]

    # Add three confidence descriptors for each expert.
    for x in arr:
        feats += _confidence_summary(x)

    a, b, c = arr
    feats += [
        float(np.mean(np.abs(a - b))),
        float(np.mean(np.abs(a - c))),
        float(np.mean(np.abs(b - c))),
        float(np.mean(((a >= 0.5) == (b >= 0.5)) & ((b >= 0.5) == (c >= 0.5)))),
        float(np.mean([a.mean(), b.mean(), c.mean()])),
    ]

    out = np.asarray(feats, dtype=np.float64)
    if len(out) != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Meta-feature count mismatch: got {len(out)}, "
            f"expected {len(FEATURE_NAMES)}."
        )
    return out


def records_to_matrix(records):
    x = np.stack([
        summarize_probabilities(r["cnn_probs"], r["cqcc_probs"], r["f0_probs"])
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


def _make_meta_classifier(c, l1_ratio):
    """Sparse logistic regression used by the late-fusion meta-model.

    On scikit-learn >=1.8, setting l1_ratio directly avoids the deprecated
    explicit penalty='elasticnet' argument while preserving the same behavior
    with the SAGA solver.
    """
    return LogisticRegression(
        C=float(c),
        solver="saga",
        l1_ratio=float(l1_ratio),
        max_iter=10000,
        tol=1e-4,
        random_state=SEED,
    )


def _safe_logit(probs):
    """Convert probabilities to finite logits for Platt calibration."""
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _fit_one_platt(records, prob_key):
    """Fit one leakage-safe Platt scaler from development window scores.

    Each clip receives one condition-balanced weight in total, divided across
    its windows, so longer recordings do not dominate calibration.
    """
    conditions = np.asarray([r["condition"] for r in records], dtype=object)
    clip_weights = condition_balanced_weights(conditions)

    x_parts, y_parts, w_parts = [], [], []
    for r, clip_w in zip(records, clip_weights):
        probs = np.asarray(r[prob_key], dtype=np.float64)
        if probs.size == 0:
            continue
        x_parts.append(_safe_logit(probs))
        y_parts.append(np.full(probs.size, int(r["label"]), dtype=int))
        w_parts.append(np.full(probs.size, float(clip_w) / probs.size, dtype=np.float64))

    x = np.concatenate(x_parts).reshape(-1, 1)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Platt calibration requires both classes.")

    # Very weak L2 regularization approximates standard Platt scaling while
    # remaining numerically stable for large development sets.
    clf = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
        random_state=SEED,
    )
    clf.fit(x, y, sample_weight=w)
    return {
        "coef": float(clf.coef_.reshape(-1)[0]),
        "intercept": float(clf.intercept_[0]),
    }


def fit_platt_calibrators(records):
    """Fit independent Platt scalers for CNN-GRU, CQCC, and F0."""
    return {
        "cnn": _fit_one_platt(records, "cnn_probs"),
        "cqcc": _fit_one_platt(records, "cqcc_probs"),
        "f0": _fit_one_platt(records, "f0_probs"),
    }


def apply_platt(probs, params):
    """Apply a saved Platt mapping to a vector of expert probabilities."""
    logits = _safe_logit(probs)
    return _sigmoid(float(params["coef"]) * logits + float(params["intercept"]))


def summarize_record(record, calibrators):
    """Calibrate each expert first, then compute the unchanged v17 summaries."""
    cnn = apply_platt(record["cnn_probs"], calibrators["cnn"])
    cqcc = apply_platt(record["cqcc_probs"], calibrators["cqcc"])
    f0 = apply_platt(record["f0_probs"], calibrators["f0"])
    return summarize_probabilities(cnn, cqcc, f0)


def records_to_calibrated_matrix(records, calibrators):
    x = np.stack([summarize_record(r, calibrators) for r in records])
    y = np.asarray([r["label"] for r in records], dtype=int)
    groups = np.asarray([r["speaker"] for r in records], dtype=object)
    conditions = np.asarray([r["condition"] for r in records], dtype=object)
    paths = np.asarray([r["path"] for r in records], dtype=object)
    return x, y, groups, conditions, paths


def build_calibrated_meta_folds(records):
    """Prepare speaker-grouped outer folds with fold-local Platt calibration.

    For each outer validation fold, all three Platt scalers are fitted only on
    that fold's training speakers. The held-out speakers therefore never
    participate in calibration, scaling, or meta-classifier fitting.
    """
    y = np.asarray([r["label"] for r in records], dtype=int)
    groups = np.asarray([r["speaker"] for r in records], dtype=object)
    conditions = np.asarray([r["condition"] for r in records], dtype=object)

    n_splits = min(4, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Need at least two speakers to fit the fusion classifier.")

    splitter = GroupKFold(n_splits=n_splits)
    payloads = []

    print("\nBuilding leakage-safe Platt-calibrated meta folds...")
    for fold_no, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups), 1):
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        calibrators = fit_platt_calibrators(train_records)
        x_train, _, _, train_conditions, _ = records_to_calibrated_matrix(
            train_records, calibrators
        )
        x_val, _, _, val_conditions, _ = records_to_calibrated_matrix(
            val_records, calibrators
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        payloads.append({
            "fold": fold_no,
            "train_idx": np.asarray(train_idx, dtype=int),
            "val_idx": np.asarray(val_idx, dtype=int),
            "x_train": x_train,
            "x_val": x_val,
            "y_train": y[train_idx],
            "y_val": y[val_idx],
            "train_w": condition_balanced_weights(train_conditions),
            "val_w": condition_balanced_weights(val_conditions),
            "calibrators": calibrators,
        })

        print(
            f"  Fold {fold_no}: train={len(train_idx):,} val={len(val_idx):,} | "
            f"Platt slopes CNN={calibrators['cnn']['coef']:.3f} "
            f"CQCC={calibrators['cqcc']['coef']:.3f} "
            f"F0={calibrators['f0']['coef']:.3f}"
        )

    return payloads, y, groups, conditions


def choose_regularization(payloads):
    best_c = None
    best_l1_ratio = None
    best_loss = float("inf")

    print("\nSelecting calibrated Elastic-Net fusion regularization...")
    for c in C_GRID:
        for l1_ratio in L1_RATIO_GRID:
            fold_losses = []

            for pld in payloads:
                clf = _make_meta_classifier(c, l1_ratio)
                clf.fit(
                    pld["x_train"],
                    pld["y_train"],
                    sample_weight=pld["train_w"],
                )
                p = clf.predict_proba(pld["x_val"])[:, 1]
                fold_losses.append(
                    log_loss(
                        pld["y_val"],
                        np.clip(p, 1e-6, 1 - 1e-6),
                        labels=[0, 1],
                        sample_weight=pld["val_w"],
                    )
                )

            mean_loss = float(np.mean(fold_losses))
            print(
                f"  C={c:<5g} l1_ratio={l1_ratio:.2f} "
                f"grouped log-loss={mean_loss:.6f}"
            )

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_c = float(c)
                best_l1_ratio = float(l1_ratio)

    print(
        f"Selected C={best_c:g}, l1_ratio={best_l1_ratio:.2f} "
        f"(grouped log-loss={best_loss:.6f})"
    )
    return best_c, best_l1_ratio, best_loss


def cross_fitted_meta_probabilities(payloads, n_records, c, l1_ratio):
    probs = np.full(n_records, np.nan, dtype=np.float64)

    for pld in payloads:
        clf = _make_meta_classifier(c, l1_ratio)
        clf.fit(
            pld["x_train"],
            pld["y_train"],
            sample_weight=pld["train_w"],
        )
        probs[pld["val_idx"]] = clf.predict_proba(pld["x_val"])[:, 1]

    if np.isnan(probs).any():
        raise RuntimeError(
            "Meta-level cross-fitting did not produce every OOF probability."
        )
    return probs


def fit_fusion_model(records):
    payloads, y, groups, conditions = build_calibrated_meta_folds(records)

    best_c, best_l1_ratio, best_loss = choose_regularization(payloads)
    meta_oof_probs = cross_fitted_meta_probabilities(
        payloads, len(records), best_c, best_l1_ratio
    )
    tuned_threshold, threshold_macro_f1, threshold_overall_f1 = choose_condition_balanced_threshold(
        y, meta_oof_probs, conditions
    )

    # Final deployment pipeline: calibration, scaling, and meta-classifier are
    # now fitted on all development OOF expert predictions only.
    final_calibrators = fit_platt_calibrators(records)
    x, _, _, _, _ = records_to_calibrated_matrix(records, final_calibrators)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = _make_meta_classifier(best_c, best_l1_ratio)
    fusion_weights = condition_balanced_weights(conditions)
    clf.fit(x_scaled, y, sample_weight=fusion_weights)

    clean_mask = np.asarray([str(c) == "clean" for c in conditions], dtype=bool)
    degraded_mask = ~clean_mask

    def subset_f1(mask):
        if mask.sum() == 0:
            return None
        return float(
            f1_score(
                y[mask],
                (meta_oof_probs[mask] >= tuned_threshold).astype(int),
                zero_division=0,
            )
        )

    artifact = {
        "version": VERSION,
        "fusion_name": FUSION_NAME,
        "feature_names": FEATURE_NAMES,
        "best_c": best_c,
        "best_l1_ratio": best_l1_ratio,
        "grouped_log_loss": best_loss,
        "platt_calibrators": final_calibrators,
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "coef": clf.coef_.reshape(-1).astype(np.float64),
        "intercept": float(clf.intercept_[0]),
        "threshold": tuned_threshold,
        "threshold_macro_condition_f1": threshold_macro_f1,
        "threshold_overall_f1": threshold_overall_f1,
        "meta_oof_clean_f1": subset_f1(clean_mask),
        "meta_oof_degraded_f1": subset_f1(degraded_mask),
        "n_oof_clips": int(len(y)),
        "n_oof_clean": int(clean_mask.sum()),
        "n_oof_degraded": int(degraded_mask.sum()),
    }

    coef = artifact["coef"]
    nonzero = np.abs(coef) > 1e-8
    artifact["n_nonzero_features"] = int(np.sum(nonzero))

    print("\nFinal Platt calibration diagnostics")
    for expert in ("cnn", "cqcc", "f0"):
        prm = final_calibrators[expert]
        print(
            f"  {expert.upper():<5} slope={prm['coef']: .5f} "
            f"intercept={prm['intercept']: .5f}"
        )

    ranked = np.argsort(np.abs(coef))[::-1]
    print("\nCalibrated sparse meta-feature diagnostics")
    print(
        f"  Non-zero coefficients: {artifact['n_nonzero_features']}"
        f"/{len(FEATURE_NAMES)}"
    )
    for idx in ranked[:12]:
        print(f"  {FEATURE_NAMES[idx]:<32} coef={coef[idx]: .5f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(MODEL_PATH, artifact, allow_pickle=True)

    print("\nFusion model fitted from OOF development predictions only.")
    print(f"  OOF clips      : {len(y):,}")
    print(f"  Elastic-Net C  : {best_c:g}")
    print(f"  L1 ratio       : {best_l1_ratio:.2f}")
    print(f"  OOF clean      : {clean_mask.sum():,}")
    print(f"  OOF degraded   : {degraded_mask.sum():,}")
    print(f"  OOF threshold  : {tuned_threshold:.3f} (condition-macro F1 selected)")
    print(f"  OOF macro F1   : {threshold_macro_f1:.4f}")
    print(f"  OOF overall F1 : {threshold_overall_f1:.4f}")
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
            "Run: python cqcc_f0_fusion_v19_calibrated.py --fit-only"
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
    calibrators = artifact["platt_calibrators"]
    x, y, _, conditions, _ = records_to_calibrated_matrix(records, calibrators)
    probs = fusion_predict_proba(artifact, x)

    threshold = float(artifact.get("threshold", DECISION_THRESHOLD))
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
        "version": VERSION,
        "split": split,
        **metrics,
    }

    if split == "test_degraded":
        condition_results = {}
        for cond in sorted(set(map(str, conditions))):
            mask = np.asarray([str(c) == cond for c in conditions], dtype=bool)
            if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                continue
            condition_results[cond] = calculate_metrics(
                y[mask], probs[mask], threshold
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
            "Leakage-safe Platt-calibrated OOF late fusion of CNN-GRU, CQCC, and F0."
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
    print("              F0      -> Aux MLP -> probability")
    print("              Fold-local Platt calibration -> confidence-aware Elastic-Net late fusion")
    print("Decision threshold: selected from cross-fitted development predictions only.")

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
    print(f"\nSaved CNN-GRU + CQCC + F0 fusion results: {RESULT_PATH}")


if __name__ == "__main__":
    set_seed(SEED)
    main()
'''

_SOURCE_improvedCQCC_cqcc_f0_fusion_v22_window_level = r'''
# cqcc_f0_fusion_v22_window_level.py
#
# v22: aligned-window prediction-level late fusion.
#
# Why this version:
#   Earlier fusion versions summarized each expert over a whole clip first.
#   That can discard useful *aligned 1-second* complementarity. This version
#   fuses CNN-GRU, CQCC, and F0 probabilities at each aligned window, then
#   aggregates the fused window probabilities to one clip probability.
#
# Methodology remains prediction-level late fusion:
#   Log-Mel -> CNN-GRU -> window probability ----\
#   CQCC    -> CQCC MLP -> window probability ----> small logistic score fusion
#   F0      -> F0 MLP -> window probability ------/        |
#                                                            v
#                                                    clip aggregation
#
# Important safeguards:
# - Base experts/features/augmentation/folds are unchanged.
# - Fold-local Platt calibration is retained.
# - Speaker-grouped OOF development predictions select fusion hyperparameters.
# - Every clip receives equal total training weight regardless of window count.
# - Held-out test data is not used to fit calibration, fusion weights, or threshold.

import os
import argparse
import numpy as np
import torch

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

import improvedCQCC_cqcc_f0_fusion_v19_calibrated as base


VERSION = "v22_aligned_window_level_cnn_gru_cqcc_f0"
FUSION_NAME = base.FUSION_NAME
MODEL_DIR = base.MODEL_DIR
MODEL_PATH = os.path.join(
    MODEL_DIR, "fusion_model_v22_aligned_window_level.npy"
)
RESULT_PATH = os.path.join(
    base.RESULTS_DIR,
    "cnn_gru_cqcc_f0_fusion_v22_aligned_window_level_results.npy",
)

C_GRID = (0.01, 0.1, 1.0, 10.0)
AGGREGATION_GRID = (
    "mean_probability",
    "mean_logit",
    "median_probability",
)
EPS = 1e-6

WINDOW_FEATURE_NAMES = [
    "cnn_logit",
    "cqcc_logit",
    "f0_logit",
    "cnn_cqcc_probability_gap",
    "cnn_f0_probability_gap",
    "cqcc_f0_probability_gap",
    "aux_minus_cnn",
    "cnn_confidence",
    "aux_agreement",
]


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def calibrated_window_probabilities(record, calibrators):
    cnn = base.apply_platt(record["cnn_probs"], calibrators["cnn"])
    cqcc = base.apply_platt(record["cqcc_probs"], calibrators["cqcc"])
    f0 = base.apply_platt(record["f0_probs"], calibrators["f0"])

    cnn = np.asarray(cnn, dtype=np.float64)
    cqcc = np.asarray(cqcc, dtype=np.float64)
    f0 = np.asarray(f0, dtype=np.float64)

    if not (len(cnn) == len(cqcc) == len(f0)):
        raise RuntimeError(
            f"Aligned-window count mismatch for {record.get('path', '<unknown>')}"
        )
    if len(cnn) == 0:
        raise RuntimeError(
            f"No windows for {record.get('path', '<unknown>')}"
        )
    return cnn, cqcc, f0


def window_features(record, calibrators):
    """Small, interpretable aligned-window late-fusion representation."""
    cnn, cqcc, f0 = calibrated_window_probabilities(record, calibrators)

    aux_mean = 0.5 * (cqcc + f0)

    x = np.column_stack(
        [
            safe_logit(cnn),
            safe_logit(cqcc),
            safe_logit(f0),
            np.abs(cnn - cqcc),
            np.abs(cnn - f0),
            np.abs(cqcc - f0),
            aux_mean - cnn,
            2.0 * np.abs(cnn - 0.5),
            1.0 - np.abs(cqcc - f0),
        ]
    )
    return np.asarray(x, dtype=np.float64)


def make_window_training_matrix(records, calibrators):
    """Concatenate windows, but give every clip equal total loss weight."""
    conditions = np.asarray([r["condition"] for r in records], dtype=object)
    clip_weights = base.condition_balanced_weights(conditions)

    xs, ys, ws = [], [], []
    for r, clip_w in zip(records, clip_weights):
        x = window_features(r, calibrators)
        n = len(x)
        xs.append(x)
        ys.append(np.full(n, int(r["label"]), dtype=int))
        ws.append(
            np.full(
                n,
                float(clip_w) / float(n),
                dtype=np.float64,
            )
        )

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(ws, axis=0),
    )


def make_classifier(c):
    # sklearn >=1.8: l1_ratio=0 expresses ordinary L2 regularization without
    # explicitly using the deprecated penalty argument.
    return LogisticRegression(
        C=float(c),
        solver="lbfgs",
        l1_ratio=0.0,
        max_iter=5000,
        tol=1e-6,
        random_state=base.SEED,
    )


def aggregate_window_probabilities(window_probs, mode):
    p = np.asarray(window_probs, dtype=np.float64)
    if p.size == 0:
        raise RuntimeError("Cannot aggregate an empty window probability vector.")

    if mode == "mean_probability":
        return float(np.mean(p))

    if mode == "mean_logit":
        return float(sigmoid(np.mean(safe_logit(p))))

    if mode == "median_probability":
        return float(np.median(p))

    raise ValueError(f"Unknown aggregation mode: {mode}")


def predict_clip_probability(record, calibrators, scaler, clf, aggregation):
    x = window_features(record, calibrators)
    x = scaler.transform(x)
    window_probs = clf.predict_proba(x)[:, 1]
    return aggregate_window_probabilities(window_probs, aggregation)


def predict_records(records, calibrators, scaler, clf, aggregation):
    return np.asarray(
        [
            predict_clip_probability(r, calibrators, scaler, clf, aggregation)
            for r in records
        ],
        dtype=np.float64,
    )


def build_oof_payloads(records):
    y = np.asarray([r["label"] for r in records], dtype=int)
    groups = np.asarray([r["speaker"] for r in records], dtype=object)
    conditions = np.asarray([r["condition"] for r in records], dtype=object)

    splitter = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    payloads = []

    print("\nBuilding leakage-safe aligned-window meta folds...")

    for fold_no, (train_idx, val_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y, groups), 1
    ):
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        calibrators = base.fit_platt_calibrators(train_records)

        xw, yw, ww = make_window_training_matrix(
            train_records, calibrators
        )

        scaler = StandardScaler()
        # Modern sklearn supports sample_weight here. If unavailable, fall back
        # to ordinary scaling; classifier training is still clip-balanced.
        try:
            scaler.fit(xw, sample_weight=ww)
        except TypeError:
            scaler.fit(xw)
        xw = scaler.transform(xw)

        payloads.append(
            {
                "fold": fold_no,
                "train_idx": np.asarray(train_idx, dtype=int),
                "val_idx": np.asarray(val_idx, dtype=int),
                "train_x": xw,
                "train_y": yw,
                "train_w": ww,
                "val_records": val_records,
                "val_y": y[val_idx],
                "val_conditions": conditions[val_idx],
                "calibrators": calibrators,
                "scaler": scaler,
            }
        )

        n_val_windows = sum(
            len(np.asarray(r["cnn_probs"])) for r in val_records
        )
        print(
            f"  Fold {fold_no}: train clips={len(train_idx):,} "
            f"val clips={len(val_idx):,} | "
            f"train windows={len(yw):,} val windows={n_val_windows:,} | "
            f"Platt slopes CNN={calibrators['cnn']['coef']:.3f} "
            f"CQCC={calibrators['cqcc']['coef']:.3f} "
            f"F0={calibrators['f0']['coef']:.3f}"
        )

    return payloads, y, groups, conditions


def choose_configuration(payloads, n_records, y, conditions):
    best = None

    print("\nSelecting window-level fusion configuration...")
    print("Selection metric: speaker-fold mean condition-balanced log-loss.")

    for c in C_GRID:
        # Train only once per fold/C. The same window predictions can be
        # aggregated in several ways without refitting.
        mode_probs = {
            mode: np.full(n_records, np.nan, dtype=np.float64)
            for mode in AGGREGATION_GRID
        }
        fold_mode_losses = {
            mode: [] for mode in AGGREGATION_GRID
        }

        for pld in payloads:
            clf = make_classifier(c)
            clf.fit(
                pld["train_x"],
                pld["train_y"],
                sample_weight=pld["train_w"],
            )

            # Predict aligned windows once; derive all clip aggregations.
            per_clip_window_probs = []
            for r in pld["val_records"]:
                x = window_features(r, pld["calibrators"])
                x = pld["scaler"].transform(x)
                per_clip_window_probs.append(
                    clf.predict_proba(x)[:, 1]
                )

            val_weights = base.condition_balanced_weights(
                pld["val_conditions"]
            )

            for mode in AGGREGATION_GRID:
                clip_p = np.asarray(
                    [
                        aggregate_window_probabilities(wp, mode)
                        for wp in per_clip_window_probs
                    ],
                    dtype=np.float64,
                )
                mode_probs[mode][pld["val_idx"]] = clip_p

                fold_mode_losses[mode].append(
                    log_loss(
                        pld["val_y"],
                        np.clip(clip_p, EPS, 1.0 - EPS),
                        labels=[0, 1],
                        sample_weight=val_weights,
                    )
                )

        for mode in AGGREGATION_GRID:
            probs = mode_probs[mode]
            if np.isnan(probs).any():
                raise RuntimeError(
                    f"Incomplete v22 OOF predictions for C={c}, mode={mode}"
                )

            losses = np.asarray(fold_mode_losses[mode], dtype=float)
            mean_loss = float(np.mean(losses))
            std_loss = float(np.std(losses))

            threshold, macro_f1, overall_f1 = (
                base.choose_condition_balanced_threshold(
                    y, probs, conditions
                )
            )

            print(
                f"  C={c:<5g} agg={mode:<18} "
                f"logloss={mean_loss:.6f} +/- {std_loss:.6f} "
                f"macroF1={macro_f1:.4f} overallF1={overall_f1:.4f}"
            )

            candidate = {
                "c": float(c),
                "aggregation": mode,
                "mean_log_loss": mean_loss,
                "std_log_loss": std_loss,
                "threshold": float(threshold),
                "macro_f1": float(macro_f1),
                "overall_f1": float(overall_f1),
                "oof_probs": probs,
            }

            if (
                best is None
                or candidate["mean_log_loss"] < best["mean_log_loss"] - 1e-12
                or (
                    abs(
                        candidate["mean_log_loss"]
                        - best["mean_log_loss"]
                    ) <= 1e-12
                    and candidate["std_log_loss"]
                    < best["std_log_loss"]
                )
            ):
                best = candidate

    print("\nSelected v22 window fusion")
    print(f"  C             : {best['c']:g}")
    print(f"  Aggregation   : {best['aggregation']}")
    print(
        f"  Grouped loss  : {best['mean_log_loss']:.6f} "
        f"+/- {best['std_log_loss']:.6f}"
    )
    print(f"  OOF threshold : {best['threshold']:.3f}")
    print(f"  OOF macro F1  : {best['macro_f1']:.4f}")
    print(f"  OOF overall F1: {best['overall_f1']:.4f}")

    return best


def fit_fusion_model(records):
    payloads, y, groups, conditions = build_oof_payloads(records)
    best = choose_configuration(
        payloads, len(records), y, conditions
    )

    oof_probs = best["oof_probs"]
    threshold = best["threshold"]

    final_calibrators = base.fit_platt_calibrators(records)
    xw, yw, ww = make_window_training_matrix(
        records, final_calibrators
    )

    scaler = StandardScaler()
    try:
        scaler.fit(xw, sample_weight=ww)
    except TypeError:
        scaler.fit(xw)
    xw_s = scaler.transform(xw)

    clf = make_classifier(best["c"])
    clf.fit(xw_s, yw, sample_weight=ww)

    clean_mask = np.asarray(
        [str(c) == "clean" for c in conditions],
        dtype=bool,
    )
    degraded_mask = ~clean_mask

    def subset_f1(mask):
        if mask.sum() == 0:
            return None
        return float(
            f1_score(
                y[mask],
                (oof_probs[mask] >= threshold).astype(int),
                zero_division=0,
            )
        )

    coef = clf.coef_.reshape(-1).astype(np.float64)
    ranked = np.argsort(np.abs(coef))[::-1]

    artifact = {
        "version": VERSION,
        "fusion_name": FUSION_NAME,
        "window_feature_names": list(WINDOW_FEATURE_NAMES),
        "best_c": float(best["c"]),
        "aggregation": best["aggregation"],
        "grouped_log_loss": float(best["mean_log_loss"]),
        "grouped_log_loss_std": float(best["std_log_loss"]),
        "platt_calibrators": final_calibrators,
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "coef": coef,
        "intercept": float(clf.intercept_[0]),
        "threshold": float(threshold),
        "oof_macro_f1": float(best["macro_f1"]),
        "oof_overall_f1": float(best["overall_f1"]),
        "oof_clean_f1": subset_f1(clean_mask),
        "oof_degraded_f1": subset_f1(degraded_mask),
        "n_oof_clips": int(len(records)),
        "n_training_windows": int(len(yw)),
    }

    print("\nFinal v22 window-level coefficients")
    for idx in ranked:
        print(
            f"  {WINDOW_FEATURE_NAMES[idx]:<32} "
            f"coef={coef[idx]: .5f}"
        )
    print(f"  intercept{'':<23} = {artifact['intercept']: .5f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(MODEL_PATH, artifact, allow_pickle=True)

    print("\nFusion model fitted from OOF development predictions only.")
    print(f"  OOF clips       : {len(records):,}")
    print(f"  Training windows: {len(yw):,}")
    print(f"  C               : {artifact['best_c']:g}")
    print(f"  Aggregation     : {artifact['aggregation']}")
    print(f"  OOF threshold   : {artifact['threshold']:.3f}")
    print(f"  OOF macro F1    : {artifact['oof_macro_f1']:.4f}")
    print(f"  OOF overall F1  : {artifact['oof_overall_f1']:.4f}")
    if artifact["oof_clean_f1"] is not None:
        print(f"  OOF clean F1    : {artifact['oof_clean_f1']:.4f}")
    if artifact["oof_degraded_f1"] is not None:
        print(f"  OOF degraded F1 : {artifact['oof_degraded_f1']:.4f}")
    print(f"  Saved           : {MODEL_PATH}")
    return artifact


def load_fusion_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing v22 fusion model: {MODEL_PATH}\n"
            "Run this script normally or with --fit-only first."
        )
    artifact = np.load(
        MODEL_PATH, allow_pickle=True
    ).item()
    if artifact.get("version") != VERSION:
        raise RuntimeError("v22 fusion artifact version mismatch.")
    return artifact


def artifact_scaler(artifact):
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(
        artifact["scaler_mean"], dtype=np.float64
    )
    scaler.scale_ = np.asarray(
        artifact["scaler_scale"], dtype=np.float64
    )
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def artifact_classifier(artifact):
    clf = make_classifier(artifact["best_c"])
    clf.classes_ = np.asarray([0, 1], dtype=int)
    clf.coef_ = np.asarray(
        artifact["coef"], dtype=np.float64
    ).reshape(1, -1)
    clf.intercept_ = np.asarray(
        [artifact["intercept"]], dtype=np.float64
    )
    clf.n_features_in_ = clf.coef_.shape[1]
    clf.n_iter_ = np.asarray([1], dtype=int)
    return clf


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
        "precision": float(
            precision_score(labels, preds, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, preds, zero_division=0)
        ),
        "auc": float(roc_auc_score(labels, probs)),
        "eer": eer,
        "threshold": float(threshold),
        "cm": confusion_matrix(labels, preds),
    }


def evaluate_split(split, artifact):
    files = base.get_split_files(split)
    if not files:
        raise RuntimeError(
            f"No files found for split '{split}'."
        )

    print(
        f"\nFinal expert predictions — {split} "
        f"({len(files):,} clips)"
    )

    records = base.collect_expert_predictions(
        files, final=True
    )

    scaler = artifact_scaler(artifact)
    clf = artifact_classifier(artifact)
    probs = predict_records(
        records,
        artifact["platt_calibrators"],
        scaler,
        clf,
        artifact["aggregation"],
    )

    y = np.asarray(
        [r["label"] for r in records], dtype=int
    )
    conditions = np.asarray(
        [r["condition"] for r in records],
        dtype=object,
    )

    threshold = float(artifact["threshold"])
    metrics = calculate_metrics(y, probs, threshold)

    print(f"\n{FUSION_NAME} v22 | {split}")
    print(f"  Aggregation: {artifact['aggregation']}")
    print(f"  Threshold  : {threshold:.3f}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  F1-score   : {metrics['f1']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  ROC-AUC    : {metrics['auc']:.4f}")
    print(f"  EER        : {metrics['eer']:.4f}")
    cm = metrics["cm"]
    print(
        f"  CM         : TN={cm[0,0]} FP={cm[0,1]} "
        f"FN={cm[1,0]} TP={cm[1,1]}"
    )

    result = {
        "model": FUSION_NAME,
        "version": VERSION,
        "split": split,
        **metrics,
    }

    if split == "test_degraded":
        condition_results = {}
        print("\n  Condition-wise degraded diagnostics")
        for cond in sorted(set(map(str, conditions))):
            mask = np.asarray(
                [str(c) == cond for c in conditions],
                dtype=bool,
            )
            if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                continue
            m = calculate_metrics(
                y[mask], probs[mask], threshold
            )
            condition_results[cond] = m
            print(
                f"    {cond:<20} F1={m['f1']:.4f} "
                f"Acc={m['accuracy']:.4f} "
                f"AUC={m['auc']:.4f}"
            )
        result["condition_results"] = condition_results

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "v22 aligned-window CNN-GRU + CQCC + F0 "
            "prediction-level late fusion."
        )
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help=(
            "Fit/select v22 using OOF development predictions only "
            "and stop before held-out evaluation."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate an already fitted v22 artifact.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help=(
            "Recompute base OOF expert predictions. "
            "Normally DO NOT use this."
        ),
    )
    args = parser.parse_args()

    if args.fit_only and args.eval_only:
        raise ValueError(
            "Use only one of --fit-only or --eval-only."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(base.RESULTS_DIR, exist_ok=True)

    print(f"Device: {base.DEVICE}")
    if base.DEVICE.type == "cuda":
        print(
            f"GPU   : {torch.cuda.get_device_name(0)}"
        )
    print(f"Model : {FUSION_NAME}")
    print(f"Version: {VERSION}")
    print(
        "Architecture: CNN-GRU/CQCC/F0 aligned window probabilities "
        "-> fold-local Platt calibration"
    )
    print(
        "              -> small window-level logistic score fusion "
        "-> clip probability aggregation"
    )
    print(
        "Each clip has equal total meta-training weight regardless "
        "of its number of windows."
    )

    if args.eval_only:
        artifact = load_fusion_model()
    else:
        records = base.collect_oof_predictions(
            rebuild_cache=args.rebuild_cache
        )
        artifact = fit_fusion_model(records)

    if args.fit_only:
        return

    results = [
        evaluate_split("test", artifact),
        evaluate_split("test_degraded", artifact),
    ]

    np.save(RESULT_PATH, results, allow_pickle=True)
    print(f"\nSaved v22 results: {RESULT_PATH}")


if __name__ == "__main__":
    base.set_seed(base.SEED)
    main()
'''

_SOURCE_improvedCQCC_v25_fusion_window = r'''
"""Use v25 AUC-selected CQCC/F0 experts with the unchanged v22 window fusion.

Prerequisites in E:\\Thesis\\code:
    cqcc_train.py, f0_train.py, cqcc_f0_fusion_v19_calibrated.py,
    cqcc_f0_fusion_v22_window_level.py, v25_aux_train_auc.py

First train v25 experts:
    python v25_aux_train_auc.py --expert all --stage cv
    python v25_aux_train_auc.py --expert all --stage final

Then fit v22 fusion with NEW v25 expert predictions:
    python v25_fusion_window.py --stage fit

Only after analysis/freeze (test sets have already been inspected in prior runs):
    python v25_fusion_window.py --stage evaluate

No original CQCC, F0, CNN-GRU, or v22 checkpoints are overwritten.
"""

import argparse
import os

import numpy as np
import torch

import improvedCQCC_cqcc_f0_fusion_v19_calibrated as base
import improvedCQCC_cqcc_f0_fusion_v22_window_level as window
from improvedCQCC_config import MODELS_DIR, RESULTS_DIR
from improvedCQCC_v25_aux_train_auc import VERSION as AUX_VERSION, EXPERTS

VERSION = 'v25_auc_aux_v22_aligned_window_fusion'
FUSION_DIRECTORY = os.path.join(MODELS_DIR, 'Prediction-Fusion', 'CNN-GRU-CQCC-F0-F-v25-AUC')
MODEL_PATH = os.path.join(FUSION_DIRECTORY, 'fusion_model_v25_auc_window.npy')
OOF_PATH = os.path.join(FUSION_DIRECTORY, 'oof_predictions_v25.npy')
RESULT_PATH = os.path.join(RESULTS_DIR, 'cnn_gru_cqcc_f0_v25_auc_window_results.npy')


def aux_dir(expert):
    return os.path.join(MODELS_DIR, EXPERTS[expert]['folder'])


def _fold_info(expert, fold):
    path = os.path.join(aux_dir(expert), 'fold_results.npy')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing v25 CV results: {path}')
    items = list(np.load(path, allow_pickle=True))
    if len(items) != 4 or any(r.get('training_version') != AUX_VERSION or r.get('expert') != expert for r in items):
        raise RuntimeError(f'Expected four matching v25 {expert} CV results: {path}')
    matches = [r for r in items if int(r['fold']) == int(fold)]
    if len(matches) != 1:
        raise RuntimeError(f'{expert}: invalid fold {fold} results')
    result = matches[0]
    if np.asarray(result['mean']).size != EXPERTS[expert]['dim']:
        raise RuntimeError(f'{expert}: fold {fold} normalizer feature dimension mismatch')
    return result


def _final_info(expert):
    path = os.path.join(aux_dir(expert), 'final_norm_stats.npy')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing v25 final normalization: {path}')
    stats = np.load(path, allow_pickle=True).item()
    if stats.get('training_version') != AUX_VERSION or stats.get('feature') != expert:
        raise RuntimeError(f'{expert}: final normalization from incompatible training version')
    if np.asarray(stats['mean']).size != EXPERTS[expert]['dim']:
        raise RuntimeError(f'{expert}: final normalizer dimension mismatch')
    return stats


def _check_checkpoints(stage):
    for expert in ('cqcc','f0'):
        if stage == 'fit':
            for fold in range(1,5):
                _fold_info(expert, fold)
                ckpt = os.path.join(aux_dir(expert), f'best_fold{fold}.pt')
                if not os.path.isfile(ckpt):
                    raise FileNotFoundError(f'Missing v25 fold checkpoint: {ckpt}')
        else:
            _final_info(expert)
            ckpt = os.path.join(aux_dir(expert), 'final_model.pt')
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f'Missing v25 final checkpoint: {ckpt}')


def patch_fusion_sources():
    """Override ONLY auxiliary checkpoint locations and fresh cache destinations.

    CNN-GRU loader and all feature/fusion calculation functions remain v19/v22.
    """
    original_loader = base.load_state

    def v25_load_state(model_name, fold=None, final=False):
        if model_name == 'CNN-GRU':
            return original_loader(model_name, fold=fold, final=final)
        if model_name not in ('CQCC', 'F0'):
            raise RuntimeError(f'Unexpected model requested: {model_name}')
        expert = model_name.lower()
        path = os.path.join(aux_dir(expert), 'final_model.pt' if final else f'best_fold{fold}.pt')
        if final:
            _final_info(expert)
        else:
            _fold_info(expert, fold)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return torch.load(path, map_location=base.DEVICE, weights_only=True)

    def fold_norm(expert, fold):
        item = _fold_info(expert, fold)
        return np.asarray(item['mean'], dtype=np.float32), np.asarray(item['std'], dtype=np.float32)

    def final_norm(expert):
        item = _final_info(expert)
        return np.asarray(item['mean'], dtype=np.float32), np.asarray(item['std'], dtype=np.float32)

    base.load_state = v25_load_state
    base.load_cqcc_fold_stats = lambda fold: fold_norm('cqcc', fold)
    base.load_cqcc_final_stats = lambda: final_norm('cqcc')
    base.load_f0_fold_stats = lambda fold: fold_norm('f0', fold)
    base.load_f0_final_stats = lambda: final_norm('f0')

    # These assignments deliberately isolate the v25 OOF cache + fusion artifact.
    base.MODEL_DIR = FUSION_DIRECTORY
    base.OOF_CACHE_PATH = OOF_PATH
    base.OOF_CACHE_VERSION = VERSION
    base.VERSION = VERSION
    window.VERSION = VERSION
    window.MODEL_DIR = FUSION_DIRECTORY
    window.MODEL_PATH = MODEL_PATH
    window.RESULT_PATH = RESULT_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['fit','evaluate','all'], default='fit',
                        help='Default FIT ONLY. Held-out evaluation requires an explicit command.')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Recompute v25 OOF scores using v25 experts (not original v16 cache).')
    args = parser.parse_args()

    patch_fusion_sources()
    os.makedirs(FUSION_DIRECTORY, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('v25: same v22 aligned-window logistic fusion; new AUC-selected CQCC/F0 experts')
    print('CNN-GRU and cached audio features remain unchanged')
    print(f'v25 OOF cache: {OOF_PATH}')
    if args.stage in ('fit','all'):
        _check_checkpoints('fit')
        rebuild = args.rebuild_cache
        if os.path.isfile(OOF_PATH):
            newest_checkpoint = max(
                os.path.getmtime(os.path.join(aux_dir(expert), f'best_fold{fold}.pt'))
                for expert in ('cqcc', 'f0') for fold in range(1, 5)
            )
            if newest_checkpoint > os.path.getmtime(OOF_PATH):
                print('v25 expert checkpoints newer than OOF cache: rebuilding new expert predictions.')
                rebuild = True
        records = base.collect_oof_predictions(rebuild_cache=rebuild)
        print(f'Loaded {len(records):,} development OOF clips')
        window.fit_fusion_model(records)
    if args.stage in ('evaluate','all'):
        _check_checkpoints('evaluate')
        artifact = window.load_fusion_model()
        print('NOTE: these held-out sets have been inspected in earlier experiments.')
        results = [window.evaluate_split('test', artifact),
                   window.evaluate_split('test_degraded', artifact)]
        np.save(RESULT_PATH, results, allow_pickle=True)
        print(f'Saved: {RESULT_PATH}')


if __name__ == '__main__':
    base.set_seed(base.SEED)
    main()
'''

_SOURCE_improvedCQCC_v27_clip_optimized_fusion = r'''
"""v27: clip-optimized aligned-window CNN-GRU/CQCC/F0 prediction-level fusion.

This is a CONTROLLED experiment against v25. It deliberately changes ONLY
how the fusion classifier is trained: window features are averaged within each
clip BEFORE the loss is computed. The nine v22 features, Platt calibration,
mean-logit inference, acoustic experts, augmentation, and all original feature
caches remain unchanged.

Prerequisites alongside this script in E:\\Thesis\\code:
    cqcc_f0_fusion_v19_calibrated.py
    cqcc_f0_fusion_v22_window_level.py
    v25_aux_train_auc.py
    v25_fusion_window.py
    config.py, train.py, cqcc_train.py, f0_train.py, robustness.py

First run the existing v25 OOF generation if not already done:
    python v25_fusion_window.py --stage fit

Develop/select v27 WITHOUT touching the original test labels:
    python v27_clip_optimized_fusion.py --stage fit

Later, after development model selection is frozen, evaluate with the same
original 7,031 held-out clean and degraded clips:
    python v27_clip_optimized_fusion.py --stage evaluate

No existing model/OOF cache is overwritten; v27 uses the existing v25 OOF cache.
"""

import argparse
import hashlib
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, f1_score

import improvedCQCC_cqcc_f0_fusion_v19_calibrated as base
import improvedCQCC_cqcc_f0_fusion_v22_window_level as win
import improvedCQCC_v25_fusion_window as v25
from improvedCQCC_config import MODELS_DIR, RESULTS_DIR, FOLD_ASSIGNMENTS, TEST_SPEAKERS

VERSION = "v27_clip_optimized_aligned_window_v25_experts"
FOLDER = os.path.join(MODELS_DIR, "Prediction-Fusion", "CNN-GRU-CQCC-F0-F-v27-ClipOptimized")
MODEL_PATH = os.path.join(FOLDER, "fusion_model_v27_clip_optimized.npy")
RESULT_PATH = os.path.join(RESULTS_DIR, "cnn_gru_cqcc_f0_v27_clip_optimized_results.npy")
OOF_INPUT_PATH = v25.OOF_PATH
# Kept intentionally small to limit development-fold hyperparameter search.
C_GRID = (0.01, 0.1, 1.0, 10.0)
MIN_SPEAKERS = 4


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def verify_oof(records, cache):
    if cache.get("version") != v25.VERSION:
        raise RuntimeError(
            f"Refusing incompatible OOF cache version {cache.get('version')!r}; "
            f"expected {v25.VERSION!r}. Regenerate with: "
            "python v25_fusion_window.py --stage fit"
        )
    if len(records) != 26209:
        raise RuntimeError(f"Expected 26,209 v25 OOF clips; found {len(records):,}.")
    paths = set()
    fold_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in records:
        path = os.path.normpath(str(r["path"]))
        if path in paths:
            raise RuntimeError(f"Duplicated OOF path: {path}")
        paths.add(path)
        speaker = str(r["speaker"])
        if speaker in TEST_SPEAKERS or speaker not in FOLD_ASSIGNMENTS:
            raise RuntimeError(f"OOF speaker not permitted in development split: {speaker}")
        expected_fold = int(FOLD_ASSIGNMENTS[speaker])
        if int(r.get("fold", -1)) != expected_fold:
            raise RuntimeError(f"Invalid OOF fold for {speaker} / {path}")
        fold_counts[expected_fold] += 1
        if int(r["label"]) not in (0, 1):
            raise RuntimeError(f"Invalid OOF label: {path}")
        sizes = []
        for k in ("cnn_probs", "cqcc_probs", "f0_probs"):
            p = np.asarray(r[k], dtype=float)
            if p.ndim != 1 or not p.size or not np.isfinite(p).all():
                raise RuntimeError(f"Invalid OOF expert probabilities: {k} / {path}")
            if np.any((p < 0.0) | (p > 1.0)):
                raise RuntimeError(f"Probability outside [0,1]: {k} / {path}")
            sizes.append(p.size)
        if len(set(sizes)) != 1:
            raise RuntimeError(f"OOF expert window-count mismatch: {path}")
    if not all(fold_counts.values()):
        raise RuntimeError(f"OOF development fold empty: {fold_counts}")
    return fold_counts


def load_existing_v25_oof():
    if not os.path.isfile(OOF_INPUT_PATH):
        raise FileNotFoundError(
            f"Missing v25 OOF cache: {OOF_INPUT_PATH}\n"
            "Run: python v25_fusion_window.py --stage fit"
        )
    cached = np.load(OOF_INPUT_PATH, allow_pickle=True).item()
    if not isinstance(cached, dict) or "records" not in cached:
        raise RuntimeError("v25 OOF cache is not in the expected dictionary format")
    records = list(cached["records"])
    fold_counts = verify_oof(records, cached)
    print(f"Loaded and validated {len(records):,} v25 OOF clips; folds={fold_counts}")
    return records


def clip_vector(record, calibrators):
    """Mean of EXACT nine v22 calibrated aligned-window features.

    A linear model on this vector has exactly the same clip logit as averaging
    the individual window logits (mean-logit aggregation).
    """
    window_x = win.window_features(record, calibrators)
    result = np.mean(window_x, axis=0)
    if result.shape != (len(win.WINDOW_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid v27 clip feature: {record.get('path')}")
    return result


def clip_matrix(records, calibrators):
    return np.stack([clip_vector(r, calibrators) for r in records], axis=0)


def make_classifier(c):
    # Pure L2 logistic regression, matching the v22 classifier specification.
    return LogisticRegression(
        C=float(c), solver="lbfgs", l1_ratio=0.0,
        max_iter=5000, tol=1e-6, random_state=base.SEED,
    )


def balanced_weights(conditions):
    return np.asarray(base.condition_balanced_weights(conditions), dtype=np.float64)


def build_meta_folds(records):
    labels = np.asarray([int(r["label"]) for r in records], dtype=int)
    speakers = np.asarray([str(r["speaker"]) for r in records], dtype=object)
    cond = np.asarray([str(r["condition"]) for r in records], dtype=object)
    n_groups = len(set(speakers))
    if n_groups < MIN_SPEAKERS:
        raise RuntimeError("At least four development speakers are required")

    folds = []
    print("\nFour speaker-grouped meta-folds: Platt calibration and scaling are fold-local")
    splitter = GroupKFold(n_splits=4)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(labels, labels, speakers), start=1):
        tr = [records[i] for i in train_idx]
        va = [records[i] for i in val_idx]
        if set(speakers[train_idx]) & set(speakers[val_idx]):
            raise RuntimeError("Speaker overlap in meta fold")
        calibrators = base.fit_platt_calibrators(tr)
        x_tr = clip_matrix(tr, calibrators)
        x_va = clip_matrix(va, calibrators)
        weights_tr = balanced_weights(cond[train_idx])
        weights_va = balanced_weights(cond[val_idx])
        scaler = StandardScaler()
        try:
            scaler.fit(x_tr, sample_weight=weights_tr)
        except TypeError:
            scaler.fit(x_tr)
        x_tr = scaler.transform(x_tr)
        x_va = scaler.transform(x_va)
        folds.append(dict(
            fold=fold_idx, train_idx=train_idx, val_idx=val_idx,
            x_tr=x_tr, x_va=x_va, y_tr=labels[train_idx],
            y_va=labels[val_idx], w_tr=weights_tr, w_va=weights_va,
            cond_val=cond[val_idx],
        ))
        print(f"  Fold {fold_idx}: train={len(tr):,} clips, val={len(va):,}, "
              f"train speakers={len(set(speakers[train_idx]))}, "
              f"val speakers={len(set(speakers[val_idx]))}")
    return folds, labels, speakers, cond


def select_regularization(folds, n, labels, cond):
    best = None
    print("\nSelecting L2 regularization by speaker-fold condition-balanced clip log-loss...")
    for c in C_GRID:
        out = np.full(n, np.nan, dtype=np.float64)
        losses = []
        for fold in folds:
            clf = make_classifier(c)
            clf.fit(fold["x_tr"], fold["y_tr"], sample_weight=fold["w_tr"])
            prob = clf.predict_proba(fold["x_va"])[:, 1]
            out[fold["val_idx"]] = prob
            loss = log_loss(fold["y_va"], np.clip(prob, 1e-6, 1-1e-6),
                            labels=[0,1], sample_weight=fold["w_va"])
            losses.append(float(loss))
        if not np.isfinite(out).all():
            raise RuntimeError("Missing meta-OOF predictions")
        threshold, macro_f1, overall_f1 = base.choose_condition_balanced_threshold(labels, out, cond)
        candidate = dict(c=float(c), oof_prob=out, fold_losses=losses,
                         mean_loss=float(np.mean(losses)),
                         std_loss=float(np.std(losses)),
                         threshold=float(threshold),
                         macro_f1=float(macro_f1), overall_f1=float(overall_f1))
        print(f"  C={c:<5g} fold-loss={candidate['mean_loss']:.6f} "
              f"+/-{candidate['std_loss']:.6f}  "
              f"OOF macro F1={macro_f1:.4f}  overall F1={overall_f1:.4f} "
              f"threshold={threshold:.3f}")
        if (best is None
                or candidate["mean_loss"] < best["mean_loss"] - 1e-12
                or (abs(candidate["mean_loss"] - best["mean_loss"]) <= 1e-12
                    and candidate["std_loss"] < best["std_loss"])):
            best = candidate
    return best


def fit_model():
    records = load_existing_v25_oof()
    folds, y, speakers, conditions = build_meta_folds(records)
    chosen = select_regularization(folds, len(records), y, conditions)

    # Final fit uses ONLY development records: final v25 expert checkpoints
    # are NOT refit; they are used later only to generate held-out predictions.
    calibrators = base.fit_platt_calibrators(records)
    x = clip_matrix(records, calibrators)
    clip_weights = balanced_weights(conditions)
    scaler = StandardScaler()
    try:
        scaler.fit(x, sample_weight=clip_weights)
    except TypeError:
        scaler.fit(x)
    x_s = scaler.transform(x)
    clf = make_classifier(chosen["c"])
    clf.fit(x_s, y, sample_weight=clip_weights)

    p_oof = chosen["oof_prob"]
    threshold = chosen["threshold"]
    clean_mask = np.asarray([c == "clean" for c in conditions])
    def subset_f1(mask):
        if not np.any(mask):
            return None
        return float(f1_score(y[mask], (p_oof[mask] >= threshold).astype(int), zero_division=0))

    artifact = dict(
        version=VERSION,
        source_oof_cache=OOF_INPUT_PATH,
        source_oof_sha256=_file_hash(OOF_INPUT_PATH),
        source_experts="v25_auc_aux",
        feature_names=list(win.WINDOW_FEATURE_NAMES),
        pooling="mean_window_features_exact_mean_logit",
        objective="clip_level_BCE_condition_balanced",
        best_c=float(chosen["c"]),
        aggregation="mean_logit",
        threshold=float(threshold),
        grouped_log_loss=float(chosen["mean_loss"]),
        grouped_log_loss_std=float(chosen["std_loss"]),
        fold_losses=np.asarray(chosen["fold_losses"], dtype=float),
        platt_calibrators=calibrators,
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        coef=clf.coef_.reshape(-1).astype(np.float64),
        intercept=float(clf.intercept_[0]),
        oof_macro_f1=float(chosen["macro_f1"]),
        oof_overall_f1=float(chosen["overall_f1"]),
        oof_clean_f1=subset_f1(clean_mask),
        oof_degraded_f1=subset_f1(~clean_mask),
        n_oof_clips=int(len(records)),
        n_speakers=int(len(set(speakers))),
        n_window_features=int(len(win.WINDOW_FEATURE_NAMES)),
    )
    os.makedirs(FOLDER, exist_ok=True)
    np.save(MODEL_PATH, artifact, allow_pickle=True)
    print("\nv27 clip-optimized fusion fitted on development predictions only")
    print(f"  Selected C        : {artifact['best_c']:g}")
    print(f"  OOF threshold     : {artifact['threshold']:.3f}")
    print(f"  OOF macro F1      : {artifact['oof_macro_f1']:.4f}")
    print(f"  OOF overall F1    : {artifact['oof_overall_f1']:.4f}")
    print(f"  OOF clean F1      : {artifact['oof_clean_f1']:.4f}")
    print(f"  OOF degraded F1   : {artifact['oof_degraded_f1']:.4f}")
    print(f"  Grouped log loss  : {artifact['grouped_log_loss']:.6f}")
    print(f"  Saved             : {MODEL_PATH}")
    print("\nFinal clip-level model coefficients (standardized 9-feature inputs):")
    idxs = np.argsort(np.abs(artifact["coef"]))[::-1]
    for idx in idxs:
        print(f"  {artifact['feature_names'][idx]:<32s} {artifact['coef'][idx]:+.5f}")

    old_artifact = v25.MODEL_PATH
    if os.path.isfile(old_artifact):
        try:
            old = np.load(old_artifact, allow_pickle=True).item()
            if old.get("version") == v25.VERSION:
                print("\nDevelopment OOF comparison (same base experts; different objectives):")
                print(f"  v25 window-BCE macro F1: {old['oof_macro_f1']:.4f}")
                print(f"  v27 clip-BCE macro F1  : {artifact['oof_macro_f1']:.4f}")
                print("  Note: threshold selection uses OOF labels; these are exploratory development metrics.")
        except (ValueError, KeyError, TypeError):
            pass
    return artifact


def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Missing fitted model: {MODEL_PATH}\n"
                                "Run: python v27_clip_optimized_fusion.py --stage fit")
    artifact = np.load(MODEL_PATH, allow_pickle=True).item()
    if artifact.get("version") != VERSION or list(artifact.get("feature_names", [])) != list(win.WINDOW_FEATURE_NAMES):
        raise RuntimeError("Incompatible v27 model artifact/version/features")
    return artifact


def predict_records(records, artifact):
    scaler = win.artifact_scaler(artifact)
    clf = win.artifact_classifier(artifact)
    x = clip_matrix(records, artifact["platt_calibrators"])
    probs = clf.predict_proba(scaler.transform(x))[:, 1]
    if not np.isfinite(probs).all():
        raise RuntimeError("Nonfinite fused clip probabilities")
    return probs


def check_inference_equivalence(records, artifact, n_checks=4):
    """Verify pooling reproduces original mean-logit of linear window scores."""
    scaler = win.artifact_scaler(artifact)
    clf = win.artifact_classifier(artifact)
    for r in records[:n_checks]:
        p_clip = float(predict_records([r], artifact)[0])
        p_windows = win.predict_clip_probability(
            r, artifact["platt_calibrators"], scaler, clf, "mean_logit"
        )
        if not np.isclose(p_clip, p_windows, atol=1e-8, rtol=1e-8):
            raise RuntimeError(f"Clip/window mean-logit equivalence failed for {r['path']} ")


def evaluate_split(split, artifact):
    files = base.get_split_files(split)
    if not files:
        raise RuntimeError(f"No test files found for split: {split}")
    dev_speakers = set(FOLD_ASSIGNMENTS)
    test_speakers = {os.path.basename(os.path.dirname(p)) for p in files}
    if test_speakers & dev_speakers:
        raise RuntimeError(f"Held-out test contains development speakers: {test_speakers & dev_speakers}")
    if not test_speakers.issubset(set(TEST_SPEAKERS)):
        raise RuntimeError(f"Unrecognized held-out speakers: {test_speakers - set(TEST_SPEAKERS)}")
    if len(files) != 7031:
        raise RuntimeError(f"Expected original 7,031 clips in {split}; found {len(files):,}")
    print(f"\nPredicting {split} using original frozen v25 acoustic experts ({len(files):,} clips)...")
    records = base.collect_expert_predictions(files, final=True)
    check_inference_equivalence(records, artifact)
    prob = predict_records(records, artifact)
    y = np.asarray([r["label"] for r in records], dtype=int)
    cond = np.asarray([str(r["condition"]) for r in records])
    m = win.calculate_metrics(y, prob, artifact["threshold"])
    print(f"\nv27 | {split} | C={artifact['best_c']:g} threshold={artifact['threshold']:.3f}")
    print(f"  Accuracy: {m['accuracy']:.4f}  F1: {m['f1']:.4f}")
    print(f"  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}")
    print(f"  ROC-AUC: {m['auc']:.4f}  EER: {m['eer']:.4f}")
    cm = m["cm"]
    print(f"  Confusion matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    result = dict(model="CNN-GRU-CQCC-F0-F", version=VERSION, split=split, **m)
    if split == "test_degraded":
        by_cond = {}
        print("  Condition-wise degraded diagnostics:")
        for c in sorted(set(cond)):
            mask = cond == c
            if len(np.unique(y[mask])) < 2:
                continue
            met = win.calculate_metrics(y[mask], prob[mask], artifact["threshold"])
            by_cond[c] = met
            print(f"    {c:<20} F1={met['f1']:.4f} Acc={met['accuracy']:.4f} AUC={met['auc']:.4f}")
        result["condition_results"] = by_cond
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "evaluate", "all"), default="fit",
                        help="FIT by default, without touching held-out test data")
    args = parser.parse_args()
    # Activate the same frozen v25 expert files only for evaluation; the fit
    # stage reads the verified precomputed v25 out-of-fold prediction cache.
    print("v27 Clip-Optimized Window-Level Late Fusion")
    print("  Same dataset / augmentations / 3 acoustic experts / 9 window features")
    print("  Optimization changed: clip-level BCE instead of window-level BCE")
    print(f"  OOF source: {OOF_INPUT_PATH}")
    if args.stage in ("fit", "all"):
        fit_model()
    if args.stage in ("evaluate", "all"):
        print("\nWARNING: original held-out sets have already informed earlier research decisions.")
        artifact = load_model()
        v25.patch_fusion_sources()
        v25._check_checkpoints("evaluate")
        results = [evaluate_split("test", artifact), evaluate_split("test_degraded", artifact)]
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.save(RESULT_PATH, results, allow_pickle=True)
        print(f"\nSaved held-out evaluation: {RESULT_PATH}")


if __name__ == "__main__":
    base.set_seed(base.SEED)
    main()
'''

_SOURCE_improvedCQCC_v28_temporal_evidence_fusion = r'''
"""v28: Causal temporal-residual CNN-GRU / CQCC / F0 prediction-level fusion.

Only the FUSION algorithm changes. No new audio, augmentation, feature cache,
windowing, or acoustic-expert training is performed. All three experts are
v25 frozen and the nine calibrated v22 aligned-window features are reused.

Development:
    python v28_temporal_evidence_fusion.py --stage fit

OPTIONAL held-out evaluation (after reviewing development-fold results):
    python v28_temporal_evidence_fusion.py --stage evaluate

Dependencies alongside this file in E:\\Thesis\\code:
    config.py, robustness.py, train.py, cqcc_train.py, f0_train.py,
    cqcc_f0_fusion_v19_calibrated.py, cqcc_f0_fusion_v22_window_level.py,
    v25_aux_train_auc.py, v25_fusion_window.py, v27_clip_optimized_fusion.py

Prerequisite: v25 fitted and OOF cache present (26,209 clips):
    python v25_fusion_window.py --stage fit

Architecture: clip-level L2 logistic reference + small CAUSAL dilated temporal
convolution trained as a bounded logit correction. Zero-initialized output
ensures the starting neural model exactly matches its reference. This tests
whether chronological patterns add value beyond time-averaged clip features.

Leakage boundaries: v25 expert OOF predictions exclude each corresponding
speaker from that expert's training; outer meta GroupKFold calibration/scaling/
reference fit/temporal training and early stopping use only each meta-training
speaker partition. No test data enters any fit or threshold selection. As with
prior v25-v27 stacking CV, the existing four-fold expert OOF cache is reused;
this is not a fully nested retraining of experts per meta fold.
"""

import argparse
import copy
import hashlib
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import improvedCQCC_cqcc_f0_fusion_v19_calibrated as base
import improvedCQCC_cqcc_f0_fusion_v22_window_level as win
import improvedCQCC_v25_fusion_window as v25
import improvedCQCC_v27_clip_optimized_fusion as v27
from improvedCQCC_config import MODELS_DIR, RESULTS_DIR, FOLD_ASSIGNMENTS, TEST_SPEAKERS, SEED

VERSION = "v28_causal_temporal_residual_v25_experts"
FOLDER = os.path.join(MODELS_DIR, "Prediction-Fusion", "CNN-GRU-CQCC-F0-F-v28-Temporal")
META_PATH = os.path.join(FOLDER, "fusion_model_v28_temporal.npy")
FINAL_STATE_PATH = os.path.join(FOLDER, "temporal_final_v28.pt")
RESULT_PATH = os.path.join(RESULTS_DIR, "cnn_gru_cqcc_f0_v28_temporal_results.npy")
OOF_PATH = v25.OOF_PATH

# Fixed v27 reference hyperparameter: no held-out selection or broad search.
REFERENCE_C = 10.0
HIDDEN = 16
DROPOUT = 0.10
CORRECTION_MAX = 1.5
CORRECTION_L2 = 0.002
TRAIN_EPOCHS = 28
PATIENCE = 7
MIN_EPOCHS = 8
LR = 2e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 192
N_FOLDS = 4
N_FEATURES = len(win.WINDOW_FEATURE_NAMES)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records():
    # Reuse v27's strict cache verification, including original speaker/fold map.
    return v27.load_existing_v25_oof()


def get_data(records, calibrators):
    seqs = [np.asarray(win.window_features(r, calibrators), dtype=np.float32) for r in records]
    for idx, s in enumerate(seqs):
        if s.ndim != 2 or s.shape[1] != N_FEATURES or len(s) < 1 or not np.isfinite(s).all():
            raise RuntimeError(f"Bad calibrated temporal features for {records[idx]['path']}")
    return seqs


def fit_clip_balanced_scaler(seqs, clip_w):
    # Each clip has weight clip_w, distributed evenly over its windows. Avoids
    # long recordings dominating the feature normalization fit.
    all_windows = np.concatenate(seqs, axis=0).astype(np.float64)
    all_weights = np.concatenate([np.full(len(s), w / len(s), dtype=np.float64)
                                  for s, w in zip(seqs, clip_w)])
    scaler = StandardScaler()
    try:
        scaler.fit(all_windows, sample_weight=all_weights)
    except TypeError:
        scaler.fit(all_windows)
    return scaler


def transform_seqs(seqs, scaler):
    return [np.ascontiguousarray(scaler.transform(x), dtype=np.float32) for x in seqs]


def fit_reference(x_seqs, labels, weights):
    # Same clip-level L2 logistic strategy used in v27. The reference is trained
    # separately in each outer meta fold, never from all development labels.
    x_clip = np.stack([x.mean(axis=0) for x in x_seqs]).astype(np.float64)
    clf = LogisticRegression(C=REFERENCE_C, solver="lbfgs", l1_ratio=0.0,
                             max_iter=5000, random_state=SEED)
    clf.fit(x_clip, labels, sample_weight=weights)
    return clf


def reference_logits(clf, seqs):
    x = np.stack([s.mean(axis=0) for s in seqs]).astype(np.float64)
    return clf.decision_function(x).astype(np.float32).reshape(-1)


class TemporalResidual(nn.Module):
    """Two residual-time causal conv blocks; only *past* windows used at t."""

    def __init__(self):
        super().__init__()
        self.first = nn.Conv1d(N_FEATURES, HIDDEN, kernel_size=3, dilation=1)
        self.second = nn.Conv1d(HIDDEN, HIDDEN, kernel_size=3, dilation=2)
        self.dropout = nn.Dropout(DROPOUT)
        self.out = nn.Conv1d(HIDDEN, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, lengths, baseline):
        # x: B,T,D; sequences have right-hand zero padding only.
        z = x.transpose(1, 2)
        h = F.gelu(self.first(F.pad(z, (2, 0))))
        h = self.dropout(h)
        h = F.gelu(self.second(F.pad(h, (4, 0))))
        h = self.dropout(h)
        per_window = self.out(h).squeeze(1)
        mask = (torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None])
        mean_delta = (per_window * mask).sum(dim=1) / lengths.clamp_min(1)
        delta = CORRECTION_MAX * torch.tanh(mean_delta / CORRECTION_MAX)
        return baseline + delta, delta


class SequenceDataset(Dataset):
    def __init__(self, seqs, y, reference, w):
        self.seqs = seqs
        self.labels = np.asarray(y, dtype=np.float32)
        self.reference = np.asarray(reference, dtype=np.float32)
        self.weights = np.asarray(w, dtype=np.float32)
        if not (len(seqs) == len(self.labels) == len(self.reference) == len(self.weights)):
            raise RuntimeError("Dataset length mismatch")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx], self.reference[idx], self.weights[idx]


def collate(batch):
    lengths = torch.tensor([len(z[0]) for z in batch], dtype=torch.long)
    x = torch.zeros(len(batch), int(lengths.max()), N_FEATURES, dtype=torch.float32)
    for i, item in enumerate(batch):
        x[i, :len(item[0])] = torch.as_tensor(item[0], dtype=torch.float32)
    y = torch.tensor([float(z[1]) for z in batch], dtype=torch.float32)
    ref = torch.tensor([float(z[2]) for z in batch], dtype=torch.float32)
    w = torch.tensor([float(z[3]) for z in batch], dtype=torch.float32)
    return x, lengths, y, ref, w


def make_loader(dataset, batch, shuffle, seed):
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0,
                      collate_fn=collate, generator=g, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict_dataset(model, loader, device):
    model.eval()
    probs, deltas = [], []
    for x, lengths, _y, ref, _w in loader:
        score, delta = model(x.to(device), lengths.to(device), ref.to(device))
        probs.append(torch.sigmoid(score).cpu().numpy())
        deltas.append(delta.cpu().numpy())
    return np.concatenate(probs).astype(np.float64), np.concatenate(deltas).astype(np.float64)


def balanced_weights(cond):
    return np.asarray(base.condition_balanced_weights(cond), dtype=np.float32)


def train_temporal(x_tr, y_tr, r_tr, w_tr, x_val, y_val, r_val, w_val,
                   device, seed, epochs=TRAIN_EPOCHS, label="fold"):
    seed_all(seed)
    model = TemporalResidual().to(device)
    ds_tr = SequenceDataset(x_tr, y_tr, r_tr, w_tr)
    ds_va = SequenceDataset(x_val, y_val, r_val, w_val)
    tr_loader = make_loader(ds_tr, BATCH_SIZE, True, seed)
    va_loader = make_loader(ds_va, BATCH_SIZE * 2, False, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-5)

    # Epoch 0 EXACTLY equals the trained clip-level logistic reference.
    base_prob = 1.0 / (1.0 + np.exp(-np.clip(r_val.astype(np.float64), -50, 50)))
    best_loss = float(log_loss(y_val, np.clip(base_prob, 1e-7, 1 - 1e-7),
                               labels=[0, 1], sample_weight=w_val))
    best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    best_epoch = 0
    wait = 0
    print(f"    {label}: epoch=00 clip loss={best_loss:.6f} (reference, no temporal correction)")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_weight = 0.0, 0.0
        for x, lengths, y, ref, w in tr_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            ref, w = ref.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, delta = model(x, lengths, ref)
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = (bce * w).sum() / w.sum().clamp_min(1e-8)
            loss = loss + CORRECTION_L2 * (delta.square() * w).sum() / w.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * float(w.sum().item())
            total_weight += float(w.sum().item())

        p_val, delta_val = predict_dataset(model, va_loader, device)
        val_loss = float(log_loss(y_val, np.clip(p_val, 1e-7, 1-1e-7),
                                  labels=[0, 1], sample_weight=w_val))
        scheduler.step(val_loss)
        improved = (val_loss < best_loss - 1e-5)
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 2 == 0 or improved:
            print(f"    {label}: epoch={epoch:02d} train={total_loss/max(total_weight,1e-8):.5f} "
                  f"val={val_loss:.6f} best={best_loss:.6f}@{best_epoch:02d} "
                  f"mean|delta|={np.mean(np.abs(delta_val)):.3f} patience={wait}/{PATIENCE}")
        if epoch >= MIN_EPOCHS and wait >= PATIENCE:
            print(f"    {label}: early stop at {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def collect_oof_temporal(records, device):
    y = np.array([int(r["label"]) for r in records], dtype=int)
    groups = np.array([str(r["speaker"]) for r in records], dtype=object)
    cond = np.array([str(r["condition"]) for r in records], dtype=object)
    probs = np.full(len(records), np.nan, dtype=np.float64)
    baseline_probs = np.full(len(records), np.nan, dtype=np.float64)
    reversed_probs = np.full(len(records), np.nan, dtype=np.float64)
    selected_epochs, fold_losses = [], []
    os.makedirs(FOLDER, exist_ok=True)
    print("\nSpeaker-grouped v28 temporal CV: calibration, scaling, baseline, and residual training are fold-local")
    splitter = GroupKFold(n_splits=N_FOLDS)
    for f, (tr, va) in enumerate(splitter.split(y, y, groups), 1):
        if set(groups[tr]) & set(groups[va]):
            raise RuntimeError("Speaker overlap across meta-fold split")
        train_records = [records[i] for i in tr]
        val_records = [records[i] for i in va]
        cal = base.fit_platt_calibrators(train_records)
        raw_tr = get_data(train_records, cal)
        raw_va = get_data(val_records, cal)
        w_tr, w_va = balanced_weights(cond[tr]), balanced_weights(cond[va])
        scaler = fit_clip_balanced_scaler(raw_tr, w_tr)
        x_tr = transform_seqs(raw_tr, scaler)
        x_va = transform_seqs(raw_va, scaler)
        clf = fit_reference(x_tr, y[tr], w_tr)
        ref_tr = reference_logits(clf, x_tr)
        ref_va = reference_logits(clf, x_va)
        baseline_probs[va] = 1.0 / (1.0 + np.exp(-np.clip(ref_va, -50, 50)))
        print(f"\n  Fold {f}: train={len(tr):,} clips/{len(set(groups[tr]))} speakers "
              f"val={len(va):,} clips/{len(set(groups[va]))} speakers")
        model, best_ep, best_loss = train_temporal(
            x_tr, y[tr], ref_tr, w_tr, x_va, y[va], ref_va, w_va,
            device, SEED + f, label=f"fold{f}")
        probs[va], _ = predict_dataset(
            model, make_loader(SequenceDataset(x_va, y[va], ref_va, w_va),
                               BATCH_SIZE * 2, False, SEED), device)
        # No augmentation: merely reverse cached *predictions* diagnostically.
        reversed_inputs = [np.ascontiguousarray(x[::-1]) for x in x_va]
        reversed_probs[va], _ = predict_dataset(
            model, make_loader(SequenceDataset(reversed_inputs, y[va], ref_va, w_va),
                               BATCH_SIZE * 2, False, SEED), device)
        actual_loss = float(log_loss(y[va], np.clip(probs[va], 1e-7, 1-1e-7),
                                     labels=[0,1], sample_weight=w_va))
        if abs(actual_loss - best_loss) > 1e-4:
            raise RuntimeError(f"Fold {f} restored model did not reproduce validation loss")
        print(f"  Fold {f}: selected_epoch={best_ep}, loss={actual_loss:.6f}, "
              f"mean order-reversal |p diff|={np.mean(np.abs(probs[va] - reversed_probs[va])):.6f}")
        selected_epochs.append(int(best_ep))
        fold_losses.append(actual_loss)
        torch.save({"state_dict": {k: v.cpu() for k,v in model.state_dict().items()},
                    "version": VERSION, "fold": f, "best_epoch": best_ep,
                    "val_loss": actual_loss, "train_speakers": sorted(set(groups[tr])),
                    "val_speakers": sorted(set(groups[va]))},
                   os.path.join(FOLDER, f"temporal_best_fold{f}.pt"))

    if not (np.isfinite(probs).all() and np.isfinite(baseline_probs).all()):
        raise RuntimeError("OOF predictions incomplete")
    return dict(y=y, groups=groups, conditions=cond, probs=probs,
                baseline_probs=baseline_probs, reverse_probs=reversed_probs,
                epochs=selected_epochs, fold_losses=fold_losses)


def final_train(records, epochs, device):
    # Fit on all 26,209 development clips, with new deterministic weights. No
    # held-out examples or test labels are accessed here.
    y = np.array([int(r["label"]) for r in records], dtype=int)
    cond = np.array([str(r["condition"]) for r in records], dtype=object)
    cal = base.fit_platt_calibrators(records)
    raw = get_data(records, cal)
    w = balanced_weights(cond)
    scaler = fit_clip_balanced_scaler(raw, w)
    x = transform_seqs(raw, scaler)
    clf = fit_reference(x, y, w)
    ref = reference_logits(clf, x)
    seed_all(SEED + 101)
    net = TemporalResidual().to(device)
    if epochs:
        loader = make_loader(SequenceDataset(x, y, ref, w), BATCH_SIZE, True, SEED + 101)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for ep in range(1, epochs+1):
            net.train()
            loss_sum, weight_sum = 0.0, 0.0
            for xx, lengths, yy, rr, ww in loader:
                xx, lengths, yy = xx.to(device), lengths.to(device), yy.to(device)
                rr, ww = rr.to(device), ww.to(device)
                opt.zero_grad(set_to_none=True)
                logits, delta = net(xx, lengths, rr)
                per = F.binary_cross_entropy_with_logits(logits, yy, reduction="none")
                loss = (per * ww).sum() / ww.sum().clamp_min(1e-8)
                loss = loss + CORRECTION_L2 * (delta.square()*ww).sum() / ww.sum().clamp_min(1e-8)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
                loss_sum += float(loss.item()) * float(ww.sum().item())
                weight_sum += float(ww.sum().item())
            if ep == 1 or ep % 5 == 0 or ep == epochs:
                print(f"  Final temporal epoch {ep:02d}/{epochs} loss={loss_sum/max(weight_sum,1e-8):.6f}")

    torch.save({"state_dict": {k: v.cpu() for k,v in net.state_dict().items()},
                "version": VERSION, "final_epochs": int(epochs)}, FINAL_STATE_PATH)
    return dict(calibrators=cal,
                scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
                scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
                reference_coef=np.asarray(clf.coef_.reshape(-1), dtype=np.float64),
                reference_intercept=float(clf.intercept_[0]))


def subset_f1(y, p, conditions, threshold, cond_name):
    mask = conditions == cond_name
    return float(f1_score(y[mask], (p[mask] >= threshold).astype(int), zero_division=0)) if mask.any() else None


def fit_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(SEED)
    print("v28 Temporal Evidence Fusion (causal residual TCN, v25 experts frozen)")
    print(f"  Device: {device}; OOF: {OOF_PATH}")
    records = load_records()
    source_hash = sha256(OOF_PATH)
    cross = collect_oof_temporal(records, device)
    y, cond, p = cross["y"], cross["conditions"], cross["probs"]
    threshold, macro, overall = base.choose_condition_balanced_threshold(y, p, cond)
    baseline_t, baseline_macro, baseline_overall = base.choose_condition_balanced_threshold(
        y, cross["baseline_probs"], cond)
    selected_epochs = cross["epochs"]
    final_epochs = int(np.median(selected_epochs))  # matches CV-median retraining policy
    fold_loss = np.asarray(cross["fold_losses"])
    print("\nDevelopment results (four speaker-grouped folds):")
    print(f"  v28 temporal     macro F1={macro:.4f} overall F1={overall:.4f} threshold={threshold:.3f}")
    print(f"  Clip reference   macro F1={baseline_macro:.4f} overall F1={baseline_overall:.4f} threshold={baseline_t:.3f}")
    print(f"  v28 OOF clean F1={subset_f1(y,p,cond,threshold,'clean'):.4f} "
          f"degraded F1={f1_score(y[cond!='clean'],(p[cond!='clean']>=threshold).astype(int),zero_division=0):.4f}")
    print(f"  fold loss={np.round(fold_loss,6).tolist()} mean={np.mean(fold_loss):.6f}")
    for c in sorted(set(cond)):
        print(f"  {c:<20} v28 F1={subset_f1(y,p,cond,threshold,c):.4f}")
    print(f"  Reverse-order diagnostic: mean |p - p_reversed|="
          f"{np.mean(np.abs(p-cross['reverse_probs'])):.6f}")
    print(f"  Best CV epochs={selected_epochs}; final training epochs={final_epochs}")
    if os.path.isfile(v25.MODEL_PATH):
        try:
            old = np.load(v25.MODEL_PATH, allow_pickle=True).item()
            if old.get("version") == v25.VERSION:
                older = float(old["oof_macro_f1"])
                print(f"  Existing v25 window fusion: OOF macro F1={older:.4f}")
                print(f"  v28 - v25 OOF macro F1: {macro-older:+.4f}")
                if macro <= older + 0.002:
                    print("  CAUTION: v28 improvement <=0.002 over v25; review before testing original held-out clips.")
        except (ValueError, TypeError, KeyError) as exc:
            print(f"  Could not read v25 development comparison: {exc}")

    print("\nRefitting temporal fusion on all development predictions (no held-out test access)...")
    final = final_train(records, final_epochs, device)
    artifact = dict(version=VERSION, source_oof=OOF_PATH, source_sha256=source_hash,
                    reference_c=REFERENCE_C, architecture="causal_conv_k3_dilations_1_2_residual",
                    features=list(win.WINDOW_FEATURE_NAMES), n_records=len(records),
                    n_speakers=len(set(cross['groups'])),
                    threshold=float(threshold), best_epochs=selected_epochs,
                    final_epochs=final_epochs, oof_macro_f1=float(macro),
                    oof_overall_f1=float(overall), oof_clip_baseline_macro_f1=float(baseline_macro),
                    oof_clip_baseline_threshold=float(baseline_t),
                    oof_clean_f1=subset_f1(y,p,cond,threshold,'clean'),
                    oof_degraded_f1=float(f1_score(y[cond!='clean'], (p[cond!='clean']>=threshold).astype(int),zero_division=0)),
                    order_reverse_gap=float(np.mean(np.abs(p-cross['reverse_probs']))),
                    grouped_logloss=float(np.mean(fold_loss)),
                    fold_losses=fold_loss, seed=int(SEED),
                    net_state=FINAL_STATE_PATH, **final)
    np.save(META_PATH, artifact, allow_pickle=True)
    np.save(os.path.join(FOLDER,"development_oof_v28.npy"),
            dict(labels=y, conditions=cond, speakers=cross['groups'], probabilities=p,
                 reference_probabilities=cross['baseline_probs'], reversed_probabilities=cross['reverse_probs']),
            allow_pickle=True)
    print(f"\nSaved v28 temporal model: {META_PATH}")
    print(f"Saved v28 temporal network: {FINAL_STATE_PATH}")
    print("\nDecision rule: advance to held-out testing only if development-fold improvements are meaningful;")
    print("otherwise preserve v25 as the current best F1 version.")


def load_artifact():
    if not os.path.isfile(META_PATH) or not os.path.isfile(FINAL_STATE_PATH):
        raise FileNotFoundError("Missing v28 model. Run --stage fit first.")
    artifact = np.load(META_PATH, allow_pickle=True).item()
    if artifact.get("version") != VERSION or artifact.get("features") != list(win.WINDOW_FEATURE_NAMES):
        raise RuntimeError("Incompatible v28 artifact or feature names")
    if artifact.get("source_oof") != OOF_PATH or artifact.get("source_sha256") != sha256(OOF_PATH):
        raise RuntimeError("v25 OOF cache changed since v28 fitting. Refit v28 or restore original cache.")
    return artifact


def inference(records, artifact, net, device):
    cal = artifact["calibrators"]
    mu = artifact["scaler_mean"].astype(np.float32)
    scale = artifact["scaler_scale"].astype(np.float32)
    if np.any(scale <= 0):
        raise RuntimeError("Invalid trained scale")
    raw = get_data(records, cal)
    x = [np.ascontiguousarray((s-mu)/scale, dtype=np.float32) for s in raw]
    coef = np.asarray(artifact["reference_coef"], dtype=np.float64)
    intercept = float(artifact["reference_intercept"])
    ref = np.array([np.mean(s,axis=0).astype(np.float64) @ coef + intercept
                    for s in x], dtype=np.float32)
    dummy_y = np.zeros(len(records), dtype=np.float32)
    dummy_w = np.ones(len(records), dtype=np.float32)
    p, d = predict_dataset(net,
            make_loader(SequenceDataset(x, dummy_y, ref, dummy_w), BATCH_SIZE*2, False, SEED), device)
    return p, d


def evaluate_split(split, artifact, net, device):
    files = base.get_split_files(split)
    speakers = {os.path.basename(os.path.dirname(p)) for p in files}
    if len(files) != 7031 or not speakers.issubset(set(TEST_SPEAKERS)) or (speakers & set(FOLD_ASSIGNMENTS)):
        raise RuntimeError("Held-out speaker/count check failed")
    print(f"\nFrozen v25 expert inference: {split} ({len(files):,} clips)")
    records = base.collect_expert_predictions(files, final=True)
    prob, delta = inference(records, artifact, net, device)
    y = np.asarray([int(r["label"]) for r in records])
    cond = np.asarray([str(r["condition"]) for r in records])
    met = win.calculate_metrics(y, prob, artifact["threshold"])
    print(f"  F1={met['f1']:.4f} accuracy={met['accuracy']:.4f} "
          f"AUC={met['auc']:.4f} EER={met['eer']:.4f} "
          f"precision={met['precision']:.4f} recall={met['recall']:.4f}")
    cm=met['cm']
    print(f"  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]} mean|correction|={np.mean(np.abs(delta)):.4f}")
    result = dict(model="CNN-GRU-CQCC-F0-F",version=VERSION,split=split,**met)
    if split == "test_degraded":
        details={}
        for c in sorted(set(cond)):
            mask=cond==c
            if len(set(y[mask]))<2: continue
            part=win.calculate_metrics(y[mask],prob[mask],artifact["threshold"])
            details[c]=part
            print(f"  {c:<20} F1={part['f1']:.4f} AUC={part['auc']:.4f}")
        result["condition_results"]=details
    return result


def evaluate_model():
    print("WARNING: Original held-out test sets have informed earlier research decisions.")
    artifact=load_artifact()
    # Enable v25 final expert checkpoints without affecting the training mode.
    v25.patch_fusion_sources()
    v25._check_checkpoints("evaluate")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=TemporalResidual().to(device)
    ckpt=torch.load(FINAL_STATE_PATH,map_location=device,weights_only=True)
    if ckpt.get("version")!=VERSION or int(ckpt.get("final_epochs",-1))!=int(artifact["final_epochs"]):
        raise RuntimeError("v28 temporal network state/version mismatch")
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    results=[evaluate_split("test",artifact,net,device),
             evaluate_split("test_degraded",artifact,net,device)]
    os.makedirs(RESULTS_DIR,exist_ok=True)
    np.save(RESULT_PATH,results,allow_pickle=True)
    print(f"\nSaved v28 evaluation: {RESULT_PATH}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=["fit","evaluate"],default="fit",
                    help="Default FIT on development only. Evaluation must be explicit.")
    args=ap.parse_args()
    if args.stage=="fit": fit_model()
    else: evaluate_model()


if __name__=="__main__":
    main()
'''

_SOURCE_improvedCQCC_akinci_fusion_impl = r'''
"""Controlled Akıncı–Özbay-inspired CQCC experiment in the EXISTING v28 temporal fusion.

Same CNN-GRU / F0 experts and unchanged causal temporal fusion algorithm.
Only CQCC acoustic representation and its separately trained neural expert change.
Original v25 OOF CNN-GRU/F0 predictions are reused; CQCC is rescored with
speaker-disjoint Akıncı–Özbay CQCC expert checkpoints. The NEW temporal fusion,
calibrations and threshold are then fitted on development speakers only.

  python akinci_temporal_fusion_trial.py --stage check
  python akinci_temporal_fusion_trial.py --stage oof
  python akinci_temporal_fusion_trial.py --stage fit
  # Only after dev assessment, final CQCC expert and separately generated test sidecars:
  python akinci_temporal_fusion_trial.py --stage evaluate

WARNING: The existing held-out test recordings have influenced past comparisons.
All new held-out metrics are exploratory/post-selection, not confirmatory.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import improvedCQCC_config as cfg
import improvedCQCC_cqcc_f0_fusion_v19_calibrated as base
import improvedCQCC_cqcc_f0_fusion_v22_window_level as win
import improvedCQCC_v25_fusion_window as v25
import improvedCQCC_v27_clip_optimized_fusion as v27
import improvedCQCC_v28_temporal_evidence_fusion as v28
from improvedCQCC_cqcc_train import VERSION as CQCC_TRAIN_VERSION

VERSION = 'akinci_ozbay_cqcc_v28_temporal_trial_v1'
ROOT = Path(cfg.MODELS_DIR) / 'Prediction-Fusion' / 'CNN-GRU-CQCC-F0-F-Akinci-Temporal-EXP'
CQCC_DIR = Path(cfg.MODELS_DIR) / 'CQCC-Akinci-Ozbay-DEV'
SIDE_ROOT = Path(cfg.BASE_DIR) / 'features_experiments' / 'akinci_ozbay_cqcc'
OOF = ROOT / 'oof_predictions_akinci.npy'
META = ROOT / 'fusion_model_akinci_temporal.npy'
STATE = ROOT / 'temporal_final_akinci.pt'
DEV_OOF = ROOT / 'development_oof_v28.npy'
RESULT_ROOT = Path(cfg.RESULTS_DIR) / 'akinci_ozbay_temporal_experiment'
RESULT = RESULT_ROOT / 'heldout_exploratory_results.npy'
SOURCE_OOF = Path(v25.OOF_PATH)
DEV_SIDE_VERSION = 'akinci_ozbay_2026_cqcc_dev_sidecar_v1'
TEST_SIDE_VERSION = 'akinci_ozbay_2026_cqcc_heldout_sidecar_v1'
EXPECTED_DEV_N = 26209
EXPECTED_TEST_N = 7031


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def fold_info(fold):
    path=CQCC_DIR/'fold_results.npy'
    if not path.is_file(): raise FileNotFoundError(f'New CQCC CV missing: {path}')
    results=list(np.load(path,allow_pickle=True))
    matching=[r for r in results if r.get('training_version')==CQCC_TRAIN_VERSION and int(r['fold'])==fold]
    if len(matching)!=1 or len(results)!=4:
        raise RuntimeError('New CQCC CV results are incomplete/incompatible')
    r=matching[0]
    mu=np.asarray(r['mean'],np.float32)
    std=np.asarray(r['std'],np.float32)
    if mu.shape!=(40,) or std.shape!=(40,) or np.any(std<=0):
        raise RuntimeError('Invalid experimental CQCC fold normalization')
    return mu,std


def final_info():
    path=CQCC_DIR/'final_norm_stats.npy'
    if not path.is_file():raise FileNotFoundError(f'Run: python train_akinci_cqcc_expert.py --stage final (missing {path})')
    d=np.load(path,allow_pickle=True).item()
    if d.get('training_version')!=CQCC_TRAIN_VERSION or d.get('feature')!='cqcc':
        raise RuntimeError('Experimental final CQCC stats incompatible with training version')
    mu=np.asarray(d['mean'],np.float32)
    std=np.asarray(d['std'],np.float32)
    if mu.shape!=(40,) or std.shape!=(40,) or np.any(std<=0):
        raise RuntimeError('Invalid experimental CQCC final normalization')
    return mu,std


def preflight_checkpoint(final=False):
    from improvedCQCC_cqcc_train import CQCCMLP
    CQCC_DIM=40
    folds=(None,) if final else range(1,5)
    hashes={}
    for fold in folds:
        if final:final_info()
        else:fold_info(fold)
        path=CQCC_DIR/('final_model.pt' if final else f'best_fold{fold}.pt')
        if not path.is_file():raise FileNotFoundError(f'New CQCC checkpoint missing: {path}')
        model=CQCCMLP(CQCC_DIM)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        hashes['final' if final else str(fold)]=sha256(path)
    return hashes


def side_path(original_path):
    p=Path(original_path)
    split=p.parent.parent.name
    if split not in ('train','test','test_degraded'):
        raise RuntimeError(f'Unexpected original cache split: {p}')
    try:
        expected=Path(cfg.FEATURES_DIR)/split/p.parent.name/p.name
        if p.resolve()!=expected.resolve():
            raise RuntimeError(f'Feature path outside original cache: {p}')
    except OSError:
        pass
    return SIDE_ROOT/split/p.parent.name/p.name


def check_sidecar_pair(path, old_record=None):
    side=side_path(path)
    if not side.is_file():
        raise FileNotFoundError(f'Missing paper CQCC sidecar: {side}. Run appropriate extraction command.')
    split=Path(path).parent.parent.name
    expected_version=DEV_SIDE_VERSION if split=='train' else TEST_SIDE_VERSION
    with np.load(path,allow_pickle=False) as old, np.load(side,allow_pickle=False) as d:
        if str(d.get('cache_version',''))!=expected_version:
            raise RuntimeError(f'Wrong sidecar cache version: {side}')
        arr=d['cqcc']
        orig=old['cqcc']
        if arr.shape != orig.shape or arr.ndim!=2 or arr.shape[1]!=40 or not np.isfinite(arr).all():
            raise RuntimeError(f'CQCC sidecar window/dim mismatch: {side}')
        for key in ('label','speaker','condition'):
            if str(d[key])!=str(old[key]):
                raise RuntimeError(f'Experimental CQCC {key} disagrees with original cache: {side}')
        if old_record is not None:
            if (int(d['label'])!=int(old_record['label']) or
                str(d['speaker'])!=str(old_record['speaker']) or
                str(d['condition'])!=str(old_record['condition']) or
                len(arr)!=len(old_record['cnn_probs'])):
                raise RuntimeError(f'Experimental CQCC does not match original OOF record: {side}')
    return str(side)


def patch_experiment():
    """Patch module-level data-source hooks, never write original files."""
    v25.patch_fusion_sources()
    original_load=base.load_state
    original_predict=base.predict_cqcc_windows

    def experimental_load(model_name,fold=None,final=False):
        if model_name=='CQCC':
            if final: final_info()
            else: fold_info(fold)
            p=CQCC_DIR/('final_model.pt' if final else f'best_fold{fold}.pt')
            return torch.load(p,map_location=base.DEVICE,weights_only=True)
        return original_load(model_name,fold=fold,final=final)

    def experimental_cqcc_predict(model, files, mean, std):
        mapped=[check_sidecar_pair(path) for path in files]
        if len(set(mapped))!=len(files):raise RuntimeError('Nonunique source-sidecar mapping')
        vals=original_predict(model,mapped,mean,std)
        return {str(original):vals[mapped[i]] for i,original in enumerate(files)}

    # Existing v28 uses the actual 40-D CQCC expert; some archived cqcc_train.py
    # copies still advertise an older 80-D definition. This experimental
    # checkpoint and sidecar are always 40-D. Do not alter original files.
    base.CQCC_DIM=40
    base.load_state=experimental_load
    base.load_cqcc_fold_stats=lambda fold:fold_info(fold)
    base.load_cqcc_final_stats=final_info
    base.predict_cqcc_windows=experimental_cqcc_predict

    # Avoid v28 FIT's unrelated historical v25 comparison label (we separately
    # print the actual original v28 comparison after fitting).
    v25.MODEL_PATH=str(ROOT/'unavailable_original_v25_baseline.npy')

    v28.VERSION=VERSION
    v28.FOLDER=str(ROOT)
    v28.OOF_PATH=str(OOF)
    v28.META_PATH=str(META)
    v28.FINAL_STATE_PATH=str(STATE)
    v28.RESULT_PATH=str(RESULT)
    v28.load_records=load_experimental_oof


def load_original_records():
    if not SOURCE_OOF.is_file():
        raise FileNotFoundError(f'Original v25 OOF predictions missing: {SOURCE_OOF}; run original v25 fit first.')
    return v27.load_existing_v25_oof()


def validate_experiment_payload(payload, records, check_hashes=True):
    if payload.get('version')!=VERSION or len(records)!=EXPECTED_DEV_N:
        raise RuntimeError('New fusion OOF incomplete or incompatible')
    # Use original v27 strict folds/metadata/probability verifier by providing
    # the expected provenance version solely to that read-only validator.
    v27.verify_oof(records, {'version':v25.VERSION})
    if check_hashes:
        if payload.get('source_v25_oof_sha256')!=sha256(SOURCE_OOF):
            raise RuntimeError('Original source OOF changed since new CQCC score generation')
        if payload.get('paper_fold_checkpoint_sha256')!=preflight_checkpoint(final=False):
            raise RuntimeError('New CQCC fold checkpoint changed since OOF; rebuild the experimental OOF')
    return records


def load_experimental_oof():
    if not OOF.is_file():raise FileNotFoundError(f'No new OOF: {OOF}. Run --stage oof first.')
    payload=np.load(OOF,allow_pickle=True).item()
    records=list(payload['records'])
    return validate_experiment_payload(payload,records)


def build_oof():
    if OOF.exists():
        print(f'Experimental OOF already exists, validating and reusing: {OOF}')
        return load_experimental_oof()
    cv_hashes=preflight_checkpoint(final=False)
    original=load_original_records()
    if len(original)!=EXPECTED_DEV_N:raise RuntimeError('Original development OOF record count mismatch')
    print(f'Creating separate CQCC prediction scores for {len(original):,} development clips')
    for i,r in enumerate(original,1):
        check_sidecar_pair(str(r['path']),r)
        if i%5000==0:print(f'  validated {i:,}/{len(original):,} sidecar pairs')
    combined=[]
    for fold in range(1,5):
        part=[r for r in original if int(r['fold'])==fold]
        print(f'  new CQCC predictions fold {fold}: {len(part):,} clips; CNN-GRU/F0 reused unchanged')
        cnn_cqcc=base.CQCCMLP(base.CQCC_DIM).to(base.DEVICE)
        cnn_cqcc.load_state_dict(base.load_state('CQCC',fold=fold),strict=True)
        mu,std=base.load_cqcc_fold_stats(fold)
        paths=[str(r['path']) for r in part]
        pred=base.predict_cqcc_windows(cnn_cqcc,paths,mu,std)
        for r in part:
            r2=dict(r)
            r2['cqcc_probs']=np.asarray(pred[str(r['path'])]['window_probs'],np.float32)
            if len(r2['cqcc_probs'])!=len(r2['cnn_probs']) or not np.isfinite(r2['cqcc_probs']).all():
                raise RuntimeError('New CQCC probabilities misaligned with original windows')
            combined.append(r2)
        del cnn_cqcc
    # Keep source OOF record ordering identical to old experiment for paired comparisons.
    keyed={str(r['path']):r for r in combined}
    ordered=[keyed[str(r['path'])] for r in original]
    v27.verify_oof(ordered,{'version':v25.VERSION})
    if OOF.exists():raise RuntimeError(f'Refusing overwrite existing experimental OOF: {OOF}')
    ROOT.mkdir(parents=True,exist_ok=True)
    np.save(OOF,dict(version=VERSION,records=np.asarray(ordered,dtype=object),
                     source_v25_oof=str(SOURCE_OOF),
                     source_v25_oof_sha256=sha256(SOURCE_OOF),
                     paper_fold_checkpoint_sha256=cv_hashes,
                     cqcc_cache_version=DEV_SIDE_VERSION,
                     description='Unchanged v25 CNN-GRU/F0 OOF predictions + new speaker-out-of-fold paper-inspired CQCC predictions'),
            allow_pickle=True)
    print(f'Saved isolated experimental OOF: {OOF}')
    return load_experimental_oof()


def compare_development():
    old_file=Path(cfg.MODELS_DIR)/'Prediction-Fusion'/'CNN-GRU-CQCC-F0-F-v28-Temporal'/'development_oof_v28.npy'
    old_model=Path(cfg.MODELS_DIR)/'Prediction-Fusion'/'CNN-GRU-CQCC-F0-F-v28-Temporal'/'fusion_model_v28_temporal.npy'
    if not DEV_OOF.is_file():return
    new=np.load(DEV_OOF,allow_pickle=True).item()
    current_p=np.asarray(new['probabilities'],float)
    y=np.asarray(new['labels'],int)
    cond=np.asarray(new['conditions'],str)
    new_th=float(np.load(META,allow_pickle=True).item()['threshold'])
    def show(tag,p,t):
        clean=cond=='clean';degraded=~clean
        print(f'  {tag:<19} thr={t:.3f} clean F1={win.calculate_metrics(y[clean],p[clean],t)["f1"]:.4f} '
              f'degraded F1={win.calculate_metrics(y[degraded],p[degraded],t)["f1"]:.4f} '
              f'overall AUC={win.calculate_metrics(y,p,t)["auc"]:.4f}')
    print('\nMatched speaker-grouped DEVELOPMENT OOF comparison:')
    show('Paper-inspired CQCC',current_p,new_th)
    if old_file.is_file() and old_model.is_file():
        old=np.load(old_file,allow_pickle=True).item()
        if not (np.array_equal(old['labels'],new['labels']) and
                np.array_equal(old['speakers'],new['speakers']) and
                np.array_equal(old['conditions'],new['conditions'])):
            print('  Original v28 OOF order differs; skipping unmatched comparison')
            return
        old_th=float(np.load(old_model,allow_pickle=True).item()['threshold'])
        show('Original v28',np.asarray(old['probabilities'],float),old_th)
    else:
        print('  Original v28 OOF not found; new OOF is still available')


def fit():
    if META.exists() or STATE.exists() or DEV_OOF.exists():
        raise RuntimeError('Experimental fusion artifacts already exist; refusing overwrite. Make a new named experiment if refitting.')
    if not OOF.exists():
        raise FileNotFoundError('Run --stage oof before --stage fit')
    ROOT.mkdir(parents=True,exist_ok=True)
    v28.fit_model()
    compare_development()
    print('\nNEXT: review development results. Only if warranted:')
    print('  python train_akinci_cqcc_expert.py --stage final')
    print('  python prepare_akinci_cqcc_heldout.py --stage build --split both --workers 2')
    print('  python akinci_temporal_fusion_trial.py --stage evaluate')


def read_new_artifact():
    return v28.load_artifact()


def load_new_net(artifact):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=v28.TemporalResidual().to(device)
    ckpt=torch.load(STATE,map_location=device,weights_only=True)
    if ckpt.get('version')!=VERSION or int(ckpt.get('final_epochs',-1))!=int(artifact['final_epochs']):
        raise RuntimeError('Mismatched new fusion final temporal weights')
    net.load_state_dict(ckpt['state_dict'],strict=True)
    net.eval()
    return net,device


def simple_dict(m):
    x={k:float(v) for k,v in m.items() if k!='cm'}
    cm=np.asarray(m['cm'],dtype=int)
    x.update(tn=int(cm[0,0]),fp=int(cm[0,1]),fn=int(cm[1,0]),tp=int(cm[1,1]))
    return x


def evaluate():
    print('WARNING: held-out set has informed previous model-selection decisions. Exploratory evaluation only.')
    if RESULT.exists():
        raise RuntimeError(f'Experimental held-out output already exists; refusing to overwrite: {RESULT}')
    preflight_checkpoint(final=True)
    artifact=read_new_artifact()
    for split in ('test','test_degraded'):
        files=base.get_split_files(split)
        if len(files)!=EXPECTED_TEST_N:
            raise RuntimeError(f'{split}: count mismatch {len(files)} expected {EXPECTED_TEST_N}')
        speakers={Path(p).parent.name for p in files}
        if not speakers.issubset(cfg.TEST_SPEAKERS) or speakers & set(cfg.FOLD_ASSIGNMENTS.keys()):
            raise RuntimeError(f'Non-test speakers in {split}')
        for i,path in enumerate(files,1):
            check_sidecar_pair(path)
            if i%5000==0:print(f'  checked {split}: {i:,}/{len(files):,} sidecars')
    net,device=load_new_net(artifact)
    rows=[];results=[]
    for split in ('test','test_degraded'):
        files=base.get_split_files(split)
        print(f'\nExperiment expert inference {split}: {len(files):,} clips')
        records=base.collect_expert_predictions(files,final=True)
        prob,delta=v28.inference(records,artifact,net,device)
        y=np.array([int(r['label']) for r in records],int)
        cond=np.array([str(r['condition']) for r in records],str)
        t=float(artifact['threshold'])
        if len(prob)!=EXPECTED_TEST_N or not np.isfinite(prob).all():
            raise RuntimeError('Incomplete held-out probabilities')
        metrics=win.calculate_metrics(y,prob,t)
        print(f'  {split}: threshold={t:.3f} Acc={metrics["accuracy"]:.4f} F1={metrics["f1"]:.4f} '
              f'Prec={metrics["precision"]:.4f} Rec={metrics["recall"]:.4f} AUC={metrics["auc"]:.4f} EER={metrics["eer"]:.4f}')
        print(f'   CM: {np.asarray(metrics["cm"]).tolist()} | mean|temporal delta|={np.mean(np.abs(delta)):.4f}')
        rows.append(dict(split=split,condition='overall',n=len(y),**simple_dict(metrics)))
        part={}
        if split=='test_degraded':
            for name in sorted(set(cond)):
                sel=cond==name
                met=win.calculate_metrics(y[sel],prob[sel],t)
                part[name]=simple_dict(met)
                rows.append(dict(split=split,condition=name,n=int(sel.sum()),**simple_dict(met)))
                print(f'   {name:<16} n={sel.sum():,} F1={met["f1"]:.4f} AUC={met["auc"]:.4f} '
                      f'TN/FP/FN/TP={simple_dict(met)["tn"]}/{simple_dict(met)["fp"]}/{simple_dict(met)["fn"]}/{simple_dict(met)["tp"]}')
        results.append(dict(split=split,threshold=t,metrics=simple_dict(metrics),conditions=part,
                            test_recordings=EXPECTED_TEST_N,mean_abs_temporal_correction=float(np.mean(np.abs(delta)))))
    RESULT_ROOT.mkdir(parents=True,exist_ok=True)
    np.save(RESULT,results,allow_pickle=True)
    jpath=RESULT_ROOT/'heldout_exploratory_metrics.json'
    cpath=RESULT_ROOT/'heldout_exploratory_metrics.csv'
    with open(jpath,'x',encoding='utf-8') as f:
        json.dump(dict(version=VERSION,exploratory_not_independent_holdout=True,
                       description='Original frozen held-out speakers, new CQCC and newly fitted development-only fusion',
                       results=results),f,indent=2)
    with open(cpath,'x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)
    print(f'\nExperimental outputs: {RESULT_ROOT}\nOriginal v28 checkpoints, original features and reports unchanged.')


def check():
    hashes=preflight_checkpoint(final=False)
    print('New CQCC CV checkpoints: OK for folds',list(hashes))
    print('Original v25 OOF source:',SOURCE_OOF)
    print('Original v25 OOF exists:',SOURCE_OOF.is_file())
    print('Experimental OOF exists:',OOF.is_file())
    print('New fusion model exists:',META.is_file())
    print('New CQCC final checkpoint exists:',(CQCC_DIR/'final_model.pt').is_file())
    print('Held-out sidecars are NOT accessed in check/OOF/fit.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['check','oof','fit','compare','evaluate'],default='check')
    args=ap.parse_args()
    patch_experiment()
    if args.stage=='check':check()
    elif args.stage=='oof':build_oof()
    elif args.stage=='fit':fit()
    elif args.stage=='compare':compare_development()
    else:evaluate()

if __name__=='__main__':main()
'''

def _get_fusion():
    _get_auc_expert()
    modules={}
    for name,src in (
        ('improvedCQCC_cqcc_f0_fusion_v19_calibrated',_SOURCE_improvedCQCC_cqcc_f0_fusion_v19_calibrated),
        ('improvedCQCC_cqcc_f0_fusion_v22_window_level',_SOURCE_improvedCQCC_cqcc_f0_fusion_v22_window_level),
        ('improvedCQCC_v25_fusion_window',_SOURCE_improvedCQCC_v25_fusion_window),
        ('improvedCQCC_v27_clip_optimized_fusion',_SOURCE_improvedCQCC_v27_clip_optimized_fusion),
        ('improvedCQCC_v28_temporal_evidence_fusion',_SOURCE_improvedCQCC_v28_temporal_evidence_fusion),
        ('improvedCQCC_akinci_fusion_impl',_SOURCE_improvedCQCC_akinci_fusion_impl),
    ):
        modules[name]=_compile_internal(name,src)
    # Legacy scripts are now namespaces within ONE source file.
    api=modules['improvedCQCC_akinci_fusion_impl']
    if not hasattr(api,'_original_patch_experiment'):
        api._original_patch_experiment=api.patch_experiment
        def patch_once():
            if getattr(api,'_patch_done',False):return
            api._original_patch_experiment()
            api._patch_done=True
        api.patch_experiment=patch_once
    return api,modules

def _source_oof_from_trained_models(api, modules):
    """Create aligned CNN-GRU/F0 OOF without training an obsolete CQCC expert."""
    v25=modules['improvedCQCC_v25_fusion_window']
    base=modules['improvedCQCC_cqcc_f0_fusion_v19_calibrated']
    orig=Path(v25.OOF_PATH)
    if orig.is_file():
        print('Reusing original reference OOF; preserves frozen CNN-GRU/F0 evidence:',orig)
        return orig
    # SOURCE_OOF (name only) is an implementation detail of the proven v28
    # trainer. The generated reference cache uses the NEW expert already.
    custom=api.ROOT/'aligned_cnn_f0_paper_cqcc_reference.npy'
    custom.parent.mkdir(parents=True, exist_ok=True)
    modules['improvedCQCC_v27_clip_optimized_fusion'].OOF_INPUT_PATH = str(custom)
    if custom.is_file():
        v25.OOF_PATH=str(custom);api.SOURCE_OOF=custom
        return custom
    api.patch_experiment()
    base.OOF_CACHE_PATH=str(custom)
    base.OOF_CACHE_VERSION=v25.VERSION
    base.VERSION=v25.VERSION
    v25.OOF_PATH=str(custom)
    api.SOURCE_OOF=custom
    api.preflight_checkpoint(final=False)
    print('Computing new aligned reference OOF from CNN-GRU, F0, and paper CQCC checkpoints.')
    print('Old CQCC training is NOT required.')
    result=base.collect_oof_predictions(rebuild_cache=False)
    if len(result)!=api.EXPECTED_DEV_N:
        raise RuntimeError(f'Misaligned full development OOF: {len(result)}')
    return custom

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',default='check',choices=['check','baseline-oof','oof','fit','compare','evaluate','self-test'])
    args=p.parse_args()
    api,mods=_get_fusion()
    if args.stage=='self-test':
        from improvedCQCC_features import paper_cqcc_frames,pool_by_centers
        import improvedCQCC_config as cfg
        x=np.zeros(cfg.SAMPLE_RATE,dtype=np.float32)
        y=pool_by_centers(x,paper_cqcc_frames,cfg.WIN_SAMPLES,cfg.HOP_SAMPLES)
        assert y.shape==(2,40) and np.isfinite(y).all()
        print('SELF-TEST PASSED: 40-D paper CQCC and embedded fusion modules')
        return
    if args.stage=='baseline-oof':
        _source_oof_from_trained_models(api,mods)
        return
    if args.stage in ('oof','fit'):
        # Prior saved reference OOF is reused unchanged if it exists. For
        # clean from-scratch runs a compatible reference OOF is built first.
        _source_oof_from_trained_models(api,mods)
    api.patch_experiment()
    if args.stage=='check':api.check()
    elif args.stage=='oof':api.build_oof()
    elif args.stage=='fit':api.fit()
    elif args.stage=='compare':api.compare_development()
    elif args.stage=='evaluate':api.evaluate()

if __name__=='__main__':main()
