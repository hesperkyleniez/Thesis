# retrain.py
# Retrains each model from scratch on all development/training data after 4-fold CV.
# The held-out test speakers are never used here.
#
# Final retraining changes:
# - Uses the MEDIAN best epoch across the 4 CV folds as the training budget.
# - Keeps the AdamW settings used by train.py.
# - Uses AMP on CUDA, matching train.py.
# - Uses gradient clipping, matching train.py.
# - Uses an adaptive ReduceLROnPlateau scheduler driven by training loss because
#   the final model is trained on all development data and has no validation set.
# - Uses the same DataLoader behavior as train.py (num_workers=0).
# - Saves final normalization statistics and a training history for traceability.

import os
import sys
import time
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Keep this import path consistent with the existing project structure.
sys.path.insert(0, r"D:\Thesis\code")

from config import (
    FEATURES_DIR,
    MODELS_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DROPOUT,
    SEED,
    LR_FACTOR,
    MAX_TRAIN_WINDOWS_PER_FILE,
    RETRAIN_EPOCH_POLICY,
)
from train import (
    CNNOnly,
    CNNGRUOnly,
    CNNGRUFusion,
    HybridDataset,
    collate_with_paths,
    set_seed,
    USE_AMP,
    gate_entropy_penalty,
    build_optimizer,
    compute_fold_norm_stats,
    TRAINING_VERSION,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CV_FOLDS = 4
SCHEDULER_PATIENCE = 3
MIN_LR = 1e-7
GRAD_CLIP_NORM = 1.0

MODELS = {
    "CNN-only": CNNOnly,
    "CNN-GRU": CNNGRUOnly,
    "CNN-GRU-F": CNNGRUFusion,
}


def get_all_train_files():
    files = glob.glob(
        os.path.join(FEATURES_DIR, "train", "**", "*.npz"),
        recursive=True,
    )
    files.sort()
    return files


def load_cv_results(model_name):
    """Load completed 4-fold CV results for one model."""
    results_path = os.path.join(
        MODELS_DIR,
        model_name,
        "fold_results.npy",
    )

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"CV results not found for {model_name}: {results_path}\n"
            "Complete all 4 CV folds before retraining."
        )

    results = list(np.load(results_path, allow_pickle=True))
    results.sort(key=lambda r: int(r["fold"]))

    if len(results) != MAX_CV_FOLDS:
        raise RuntimeError(
            f"{model_name} has {len(results)} completed CV folds. "
            f"Exactly {MAX_CV_FOLDS} folds are required before retraining."
        )

    expected_folds = list(range(1, MAX_CV_FOLDS + 1))
    actual_folds = [int(r["fold"]) for r in results]

    if actual_folds != expected_folds:
        raise RuntimeError(
            f"Incomplete or invalid fold results for {model_name}: {actual_folds}. "
            f"Expected {expected_folds}."
        )

    return results


def get_target_epoch(model_name):
    """Choose the final training budget from the CV best epochs."""
    results = load_cv_results(model_name)

    epochs = []
    for result in results:
        if "best_epoch" not in result:
            raise RuntimeError(
                f"Missing best_epoch in fold {result['fold']} for {model_name}."
            )
        epochs.append(int(result["best_epoch"]))

    if RETRAIN_EPOCH_POLICY == "median":
        target_epochs = max(1, int(np.median(epochs)))
    elif RETRAIN_EPOCH_POLICY == "mean":
        target_epochs = max(1, int(round(np.mean(epochs))))
    else:
        target_epochs = max(epochs)

    print(f"  CV best epochs per fold: {epochs}")
    print(f"  {RETRAIN_EPOCH_POLICY.capitalize()} best epoch: {target_epochs}")

    return target_epochs


def make_scaler():
    if not USE_AMP:
        return None

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def train_epoch(model, loader, optimizer, criterion, scaler):
    """Train one epoch using the same AMP/gradient-clipping behavior as train.py."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for logmel, mfcc, f0, labels, _ in loader:
        logmel = logmel.to(DEVICE, non_blocking=True)
        mfcc = mfcc.to(DEVICE, non_blocking=True)
        f0 = f0.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(logmel, mfcc, f0)
                loss = criterion(logits, labels)
                loss = loss + gate_entropy_penalty(model)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(logmel, mfcc, f0)
            loss = criterion(logits, labels)
            loss = loss + gate_entropy_penalty(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += loss.detach().item()
        preds = (torch.sigmoid(logits.detach()) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return (
        total_loss / max(len(loader), 1),
        correct / max(total, 1),
    )


def retrain_model(model_name, model_class):
    print(f"\n{'=' * 70}")
    print(f"  FINAL RETRAINING: {model_name}")
    print(f"{'=' * 70}")

    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, "final_model.pt")
    norm_stats_path = os.path.join(model_dir, "final_norm_stats.npy")
    history_path = os.path.join(model_dir, "final_training_history.npy")

    if os.path.exists(final_model_path) and os.path.exists(norm_stats_path):
        try:
            existing_norm = np.load(norm_stats_path, allow_pickle=True).item()
            if existing_norm.get("training_version") == TRAINING_VERSION:
                print("  final_model.pt already exists for this training version — skipping")
                print(f"  Delete this file if you intentionally want to retrain: {final_model_path}")
                return
            else:
                print(
                    f"  Existing final_model.pt is from an older training version "
                    f"({existing_norm.get('training_version')!r} != {TRAINING_VERSION!r}) "
                    f"— retraining."
                )
        except Exception as exc:
            print(f"  Could not verify existing final_model.pt version ({exc}) — retraining to be safe.")
    elif os.path.exists(final_model_path):
        print("  Existing final_model.pt has no version stamp — retraining to be safe.")

    target_epochs = get_target_epoch(model_name)

    all_files = get_all_train_files()
    if not all_files:
        raise RuntimeError(
            f"No training feature files found under: "
            f"{os.path.join(FEATURES_DIR, 'train')}"
        )

    print(f"  Total development files: {len(all_files):,}")

    mfcc_mean, mfcc_std, f0_speaker_stats, f0_fallback = compute_fold_norm_stats(
        all_files
    )

    np.save(
        norm_stats_path,
        {
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std,
            "f0_speaker_stats": f0_speaker_stats,
            "f0_fallback": f0_fallback,
            "training_version": TRAINING_VERSION,
        },
    )
    print(f"  Normalization stats saved: {norm_stats_path}")

    print("  Loading development windows into RAM...")
    train_ds = HybridDataset(
        all_files,
        mfcc_mean,
        mfcc_std,
        f0_speaker_stats,
        f0_fallback,
        desc="Full Train",
        max_windows=MAX_TRAIN_WINDOWS_PER_FILE,
        seed=SEED,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    # Match train.py's DataLoader behavior.
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        generator=generator,
        collate_fn=collate_with_paths,
    )

    print(f"  Training windows: {len(train_ds):,}")
    print(f"  Target epochs    : {target_epochs}")
    print(f"  Initial LR       : {LEARNING_RATE:.2e}")
    print(f"  Weight decay     : {WEIGHT_DECAY:.2e}")
    print(f"  LR factor        : {LR_FACTOR}")
    print(f"  LR patience      : {SCHEDULER_PATIENCE}")
    print(f"  Minimum LR       : {MIN_LR:.2e}")
    print(f"  AMP              : {USE_AMP}")
    print()

    # Fresh model and optimizer, exactly as required for final retraining.
    model = model_class(dropout=DROPOUT).to(DEVICE)

    optimizer = build_optimizer(model)

    criterion = nn.BCEWithLogitsLoss()

    # The final model has no validation set because all development speakers
    # are used for training. Therefore, ReduceLROnPlateau is driven by the
    # training loss rather than validation loss.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    scaler = make_scaler()
    history = []

    total_start = time.time()

    for epoch in range(1, target_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
        )

        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "learning_rate": float(current_lr),
            }
        )

        print(
            f"  Epoch {epoch:02d}/{target_epochs} | "
            f"Train {train_loss:.4f}/{train_acc:.4f} | "
            f"LR {current_lr:.2e} | "
            f"{elapsed:.0f}s"
        )

    total_elapsed = time.time() - total_start

    # Save the final model at the end of the complete CV-derived training budget.
    torch.save(model.state_dict(), final_model_path)
    np.save(history_path, history)

    print(f"\n  Final model saved: {final_model_path}")
    print(f"  Training history saved: {history_path}")
    print(f"  Total retraining time: {total_elapsed / 60:.1f} minutes")

    if hasattr(model, "get_fusion_weights"):
        weights = model.get_fusion_weights()
        if weights:
            print("  Final fusion weights:")
            for key, value in weights.items():
                print(f"    {key}: {value:.4f}")

    del train_ds, train_loader, model, optimizer, scheduler
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def main():
    print(f"Device: {DEVICE}")
    print(
        f"GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}"
    )

    set_seed(SEED)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["CNN-only", "CNN-GRU", "CNN-GRU-F", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.model == "all":
        model_list = list(MODELS.items())
    else:
        model_list = [(args.model, MODELS[args.model])]

    total_start = time.time()

    for model_name, model_class in model_list:
        retrain_model(model_name, model_class)

    total_elapsed = (time.time() - total_start) / 3600

    print(f"\n{'=' * 70}")
    print(f"  Final retraining complete in {total_elapsed:.2f} hours")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
