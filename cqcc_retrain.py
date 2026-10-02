# cqcc_retrain.py
# Final retraining for the CQCC expert used by CNN-GRU + CQCC late fusion.

import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, r"D:\Thesis\code")
from config import (
    FEATURES_DIR, MODELS_DIR, LEARNING_RATE, WEIGHT_DECAY,
    SEED, LR_FACTOR, MAX_TRAIN_WINDOWS_PER_FILE, RETRAIN_EPOCH_POLICY,
    CQCC_BATCH_SIZE, CQCC_HIDDEN, CQCC_DROPOUT,
)
from cqcc_train import (
    CQCC_FEATURE, CQCC_DIM, CQCC_VERSION, CQCCDataset, CQCCMLP,
    get_all_train_files, make_loader, make_scaler, train_epoch,
    DEVICE, USE_AMP,
)


def get_target_epoch(name):
    path = os.path.join(MODELS_DIR, name, "fold_results.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CV results: {path}")

    results = list(np.load(path, allow_pickle=True))
    if len(results) != 4:
        raise RuntimeError(f"{name}: expected 4 CV folds, found {len(results)}")

    epochs = [int(r["best_epoch"]) for r in results]

    if RETRAIN_EPOCH_POLICY == "median":
        target = int(np.median(epochs))
    elif RETRAIN_EPOCH_POLICY == "mean":
        target = max(1, int(round(np.mean(epochs))))
    else:
        target = max(epochs)

    print(f"{name} CV best epochs: {epochs}")
    print(f"{name} retraining epoch budget ({RETRAIN_EPOCH_POLICY}): {target}")
    return target


def retrain_cqcc():
    name = "CQCC"
    feature = CQCC_FEATURE
    model_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    target_epochs = get_target_epoch(name)
    files = get_all_train_files()

    # Use development-wide statistics only. Test data is never used here.
    arrays = []
    for path in tqdm(files, desc=f"Stats {name}", leave=False):
        with np.load(path, allow_pickle=True) as d:
            arrays.append(np.asarray(d[feature], dtype=np.float32))
    cat = np.concatenate(arrays, axis=0)
    mean = cat.mean(axis=0)
    std = cat.std(axis=0) + 1e-8

    norm_path = os.path.join(model_dir, "final_norm_stats.npy")
    np.save(norm_path, {
        "mean": mean,
        "std": std,
        "training_version": CQCC_VERSION,
        "feature": feature,
    })

    ds = CQCCDataset(
        files, feature, mean, std,
        max_windows=MAX_TRAIN_WINDOWS_PER_FILE,
        seed=SEED,
    )
    loader = DataLoader(
        ds, batch_size=CQCC_BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    model = CQCCMLP(CQCC_DIM, CQCC_HIDDEN, CQCC_DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=3, min_lr=1e-7
    )
    scaler = make_scaler()

    history = []
    start_all = time.time()

    for epoch in range(1, target_epochs + 1):
        start = time.time()
        loss = train_epoch(model, loader, optimizer, criterion, scaler)
        scheduler.step(loss)
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": float(loss),
            "learning_rate": float(lr),
        })
        print(
            f"{name} Epoch {epoch:02d}/{target_epochs} "
            f"loss={loss:.4f} lr={lr:.2e} time={time.time()-start:.0f}s"
        )

    model_path = os.path.join(model_dir, "final_model.pt")
    hist_path = os.path.join(model_dir, "final_training_history.npy")
    torch.save(model.state_dict(), model_path)
    np.save(hist_path, history)

    print(f"{name}: saved {model_path}")
    print(f"{name}: total retrain time {(time.time()-start_all)/60:.1f} min")


if __name__ == "__main__":
    retrain_cqcc()
