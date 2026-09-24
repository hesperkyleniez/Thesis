# aux_train.py
# Speaker-grouped 4-fold CV for MFCC, CQCC, and F0 auxiliary classifiers.
# These models are intentionally independent from the CNN-GRU so their final
# probabilities can be combined with prediction-level late fusion.

import os
import sys
import time
import glob
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, r"D:\Thesis\code")
from config import (
    FEATURES_DIR, MODELS_DIR, FOLD_ASSIGNMENTS,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DROPOUT, SEED,
    AUX_BATCH_SIZE, AUX_HIDDEN, AUX_DROPOUT, AUX_MAX_EPOCHS, AUX_PATIENCE,
    LR_FACTOR, MAX_TRAIN_WINDOWS_PER_FILE,
    RETRAIN_EPOCH_POLICY,
)

AUX_VERSION = "v9_prediction_fusion_aux"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

# Safe CUDA throughput setting. Reproducibility remains controlled by the seed.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


class AuxDataset(Dataset):
    """Loads only one compact auxiliary representation into RAM."""

    def __init__(self, files, feature_name, mean, std, max_windows=None, seed=SEED):
        self.x = []
        self.y = []
        self.paths = []

        total_seen = 0
        total_kept = 0

        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)

        for path in tqdm(files, desc=f"Loading {feature_name}", leave=False):
            try:
                with np.load(path, allow_pickle=True) as d:
                    arr = np.asarray(d[feature_name], dtype=np.float32)
                    label = int(d["label"])
                    n = len(arr)

                    if max_windows is not None and n > max_windows:
                        # Same deterministic coverage cap used by train.py.
                        idx = np.linspace(0, n - 1, max_windows, dtype=np.int64)
                    else:
                        idx = np.arange(n)

                    arr = arr[idx]
                    arr = (arr - mean) / std

                    self.x.append(arr)
                    self.y.extend([label] * len(arr))
                    self.paths.extend([path] * len(arr))

                    total_seen += n
                    total_kept += len(arr)

            except Exception as exc:
                print(f"WARNING: {os.path.basename(path)}: {exc}")

        if not self.x:
            raise RuntimeError(f"No valid {feature_name} data found.")

        self.x = torch.as_tensor(np.concatenate(self.x, axis=0), dtype=torch.float32)
        self.y = torch.as_tensor(self.y, dtype=torch.float32).unsqueeze(1)

        print(
            f"  {feature_name}: kept {total_kept:,}/{total_seen:,} windows "
            f"({100*total_kept/max(total_seen,1):.1f}%)"
        )
        print(f"  {feature_name}: {len(self.y):,} windows loaded into RAM")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.paths[idx]


class AuxMLP(nn.Module):
    def __init__(self, input_dim, hidden=AUX_HIDDEN, dropout=AUX_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(max(hidden // 2, 16), 1),
        )

    def forward(self, x):
        return self.net(x)


AUX_SPECS = {
    "MFCC": {"feature": "mfcc", "dim": 80},
    "CQCC": {"feature": "cqcc", "dim": 80},
    "F0": {"feature": "f0", "dim": 2},
}


def get_all_train_files():
    return sorted(
        glob.glob(
            os.path.join(FEATURES_DIR, "train", "**", "*.npz"),
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


def compute_feature_stats(files, feature_name):
    arrays = []
    for path in tqdm(files, desc=f"Stats {feature_name}", leave=False):
        try:
            with np.load(path, allow_pickle=True) as d:
                arrays.append(np.asarray(d[feature_name], dtype=np.float32))
        except Exception as exc:
            print(f"WARNING: stats {os.path.basename(path)}: {exc}")

    if not arrays:
        raise RuntimeError(f"No {feature_name} arrays available.")

    cat = np.concatenate(arrays, axis=0)
    return cat.mean(axis=0), cat.std(axis=0) + 1e-8


def make_loader(ds, shuffle, generator=None):
    return DataLoader(
        ds,
        batch_size=AUX_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        generator=generator,
    )


def make_scaler():
    if not USE_AMP:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0

    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.detach().item())

    return total_loss / max(len(loader), 1)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    clip_probs = {}
    clip_labels = {}

    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            probs = torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
            labels = y.cpu().numpy().astype(int).reshape(-1)
            total_loss += float(loss.item())

            for path, prob, label in zip(paths, probs, labels):
                clip_probs.setdefault(path, []).append(float(prob))
                clip_labels[path] = int(label)

    scores = np.asarray(
        [np.mean(clip_probs[p]) for p in clip_probs], dtype=float
    )
    labels = np.asarray(
        [clip_labels[p] for p in clip_probs], dtype=int
    )
    preds = (scores >= 0.5).astype(int)

    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0

    return total_loss / max(len(loader), 1), f1, auc


def eer(labels, probs):
    if len(np.unique(labels)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def train_aux_model(name):
    spec = AUX_SPECS[name]
    feature = spec["feature"]
    model_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    results_path = os.path.join(model_dir, "fold_results.npy")

    files = get_all_train_files()
    results = []
    completed = set()

    if os.path.exists(results_path):
        try:
            old = list(np.load(results_path, allow_pickle=True))
            if old and all(r.get("training_version") == AUX_VERSION for r in old):
                results = old
                completed = {int(r["fold"]) for r in old}
        except Exception:
            pass

    for fold in [1, 2, 3, 4]:
        if fold in completed:
            print(f"{name}: skipping completed Fold {fold}")
            continue

        train_files, val_files = get_fold_files(files, fold)
        mean, std = compute_feature_stats(train_files, feature)

        train_ds = AuxDataset(
            train_files, feature, mean, std,
            max_windows=MAX_TRAIN_WINDOWS_PER_FILE,
            seed=SEED + fold,
        )
        val_ds = AuxDataset(
            val_files, feature, mean, std,
            max_windows=None,
            seed=SEED + fold,
        )

        gen = torch.Generator().manual_seed(SEED + fold)
        train_loader = make_loader(train_ds, True, gen)
        val_loader = make_loader(val_ds, False)

        model = AuxMLP(spec["dim"]).to(DEVICE)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_FACTOR,
            patience=3, min_lr=1e-7,
        )
        scaler = make_scaler()

        best_f1 = -1.0
        best_epoch = 1
        patience = 0
        best_path = os.path.join(model_dir, f"best_fold{fold}.pt")

        print(f"\n{name} — Fold {fold}")
        for epoch in range(1, AUX_MAX_EPOCHS + 1):
            start = time.time()
            loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_f1, val_auc = validate(model, val_loader, criterion)
            scheduler.step(val_loss)

            if val_f1 > best_f1 + 1e-6:
                best_f1 = val_f1
                best_epoch = epoch
                patience = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1

            print(
                f"  Epoch {epoch:02d}/{AUX_MAX_EPOCHS} "
                f"TrainLoss {loss:.4f} ValLoss {val_loss:.4f} "
                f"ClipF1 {val_f1:.4f} AUC {val_auc:.4f} "
                f"ES {patience}/{AUX_PATIENCE} "
                f"{time.time()-start:.0f}s"
            )

            if patience >= AUX_PATIENCE:
                break

        state = torch.load(best_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        _, _, val_auc = validate(model, val_loader, criterion)

        results.append({
            "training_version": AUX_VERSION,
            "fold": fold,
            "best_epoch": best_epoch,
            "best_f1": best_f1,
            "auc": val_auc,
            "mean": mean,
            "std": std,
        })
        results.sort(key=lambda r: int(r["fold"]))
        np.save(results_path, results)

        del train_ds, val_ds, train_loader, val_loader, model, optimizer
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{name} CV best epochs:", [r["best_epoch"] for r in results])
    print(f"{name} median best epoch:", int(np.median([r["best_epoch"] for r in results])))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(AUX_SPECS) + ["all"], default="all")
    args = parser.parse_args()

    set_seed(SEED)
    names = list(AUX_SPECS) if args.model == "all" else [args.model]
    for name in names:
        train_aux_model(name)
