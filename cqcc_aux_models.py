"""CQCC+FFV and CQCC+F0 comparison models.

The design follows the anti-spoofing motivation of Pal, Paul & Saha (2018):
CQCC captures spectral/cepstral manipulation artifacts while FFV contributes
complementary pitch-variation evidence.  To make CQCC+F0 a controlled
comparison, both systems use the SAME CQCC branch, SAME fusion architecture,
training schedule, folds and threshold; only the auxiliary feature changes.

For runtime, both branches consume compact cached vectors (40-D CQCC + 14-D
FFV or 8-D F0).  There is no CNN/GRU in these two ablation models.
"""

import argparse
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from tqdm import tqdm

from config import (
    FEATURES_DIR, MODELS_DIR, RESULTS_DIR, FOLD_ASSIGNMENTS, SEED,
    CQCC_VECTOR_DIM, FFV_VECTOR_DIM, F0_VECTOR_DIM,
    LEARNING_RATE, WEIGHT_DECAY, LR_FACTOR,
    MAX_TRAIN_WINDOWS_PER_FILE,
    AUX_PAIR_BATCH_SIZE, AUX_PAIR_HIDDEN, AUX_PAIR_DROPOUT,
    AUX_PAIR_MAX_EPOCHS, AUX_PAIR_PATIENCE,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
VERSION = "v14_cqcc_aux_compact_fusion"
DECISION_THRESHOLD = 0.50

MODEL_SPECS = {
    "CQCC-FFV": ("ffv", FFV_VECTOR_DIM),
    "CQCC-F0": ("f0", F0_VECTOR_DIM),
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True is faster for stable tensor shapes. Seeded data/model
    # initialization still provides practical reproducibility for this thesis.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


class PairDataset(Dataset):
    def __init__(self, files, aux_name, cq_mean, cq_std, aux_mean, aux_std,
                 max_windows=None, desc="data"):
        cq, aux, y, paths = [], [], [], []
        seen = kept = 0
        for path in tqdm(files, desc=f"Loading {desc}", leave=False):
            try:
                with np.load(path, allow_pickle=True) as d:
                    a = np.asarray(d["cqcc"], dtype=np.float32)
                    b = np.asarray(d[aux_name], dtype=np.float32)
                    n = min(len(a), len(b))
                    if n == 0:
                        continue
                    if max_windows is not None and n > max_windows:
                        idx = np.linspace(0, n - 1, max_windows, dtype=np.int64)
                    else:
                        idx = np.arange(n, dtype=np.int64)
                    cq.append((a[idx] - cq_mean) / cq_std)
                    aux.append((b[idx] - aux_mean) / aux_std)
                    label = int(d["label"])
                    y.extend([label] * len(idx))
                    paths.extend([path] * len(idx))
                    seen += n
                    kept += len(idx)
            except Exception as exc:
                print(f"WARNING: {os.path.basename(path)}: {exc}")
        if not cq:
            raise RuntimeError(f"No valid {aux_name} feature data for {desc}.")
        self.cq = torch.from_numpy(np.concatenate(cq)).float()
        self.aux = torch.from_numpy(np.concatenate(aux)).float()
        self.y = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)
        self.paths = paths
        print(f"  {desc}: kept {kept:,}/{seen:,} windows; {len(self.y):,} in RAM")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.cq[i], self.aux[i], self.y[i], self.paths[i]


class CompactBranch(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        hidden = max(out_dim * 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CQCCAuxFusion(nn.Module):
    """Small two-branch gated fusion network.

    Separate encoders prevent the low-dimensional prosody branch from being
    numerically swamped by CQCC. Learnable positive gates let validation learn
    how much each representation contributes without adding a heavy backbone.
    """
    def __init__(self, aux_dim, hidden=AUX_PAIR_HIDDEN, dropout=AUX_PAIR_DROPOUT):
        super().__init__()
        cq_emb = hidden
        aux_emb = max(hidden // 2, 24)
        self.cqcc_branch = CompactBranch(CQCC_VECTOR_DIM, cq_emb, dropout)
        self.aux_branch = CompactBranch(aux_dim, aux_emb, dropout)
        self.cqcc_gate_raw = nn.Parameter(torch.tensor(0.0))
        self.aux_gate_raw = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Sequential(
            nn.LayerNorm(cq_emb + aux_emb),
            nn.Linear(cq_emb + aux_emb, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden // 2, 24)),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(max(hidden // 2, 24), 1),
        )

    def forward(self, cqcc, aux):
        # softplus keeps gates positive and smooth; +eps avoids a dead branch.
        g_cq = torch.nn.functional.softplus(self.cqcc_gate_raw) + 1e-4
        g_ax = torch.nn.functional.softplus(self.aux_gate_raw) + 1e-4
        z = torch.cat([g_cq * self.cqcc_branch(cqcc),
                       g_ax * self.aux_branch(aux)], dim=1)
        return self.classifier(z)

    def fusion_weights(self):
        with torch.no_grad():
            a = float(torch.nn.functional.softplus(self.cqcc_gate_raw).cpu())
            b = float(torch.nn.functional.softplus(self.aux_gate_raw).cpu())
            total = a + b + 1e-12
            return {"cqcc": a / total, "aux": b / total}


def all_train_files():
    return sorted(glob.glob(os.path.join(FEATURES_DIR, "train", "**", "*.npz"), recursive=True))


def split_files(name):
    return sorted(glob.glob(os.path.join(FEATURES_DIR, name, "**", "*.npz"), recursive=True))


def fold_files(files, fold):
    tr, va = [], []
    for p in files:
        spk = os.path.basename(os.path.dirname(p))
        f = FOLD_ASSIGNMENTS.get(spk)
        if f is None:
            continue
        (va if f == fold else tr).append(p)
    return tr, va


def compute_stats(files, aux_name):
    # Streaming sums avoid concatenating every cached window just to normalize.
    sums = {"cqcc": None, aux_name: None}
    sums2 = {"cqcc": None, aux_name: None}
    count = 0
    for p in tqdm(files, desc="Normalization stats", leave=False):
        with np.load(p, allow_pickle=True) as d:
            a = np.asarray(d["cqcc"], dtype=np.float64)
            b = np.asarray(d[aux_name], dtype=np.float64)
            n = min(len(a), len(b))
            if n == 0:
                continue
            a, b = a[:n], b[:n]
            if sums["cqcc"] is None:
                sums["cqcc"] = a.sum(0); sums2["cqcc"] = (a*a).sum(0)
                sums[aux_name] = b.sum(0); sums2[aux_name] = (b*b).sum(0)
            else:
                sums["cqcc"] += a.sum(0); sums2["cqcc"] += (a*a).sum(0)
                sums[aux_name] += b.sum(0); sums2[aux_name] += (b*b).sum(0)
            count += n
    if count == 0:
        raise RuntimeError("No feature windows available for normalization.")
    def finish(key):
        mean = sums[key] / count
        var = np.maximum(sums2[key] / count - mean*mean, 1e-8)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)
    return (*finish("cqcc"), *finish(aux_name))


def loader(ds, shuffle, seed):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=AUX_PAIR_BATCH_SIZE, shuffle=shuffle,
        num_workers=0, pin_memory=DEVICE.type == "cuda", generator=gen,
    )


def scaler_for_amp():
    if not USE_AMP:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def run_epoch(model, dl, optimizer, criterion, scaler):
    model.train()
    total = 0.0
    for cq, ax, y, _ in dl:
        cq = cq.to(DEVICE, non_blocking=True)
        ax = ax.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(cq, ax)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(cq, ax)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach())
    return total / max(len(dl), 1)


def clip_predictions(model, dl, criterion=None):
    model.eval()
    probs_by_path, labels_by_path = {}, {}
    total_loss = 0.0
    with torch.inference_mode():
        for cq, ax, y, paths in dl:
            cq = cq.to(DEVICE, non_blocking=True)
            ax = ax.to(DEVICE, non_blocking=True)
            ydev = y.to(DEVICE, non_blocking=True)
            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(cq, ax)
                    loss = criterion(logits, ydev) if criterion else None
            else:
                logits = model(cq, ax)
                loss = criterion(logits, ydev) if criterion else None
            if loss is not None:
                total_loss += float(loss)
            p = torch.sigmoid(logits).float().cpu().numpy().ravel()
            yy = y.numpy().astype(int).ravel()
            for path, score, lab in zip(paths, p, yy):
                probs_by_path.setdefault(path, []).append(float(score))
                labels_by_path[path] = int(lab)
    paths = list(probs_by_path)
    probs = np.asarray([np.mean(probs_by_path[p]) for p in paths], dtype=np.float64)
    labels = np.asarray([labels_by_path[p] for p in paths], dtype=np.int64)
    return total_loss / max(len(dl), 1), labels, probs, paths


def metrics(labels, probs, threshold=DECISION_THRESHOLD):
    pred = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        fnr = 1.0 - tpr
        i = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[i] + fnr[i]) / 2.0)
    else:
        eer = 0.0
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "auc": auc, "eer": eer,
        "cm": confusion_matrix(labels, pred),
    }


def train_cv(model_name):
    aux_name, aux_dim = MODEL_SPECS[model_name]
    files = all_train_files()
    if not files:
        raise RuntimeError("No prepared training caches. Run prepare_dataset.py first.")
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    results_path = os.path.join(model_dir, "fold_results.npy")
    results = []

    print(f"\n=== {model_name} | device={DEVICE} ===")
    for fold in (1, 2, 3, 4):
        tr_files, va_files = fold_files(files, fold)
        cq_mean, cq_std, ax_mean, ax_std = compute_stats(tr_files, aux_name)
        tr_ds = PairDataset(tr_files, aux_name, cq_mean, cq_std, ax_mean, ax_std,
                            MAX_TRAIN_WINDOWS_PER_FILE, f"{model_name} fold {fold} train")
        va_ds = PairDataset(va_files, aux_name, cq_mean, cq_std, ax_mean, ax_std,
                            None, f"{model_name} fold {fold} val")
        tr_dl = loader(tr_ds, True, SEED + fold)
        va_dl = loader(va_ds, False, SEED + fold)

        set_seed(SEED + fold)
        model = CQCCAuxFusion(aux_dim).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_FACTOR, patience=2, min_lr=1e-7
        )
        scaler = scaler_for_amp()
        best_f1, best_epoch, stale = -1.0, 1, 0
        best_path = os.path.join(model_dir, f"best_fold{fold}.pt")

        for epoch in range(1, AUX_PAIR_MAX_EPOCHS + 1):
            t0 = time.time()
            train_loss = run_epoch(model, tr_dl, optimizer, criterion, scaler)
            val_loss, y, p, _ = clip_predictions(model, va_dl, criterion)
            m = metrics(y, p)
            scheduler.step(val_loss)
            if m["f1"] > best_f1 + 1e-6:
                best_f1, best_epoch, stale = m["f1"], epoch, 0
                torch.save(model.state_dict(), best_path)
            else:
                stale += 1
            print(
                f"Fold {fold} Epoch {epoch:02d} | {time.time()-t0:.1f}s | "
                f"loss {train_loss:.4f}/{val_loss:.4f} | "
                f"clip F1 {m['f1']:.4f} AUC {m['auc']:.4f} | ES {stale}/{AUX_PAIR_PATIENCE}"
            )
            if stale >= AUX_PAIR_PATIENCE:
                break

        model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
        _, y, p, _ = clip_predictions(model, va_dl)
        m = metrics(y, p)
        results.append({
            "training_version": VERSION, "fold": fold, "best_epoch": best_epoch,
            "metrics": m, "cq_mean": cq_mean, "cq_std": cq_std,
            "aux_mean": ax_mean, "aux_std": ax_std,
            "fusion_weights": model.fusion_weights(),
        })
        np.save(results_path, results, allow_pickle=True)
        del tr_ds, va_ds, tr_dl, va_dl, model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print(f"{model_name} median best epoch: {int(np.median([r['best_epoch'] for r in results]))}")
    return results


def retrain_final(model_name):
    aux_name, aux_dim = MODEL_SPECS[model_name]
    model_dir = os.path.join(MODELS_DIR, model_name)
    results_path = os.path.join(model_dir, "fold_results.npy")
    if not os.path.exists(results_path):
        raise RuntimeError(f"Run CV first for {model_name}.")
    results = list(np.load(results_path, allow_pickle=True))
    if not results or any(r.get("training_version") != VERSION for r in results):
        raise RuntimeError(f"Outdated CV results for {model_name}; retrain CV.")
    epochs = max(1, int(np.median([r["best_epoch"] for r in results])))
    files = all_train_files()
    cq_mean, cq_std, ax_mean, ax_std = compute_stats(files, aux_name)
    ds = PairDataset(files, aux_name, cq_mean, cq_std, ax_mean, ax_std,
                     MAX_TRAIN_WINDOWS_PER_FILE, f"{model_name} final")
    dl = loader(ds, True, SEED + 99)
    set_seed(SEED + 99)
    model = CQCCAuxFusion(aux_dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()
    scaler = scaler_for_amp()
    print(f"\nRetraining {model_name} for {epochs} epoch(s)...")
    for epoch in range(1, epochs + 1):
        t0 = time.time(); loss = run_epoch(model, dl, opt, crit, scaler)
        print(f"  Epoch {epoch:02d}/{epochs}: loss {loss:.4f} | {time.time()-t0:.1f}s")
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    np.save(os.path.join(model_dir, "final_norm_stats.npy"), {
        "training_version": VERSION, "aux_name": aux_name,
        "cq_mean": cq_mean, "cq_std": cq_std,
        "aux_mean": ax_mean, "aux_std": ax_std,
        "epochs": epochs, "fusion_weights": model.fusion_weights(),
    }, allow_pickle=True)
    return model


def evaluate_final(model_name, split):
    aux_name, aux_dim = MODEL_SPECS[model_name]
    model_dir = os.path.join(MODELS_DIR, model_name)
    stats_path = os.path.join(model_dir, "final_norm_stats.npy")
    model_path = os.path.join(model_dir, "final_model.pt")
    if not (os.path.exists(stats_path) and os.path.exists(model_path)):
        raise RuntimeError(f"Missing final {model_name}; run --stage final or --stage all.")
    st = np.load(stats_path, allow_pickle=True).item()
    files = split_files(split)
    if not files:
        raise RuntimeError(f"No feature files in split '{split}'.")
    ds = PairDataset(files, aux_name, st["cq_mean"], st["cq_std"],
                     st["aux_mean"], st["aux_std"], None, f"{model_name} {split}")
    dl = loader(ds, False, SEED)
    model = CQCCAuxFusion(aux_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    _, y, p, paths = clip_predictions(model, dl)
    m = metrics(y, p)
    # Per-condition metrics for degraded test, without changing assignment.
    by_condition = {}
    if split == "test_degraded":
        conditions = []
        for path in paths:
            with np.load(path, allow_pickle=True) as d:
                conditions.append(str(d["condition"].item()))
        conditions = np.asarray(conditions)
        for c in sorted(set(conditions)):
            mask = conditions == c
            by_condition[c] = metrics(y[mask], p[mask])
    result = {
        "training_version": VERSION, "model": model_name, "split": split,
        "metrics": m, "by_condition": by_condition,
        "fusion_weights": model.fusion_weights(),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, f"{model_name.lower()}_{split}.npy"), result, allow_pickle=True)
    print(f"\n{model_name} — {split}")
    print(f"  Accuracy {m['accuracy']:.4f} | Precision {m['precision']:.4f} | Recall {m['recall']:.4f}")
    print(f"  F1 {m['f1']:.4f} | AUC {m['auc']:.4f} | EER {m['eer']:.4f}")
    print(f"  Fusion weights: {model.fusion_weights()}")
    if by_condition:
        print("  Degraded F1 by condition:")
        for c, cm in by_condition.items():
            print(f"    {c:<20} {cm['f1']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["CQCC-FFV", "CQCC-F0", "all"], default="all")
    parser.add_argument("--stage", choices=["cv", "final", "eval", "all"], default="all")
    args = parser.parse_args()
    set_seed(SEED)
    names = list(MODEL_SPECS) if args.model == "all" else [args.model]
    for name in names:
        if args.stage in ("cv", "all"):
            train_cv(name)
        if args.stage in ("final", "all"):
            retrain_final(name)
        if args.stage in ("eval", "all"):
            evaluate_final(name, "test")
            if split_files("test_degraded"):
                evaluate_final(name, "test_degraded")


if __name__ == "__main__":
    main()
