# f0_train.py
# Speaker-grouped 4-fold CV for the F0 expert used only by the
# CNN-GRU + F0 prediction-level late-fusion model.

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
from improvedCQCC_robustness import macro_condition_f1

# The running script directory is already on sys.path; do not inject historical code paths.
from improvedCQCC_config import (
    FEATURES_DIR, MODELS_DIR, FOLD_ASSIGNMENTS,
    LEARNING_RATE, WEIGHT_DECAY, SEED,
    F0_BATCH_SIZE, F0_HIDDEN, F0_DROPOUT, F0_MAX_EPOCHS, F0_PATIENCE,
    F0_VECTOR_DIM,
    LR_FACTOR, MAX_TRAIN_WINDOWS_PER_FILE,
)

F0_VERSION = "v16_robust_all_f0_expert"


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


class F0Dataset(Dataset):
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


class F0MLP(nn.Module):
    def __init__(self, input_dim, hidden=F0_HIDDEN, dropout=F0_DROPOUT):
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


F0_FEATURE = "f0"
F0_DIM = F0_VECTOR_DIM


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
        batch_size=F0_BATCH_SIZE,
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
    paths = list(clip_probs.keys())
    cond_macro = macro_condition_f1(paths, labels, scores, threshold=0.5)

    return total_loss / max(len(loader), 1), f1, auc, cond_macro


def eer(labels, probs):
    if len(np.unique(labels)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def train_f0():
    name = "F0"
    feature = F0_FEATURE
    model_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    results_path = os.path.join(model_dir, "fold_results.npy")

    files = get_all_train_files()
    results = []
    completed = set()

    if os.path.exists(results_path):
        try:
            old = list(np.load(results_path, allow_pickle=True))
            if old and all(r.get("training_version") == F0_VERSION for r in old):
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

        train_ds = F0Dataset(
            train_files, feature, mean, std,
            max_windows=MAX_TRAIN_WINDOWS_PER_FILE,
            seed=SEED + fold,
        )
        val_ds = F0Dataset(
            val_files, feature, mean, std,
            max_windows=None,
            seed=SEED + fold,
        )

        gen = torch.Generator().manual_seed(SEED + fold)
        train_loader = make_loader(train_ds, True, gen)
        val_loader = make_loader(val_ds, False)

        model = F0MLP(F0_DIM).to(DEVICE)
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
        best_macro = -1.0
        best_epoch = 1
        patience = 0
        best_path = os.path.join(model_dir, f"best_fold{fold}.pt")

        print(f"\n{name} — Fold {fold}")
        for epoch in range(1, F0_MAX_EPOCHS + 1):
            start = time.time()
            loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_f1, val_auc, val_macro = validate(model, val_loader, criterion)
            scheduler.step(val_loss)

            if val_macro > best_macro + 1e-6:
                best_macro = val_macro
                best_f1 = val_f1
                best_epoch = epoch
                patience = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1

            print(
                f"  Epoch {epoch:02d}/{F0_MAX_EPOCHS} "
                f"TrainLoss {loss:.4f} ValLoss {val_loss:.4f} "
                f"ClipF1 {val_f1:.4f} CondMacroF1 {val_macro:.4f} AUC {val_auc:.4f} "
                f"ES {patience}/{F0_PATIENCE} "
                f"{time.time()-start:.0f}s"
            )

            if patience >= F0_PATIENCE:
                break

        state = torch.load(best_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        _, _, val_auc, _ = validate(model, val_loader, criterion)

        results.append({
            "training_version": F0_VERSION,
            "fold": fold,
            "best_epoch": best_epoch,
            "best_f1": best_f1,
            "best_condition_macro_f1": best_macro,
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


if False and __name__ == "__main__":
    set_seed(SEED)
    train_f0()


# Historical AUC-criterion F0 training used by the final proposed fusion.

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

_SOURCE_auc_f0 = r'''
"""v25 auxiliary-expert experiment: AUC-selected CQCC and F0 MLPs.

Run from the thesis code directory, after copying this script there:
    python v25_aux_train_auc.py --expert all --stage cv
    python v25_aux_train_auc.py --expert all --stage final

Only existing features/train/*.npz are read. Nothing is re-extracted or
augmented; condition maps and CNN-GRU checkpoints are never modified.

Reuses the exact architecture, normalization convention, optimizer,
window-level BCE objective, and per-clip window cap from cqcc_train.py and
f0_train.py. The experimental change is checkpoint selection: clip ROC-AUC
rather than threshold-dependent condition-macro F1. An initial 15-epoch
minimum run prevents an early patience cutoff while ranking improves.
"""

import argparse
import glob
import hashlib
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, log_loss, roc_auc_score)

from improvedCQCC_config import (FEATURES_DIR, MODELS_DIR, FOLD_ASSIGNMENTS,
                    LEARNING_RATE, WEIGHT_DECAY, SEED, LR_FACTOR,
                    MAX_TRAIN_WINDOWS_PER_FILE, RETRAIN_EPOCH_POLICY,
                    CQCC_HIDDEN, CQCC_DROPOUT, CQCC_MAX_EPOCHS,
                    F0_HIDDEN, F0_DROPOUT, F0_MAX_EPOCHS,
                    CQCC_VECTOR_DIM, F0_VECTOR_DIM)
from improvedCQCC_cqcc_train import CQCCMLP, CQCCDataset, cqcc_stats
from improvedCQCC_f0_train import F0MLP, F0Dataset

VERSION = 'v25_auc_selected_window_bce_v1'
MIN_EPOCHS = 15                 # At least this many epochs before early-stop.
PATIENCE = 10                   # Patience measured against clip-level ROC-AUC.
AUC_MIN_DELTA = 0.0001         # Tiny AUC changes count as meaningful.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXPERTS = {
    'cqcc': dict(name='CQCC', folder='CQCC-v25-AUC', dim=CQCC_VECTOR_DIM,
                 hidden=CQCC_HIDDEN, dropout=CQCC_DROPOUT,
                 epochs=CQCC_MAX_EPOCHS, batch_size=128,
                 model=CQCCMLP, dataset=CQCCDataset),
    'f0': dict(name='F0', folder='F0-v25-AUC', dim=F0_VECTOR_DIM,
               hidden=F0_HIDDEN, dropout=F0_DROPOUT,
               epochs=F0_MAX_EPOCHS, batch_size=128,
               model=F0MLP, dataset=F0Dataset),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def all_files():
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, 'train', '**', '*.npz'), recursive=True))
    if not files:
        raise RuntimeError(f'No development cache files in {FEATURES_DIR}\\train')
    speakers = Counter(os.path.basename(os.path.dirname(p)) for p in files)
    unknown = sorted(set(speakers) - set(FOLD_ASSIGNMENTS))
    if unknown:
        raise RuntimeError(f'Unassigned speakers found: {unknown}. Stopping to avoid dropping samples.')
    print(f'Development cache: {len(files):,} clips, {len(speakers)} speakers')
    return files


def preflight(files, feature, expected_dim):
    """No cache rewrite, only check a few files for shape/finiteness."""
    rng = np.random.default_rng(SEED)
    indices = sorted(set([0, len(files)-1] + rng.choice(len(files), min(100, len(files)), replace=False).tolist()))
    versions = Counter()
    for index in indices:
        with np.load(files[index], allow_pickle=False) as d:
            if feature not in d.files:
                raise RuntimeError(f'{files[index]} missing {feature}')
            arr = d[feature]
            if arr.ndim != 2 or arr.shape[1] != expected_dim or len(arr) == 0:
                raise RuntimeError(f'{files[index]} has {feature} shape {arr.shape}; expected (N,{expected_dim})')
            if not np.isfinite(arr).all():
                raise RuntimeError(f'{files[index]} contains NaN/Inf {feature} values')
            if 'cache_version' in d.files:
                versions[str(d['cache_version'])] += 1
    print(f'  Sampled {len(indices)} {feature} caches: dimension={expected_dim}, finite=True')
    print(f'  Cache versions in sample: {dict(versions)}')
    if len(versions) > 1:
        raise RuntimeError('Mixed cache versions detected. Freeze preprocessing before training.')


def prepare_loader(files, spec, mean, std, training, seed):
    ds = spec['dataset'](files, spec_key(spec), mean, std,
                         max_windows=MAX_TRAIN_WINDOWS_PER_FILE if training else None,
                         seed=seed)
    if not torch.isfinite(ds.x).all():
        raise RuntimeError('Nonfinite normalized auxiliary features detected.')
    from torch.utils.data import DataLoader
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=spec['batch_size'], shuffle=training,
                        num_workers=0, pin_memory=DEVICE.type == 'cuda',
                        generator=generator if training else None)
    return ds, loader


def spec_key(spec):
    return spec['name'].lower()


def make_model(spec):
    return spec['model'](spec['dim'], spec['hidden'], spec['dropout']).to(DEVICE)


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    weighted_loss = 0.0
    total = 0
    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
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
        weighted_loss += float(loss.detach()) * len(x)
        total += len(x)
    return weighted_loss / max(total, 1)


def validate(model, loader, cond_by_file):
    """Use mean WINDOW probability per clip, exactly as original validation."""
    model.eval()
    score_lists = {}
    labels = {}
    with torch.inference_mode():
        for x, y, paths in loader:
            x = x.to(DEVICE, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
            target = y.numpy().astype(int).reshape(-1)
            for path, prob, label in zip(paths, probs, target):
                score_lists.setdefault(path, []).append(float(prob))
                labels[path] = int(label)
    paths = sorted(score_lists)
    y = np.asarray([labels[p] for p in paths], dtype=int)
    p = np.asarray([np.mean(score_lists[path]) for path in paths], dtype=float)
    pred = (p >= 0.5).astype(int)
    if len(set(y)) < 2:
        raise RuntimeError('Validation labels lack one binary class; check speaker folds.')
    condition_metrics = {}
    for c in sorted(set(cond_by_file[p] for p in paths)):
        mask = np.asarray([cond_by_file[path] == c for path in paths])
        condition_metrics[c] = {
            'count': int(mask.sum()),
            'auc': float(roc_auc_score(y[mask], p[mask])) if len(np.unique(y[mask])) > 1 else None,
            'f1_at_0_5': float(f1_score(y[mask], pred[mask], zero_division=0)),
            'ai_prediction_rate': float(np.mean(pred[mask])),
        }
    return {
        'auc': float(roc_auc_score(y, p)),
        'clip_log_loss': float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0,1])),
        'f1_at_0_5': float(f1_score(y, pred, zero_division=0)),
        'accuracy_at_0_5': float(accuracy_score(y, pred)),
        'balanced_acc_at_0_5': float(balanced_accuracy_score(y, pred)),
        'ai_prediction_rate': float(np.mean(pred)),
        'ai_ground_truth_rate': float(np.mean(y)),
        'p05': float(np.quantile(p, 0.05)),
        'p50': float(np.quantile(p, 0.50)),
        'p95': float(np.quantile(p, 0.95)),
        'conditions': condition_metrics,
    }


def read_conditions(files):
    result = {}
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            result[path] = str(d['condition'].item()) if 'condition' in d.files else 'clean'
    return result


def fold_paths(files, fold):
    tr = [p for p in files if FOLD_ASSIGNMENTS[os.path.basename(os.path.dirname(p))] != fold]
    va = [p for p in files if FOLD_ASSIGNMENTS[os.path.basename(os.path.dirname(p))] == fold]
    train_speakers = {os.path.basename(os.path.dirname(p)) for p in tr}
    val_speakers = {os.path.basename(os.path.dirname(p)) for p in va}
    if not tr or not va or (train_speakers & val_speakers):
        raise RuntimeError(f'Fold {fold} is invalid or overlaps speakers')
    return tr, va


def run_cv(expert, files, force=False):
    spec = EXPERTS[expert]
    directory = os.path.join(MODELS_DIR, spec['folder'])
    os.makedirs(directory, exist_ok=True)
    result_path = os.path.join(directory, 'fold_results.npy')
    results = []
    if os.path.exists(result_path) and not force:
        old = list(np.load(result_path, allow_pickle=True))
        if old and all(r.get('training_version') == VERSION and r.get('expert') == expert for r in old):
            results = old
    finished = {int(r['fold']) for r in results
                if os.path.isfile(os.path.join(directory, f"best_fold{int(r['fold'])}.pt"))}
    for fold in range(1, 5):
        if fold in finished:
            print(f'{spec["name"]} fold {fold}: existing v25 checkpoint; skipping')
            continue
        seed_everything(SEED + fold)
        tr, va = fold_paths(files, fold)
        print(f'\n{spec["name"]} v25 fold {fold}: train={len(tr):,} val={len(va):,}')
        mean, std = cqcc_stats(tr, expert)
        if len(mean) != spec['dim'] or not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise RuntimeError(f'{expert} fold {fold}: invalid normalizer')
        train_ds, train_loader = prepare_loader(tr, spec, mean, std, True, SEED + fold)
        val_ds, val_loader = prepare_loader(va, spec, mean, std, False, SEED + fold)
        cond_by_file = read_conditions(va)
        model = make_model(spec)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=3, min_lr=1e-7)
        scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
        criterion = torch.nn.BCEWithLogitsLoss()
        best_auc = -1.0
        best_epoch = 0
        best_stats = None
        history = []
        checkpoint = os.path.join(directory, f'best_fold{fold}.pt')
        for epoch in range(1, spec['epochs']+1):
            start = time.perf_counter()
            loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            diag = validate(model, val_loader, cond_by_file)
            scheduler.step(diag['clip_log_loss'])
            improved = diag['auc'] > best_auc + AUC_MIN_DELTA
            if improved:
                best_auc, best_epoch, best_stats = diag['auc'], epoch, diag
                torch.save(model.state_dict(), checkpoint)
            history.append({'epoch': epoch, 'train_window_bce': loss,
                            'val': diag, 'lr': optimizer.param_groups[0]['lr']})
            print(f'  Epoch {epoch:02d}/{spec["epochs"]} TrainBCE={loss:.4f} '
                  f'ClipAUC={diag["auc"]:.4f} ClipLoss={diag["clip_log_loss"]:.4f} '
                  f'F1@.5={diag["f1_at_0_5"]:.4f} AI-pred={diag["ai_prediction_rate"]:.1%} '
                  f'best={best_epoch} time={time.perf_counter()-start:.1f}s')
            if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
                print(f'  AUC-based early stop: epoch={epoch}, best={best_epoch}')
                break
        result = {
            'training_version': VERSION, 'expert': expert, 'fold': fold,
            'best_epoch': best_epoch, 'best_auc': best_auc,
            'best_stats': best_stats, 'mean': np.asarray(mean, dtype=np.float32),
            'std': np.asarray(std, dtype=np.float32), 'history': history,
            'selection_metric': 'clip_auc', 'objective': 'window_bce',
            'train_clips': len(tr), 'val_clips': len(va),
        }
        results = [r for r in results if int(r['fold']) != fold] + [result]
        results.sort(key=lambda r: int(r['fold']))
        np.save(result_path, np.asarray(results, dtype=object), allow_pickle=True)
        print(f'  Saved {checkpoint} (AUC={best_auc:.4f}, best epoch={best_epoch})')
        del train_ds, val_ds, train_loader, val_loader, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'{spec["name"]}: CV best epochs: {[r["best_epoch"] for r in results]}')
    print(f'{spec["name"]}: CV selected AUCs: {[round(r["best_auc"], 4) for r in results]}')


def run_final(expert, files, force=False):
    spec = EXPERTS[expert]
    directory = os.path.join(MODELS_DIR, spec['folder'])
    result_path = os.path.join(directory, 'fold_results.npy')
    if not os.path.isfile(result_path):
        raise FileNotFoundError(f'Run CV first: {result_path}')
    results = list(np.load(result_path, allow_pickle=True))
    if len(results) != 4 or {int(r['fold']) for r in results} != {1,2,3,4} or any(r.get('training_version') != VERSION for r in results):
        raise RuntimeError('Incomplete or incompatible v25 CV results.')
    epochs = [int(r['best_epoch']) for r in results]
    if RETRAIN_EPOCH_POLICY == 'median':
        target = max(1, int(np.median(epochs)))
    elif RETRAIN_EPOCH_POLICY == 'mean':
        target = max(1, int(round(np.mean(epochs))))
    else:
        target = max(epochs)
    print(f'{spec["name"]}: CV best epochs={epochs}, final epoch budget={target} ({RETRAIN_EPOCH_POLICY})')
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, 'final_model.pt')
    stats_path = os.path.join(directory, 'final_norm_stats.npy')
    if os.path.exists(model_path) and os.path.exists(stats_path) and not force:
        old = np.load(stats_path, allow_pickle=True).item()
        if old.get('training_version') == VERSION:
            if os.path.getmtime(model_path) < os.path.getmtime(result_path):
                raise RuntimeError('CV results are newer than the saved final checkpoint. Refit final using --force.')
            print(f'{spec["name"]}: saved v25 final model exists; use --force to refit.')
            return
        raise RuntimeError('Existing final file belongs to another version; refusing overwrite without --force.')
    seed_everything(SEED)
    mean, std = cqcc_stats(files, expert)
    ds, loader = prepare_loader(files, spec, mean, std, True, SEED)
    model = make_model(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=3, min_lr=1e-7)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, target+1):
        start = time.perf_counter()
        loss = train_epoch(model, loader, optimizer, criterion, scaler)
        scheduler.step(loss)
        history.append({'epoch': epoch, 'window_bce': loss, 'lr': optimizer.param_groups[0]['lr']})
        print(f'{spec["name"]} final {epoch:02d}/{target} BCE={loss:.4f}, {time.perf_counter()-start:.1f}s')
    torch.save(model.state_dict(), model_path)
    np.save(stats_path, {'mean': np.asarray(mean, dtype=np.float32),
                         'std': np.asarray(std, dtype=np.float32),
                         'training_version': VERSION, 'feature': expert,
                         'objective': 'window_bce', 'checkpoint_selection': 'clip_auc',
                         'final_epochs': target}, allow_pickle=True)
    np.save(os.path.join(directory, 'final_training_history.npy'), history, allow_pickle=True)
    print(f'Saved v25 final model: {model_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert', choices=['cqcc','f0','all'], default='all')
    parser.add_argument('--stage', choices=['cv','final','all'], default='cv')
    parser.add_argument('--force', action='store_true', help='Overwrite existing v25 checkpoints (never old v16/v20).')
    args = parser.parse_args()
    print(f'v25 auxiliary training | device={DEVICE} | metric=clip AUC | minimum={MIN_EPOCHS}, patience={PATIENCE}')
    files = all_files()
    for expert in (('cqcc','f0') if args.expert == 'all' else (args.expert,)):
        print(f'\n===== {expert.upper()} v25 =====')
        preflight(files, expert, EXPERTS[expert]['dim'])
        if args.stage in ('cv', 'all'):
            run_cv(expert, files, force=args.force)
        if args.stage in ('final', 'all'):
            run_final(expert, files, force=args.force)


if False and __name__ == '__main__':
    main()
'''

def _get_auc_expert():
    return _compile_internal('improvedCQCC_v25_aux_train_auc',_SOURCE_auc_f0)

def train_auc_f0(stage='cv',force=False):
    """Preserve the established F0 AUC-selection and checkpoint format.
    Old CQCC is NOT trained. The recorded output directory is F0-v25-AUC.
    """
    import sys
    mod=_get_auc_expert()
    if stage=='check':
        files=mod.all_files()
        mod.preflight(files,'f0',mod.EXPERTS['f0']['dim'])
        print('PASS: F0 development cache compatible with AUC-selected expert')
        return
    orig=sys.argv
    try:
        sys.argv=[orig[0],'--expert','f0','--stage',stage]+(['--force'] if force else [])
        return mod.main()
    finally:sys.argv=orig

if __name__=='__main__':
    # The latest proposed model uses AUC-selected F0, not the former F1-only expert.
    import argparse
    parser=argparse.ArgumentParser(description='Train the selected F0 expert for proposed fusion')
    parser.add_argument('--stage',choices=['cv','final','check'],default='cv')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    train_auc_f0(stage=args.stage,force=args.force)
