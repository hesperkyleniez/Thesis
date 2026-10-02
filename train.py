# train.py
# 4-fold speaker-grouped CV for CNN-only, CNN-GRU, and CNN-GRU late fusion.
# The CNN-GRU models keep the paper's lightweight design: three CNN blocks,
# one 128-unit GRU, Log-Mel input, and MFCC/F0 late fusion with learnable scalars.

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
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from tqdm import tqdm

sys.path.insert(0, r"D:\Thesis\code")
from config import (
    FEATURES_DIR, MODELS_DIR, RESULTS_DIR,
    FOLD_ASSIGNMENTS, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, LR_FACTOR, DROPOUT, SEED,
    MAX_TRAIN_WINDOWS_PER_FILE, MAX_VAL_WINDOWS_PER_FILE,
)


# -----------------------------------------------------------------------------
# Reproducibility and device
# -----------------------------------------------------------------------------
TRAINING_VERSION = "v11_clean_four_models_robust_telephone"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"

# Safe CUDA throughput setting. Reproducibility remains controlled by the seed.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class HybridDataset(Dataset):
    def __init__(
        self,
        npz_files,
        mfcc_mean,
        mfcc_std,
        f0_speaker_stats,
        f0_fallback,
        desc="Loading",
        max_windows=None,
        seed=SEED,
    ):
        # NOTE ON max_windows:
        # This used to default to 3. HybridDataset is built ONCE per fold,
        # before the epoch loop starts, and that subsample is never
        # re-drawn -- so with the old default, every model trained on the
        # *same fixed 3 windows per file* for all 50 epochs, every fold,
        # and the final retrain. That is a large, unnecessary cut to
        # training diversity: NpzFile access below already decompresses
        # each array in full regardless of how many windows you keep, so
        # the old cap saved no I/O -- it only threw away signal. Default
        # is now "keep every window"; pass an integer here only if you
        # hit a real memory ceiling, and prefer the highest value your
        # RAM allows over a small fixed number.
        mfcc_mean_t = torch.as_tensor(mfcc_mean, dtype=torch.float32)
        mfcc_std_t = torch.as_tensor(mfcc_std, dtype=torch.float32)

        self.index = []
        logmel_list = []
        mfcc_list = []
        f0_list = []
        label_list = []

        total_windows_seen = 0
        total_windows_kept = 0

        for path in tqdm(npz_files, desc=desc, leave=False):
            try:
                with np.load(path, allow_pickle=True) as d:
                    n = int(d["n_windows"])
                    label = int(d["label"])

                    # Each key is read out of the NPZ exactly once --
                    # NpzFile re-decompresses the whole array on every
                    # d["key"] access, so indexing it in a loop (the old
                    # code did d["logmel"][i] per window) silently
                    # re-decompressed the full array once per window.
                    full_logmel = d["logmel"]
                    full_mfcc = d["mfcc"]
                    full_f0 = d["f0"]

                    if max_windows is not None and n > max_windows:
                        # Deterministic coverage cap. Evenly spaced windows retain
                        # the beginning, middle, and end of long recordings instead
                        # of repeatedly drawing the same arbitrary subset.
                        indices = np.linspace(
                            0, n - 1, max_windows, dtype=np.int64
                        )
                    else:
                        indices = np.arange(n, dtype=np.int64)

                    # Unchanged: MFCC stays globally normalized at the end.
                    # F0 is normalized per-speaker right here, using this
                    # file's own speaker's stats (folder name = speaker id,
                    # matching prepare_dataset.py's storage layout and
                    # get_fold_files()'s existing speaker lookup).
                    speaker = os.path.basename(os.path.dirname(path))
                    f0_mean, f0_std = f0_speaker_stats.get(speaker, f0_fallback)

                    total_windows_seen += n
                    total_windows_kept += len(indices)

                    for i in indices:
                        self.index.append((path, int(i)))
                        logmel_list.append(full_logmel[i])
                        mfcc_list.append(full_mfcc[i])
                        f0_list.append((full_f0[i] - f0_mean) / f0_std)
                        label_list.append(label)

            except Exception as e:
                print(f"WARNING: {os.path.basename(path)}: {e}")

        if not label_list:
            raise RuntimeError(f"No valid windows found for {desc} dataset.")

        if total_windows_seen > 0:
            kept_pct = 100.0 * total_windows_kept / total_windows_seen
            print(
                f"  {desc}: kept {total_windows_kept:,}/{total_windows_seen:,} "
                f"windows ({kept_pct:.1f}%) from {len(npz_files):,} files"
            )

        self.logmel = torch.as_tensor(
            np.stack(logmel_list), dtype=torch.float32
        ).unsqueeze(1).contiguous()
        self.mfcc = (
            torch.as_tensor(np.stack(mfcc_list), dtype=torch.float32)
            - mfcc_mean_t
        ) / mfcc_std_t
        self.f0 = torch.as_tensor(
            np.stack(f0_list), dtype=torch.float32
        )
        self.labels = torch.as_tensor(
            label_list, dtype=torch.float32
        ).unsqueeze(1)

        print(f"  {desc}: {len(self.labels):,} windows loaded into RAM")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path, _ = self.index[idx]
        return (
            self.logmel[idx],
            self.mfcc[idx],
            self.f0[idx],
            self.labels[idx],
            path,
        )


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, spatial_dropout=0.0, bias=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        if spatial_dropout > 0:
            layers.append(nn.Dropout2d(spatial_dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SpecAugment(nn.Module):
    def __init__(self, freq_mask=20, time_mask=20, num_masks=2):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.num_masks = num_masks

    def forward(self, x):
        if not self.training:
            return x

        x = x.clone()
        _, _, freq, frames = x.shape

        for _ in range(self.num_masks):
            f = torch.randint(0, self.freq_mask + 1, (1,)).item()
            if f > 0:
                f0 = torch.randint(0, freq - f + 1, (1,)).item()
                x[:, :, f0:f0 + f, :] = 0

            t = torch.randint(0, self.time_mask + 1, (1,)).item()
            if t > 0:
                t0 = torch.randint(0, frames - t + 1, (1,)).item()
                x[:, :, :, t0:t0 + t] = 0

        return x


class TemporalCNNGRUEncoder(nn.Module):
    """
    Lightweight temporal encoder.

    Paper-aligned structure:
      Log-Mel -> 3 CNN blocks -> single 128-unit GRU -> 128-d embedding.

    Optimization:
      The post-CNN feature map is reduced across frequency only, preserving
      all 16 time positions instead of collapsing the map to 4 timesteps.
      This gives the GRU a substantially more useful temporal sequence without
      adding GRU layers or increasing the GRU hidden size.
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        spatial_dropout = min(0.10, dropout * 0.20)

        self.spec_aug = SpecAugment(freq_mask=20, time_mask=20, num_masks=2)
        self.cnn = nn.Sequential(
            CNNBlock(1, 32, spatial_dropout, bias=False),
            CNNBlock(32, 64, spatial_dropout, bias=False),
            CNNBlock(64, 128, spatial_dropout, bias=False),
        )
        self.sequence_norm = nn.LayerNorm(128)
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.gru_norm = nn.LayerNorm(128)
        self.gru_dropout = nn.Dropout(dropout * 0.6)

        self._initialize_recurrent_weights()

    def _initialize_recurrent_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, logmel):
        x = self.spec_aug(logmel)
        x = self.cnn(x)

        # (B, 128, 16, 16) -> (B, 16, 128)
        # Frequency is averaged while the time axis is preserved.
        x = x.mean(dim=2).transpose(1, 2).contiguous()
        x = self.sequence_norm(x)

        out, _ = self.gru(x)
        emb = self.gru_norm(out[:, -1, :])
        return self.gru_dropout(emb)


class CNNOnly(nn.Module):
    """Baseline 1: CNN only."""
    def __init__(self, dropout=0.5):
        super().__init__()
        self.spec_aug = SpecAugment()
        self.cnn = nn.Sequential(
            CNNBlock(1, 32),
            CNNBlock(32, 64),
            CNNBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, 1),
        )

    def forward(self, logmel, mfcc=None, f0=None):
        x = self.spec_aug(logmel)
        x = self.cnn(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_fusion_weights(self):
        return {}


class CNNGRUOnly(nn.Module):
    """
    Baseline 2: CNN-GRU using Log-Mel only.
    Keeps the paper's single 128-unit GRU and lightweight three-block CNN.
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.encoder = TemporalCNNGRUEncoder(dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, logmel, mfcc=None, f0=None):
        emb = self.encoder(logmel)
        return self.classifier(emb)

    def get_fusion_weights(self):
        return {}

class CNNGRUFusion(nn.Module):
    """
    Thesis feature-fusion model:
      Log-Mel -> CNN-GRU -> 128-D embedding
      MFCC -> pooled 80-D vector
      F0 -> pooled 2-D vector

    The three representations are weighted by independent learnable scalar
    parameters alpha, beta, and gamma (initialized to 1.0), concatenated, and
    classified. This matches the current Chapter III methodology.
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.encoder = TemporalCNNGRUEncoder(dropout=dropout)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))

        self.classifier = nn.Sequential(
            nn.Linear(128 + 80 + 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, 1),
        )

    def forward(self, logmel, mfcc, f0):
        cnn_emb = self.encoder(logmel)
        fused = torch.cat(
            [self.alpha * cnn_emb, self.beta * mfcc, self.gamma * f0],
            dim=1,
        )
        return self.classifier(fused)

    def get_fusion_weights(self):
        return {
            "alpha (CNN-GRU)": float(self.alpha.detach().cpu().item()),
            "beta (MFCC)": float(self.beta.detach().cpu().item()),
            "gamma (F0)": float(self.gamma.detach().cpu().item()),
        }


MODELS = {
    "CNN-only": CNNOnly,
    "CNN-GRU": CNNGRUOnly,
    "CNN-GRU-F": CNNGRUFusion,
}


# -----------------------------------------------------------------------------
# File collection and fold handling
# -----------------------------------------------------------------------------
def get_all_train_files():
    return glob.glob(
        os.path.join(FEATURES_DIR, "train", "**", "*.npz"),
        recursive=True,
    )


def get_fold_files(all_files, val_fold):
    train_files = []
    val_files = []

    for path in all_files:
        speaker = os.path.basename(os.path.dirname(path))
        fold = FOLD_ASSIGNMENTS.get(speaker)

        if fold is None:
            continue

        if fold == val_fold:
            val_files.append(path)
        else:
            train_files.append(path)

    return sorted(train_files), sorted(val_files)


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------
def compute_per_speaker_f0_stats(files):
    """
    Returns ({speaker_id: (mean, std)}, fallback_mean_std) computed only
    from each speaker's own files.

    Absolute pitch (F0) is strongly speaker-dependent -- driven mostly by
    anatomy/gender, not by whether audio is real or AI-generated. Z-scoring
    F0 against a single dataset-wide mean/std leaves each speaker's
    absolute pitch level intact as a feature, which gives the network a
    speaker-identity shortcut instead of forcing it to learn genuine
    real-vs-AI pitch *dynamics*. Normalizing per-speaker removes that.

    Safe to call independently on train/val/test files with no label
    leakage: it only reads raw F0 values from that split's own speakers,
    never touches labels, and never reads another split's audio.
    """
    per_speaker_f0 = {}

    for path in tqdm(files, desc="Per-speaker F0 stats", leave=False):
        speaker = os.path.basename(os.path.dirname(path))
        try:
            with np.load(path, allow_pickle=True) as d:
                per_speaker_f0.setdefault(speaker, []).append(d["f0"])
        except Exception as e:
            print(f"WARNING: f0 stats {os.path.basename(path)}: {e}")

    speaker_stats = {}
    all_f0 = []

    for speaker, arrays in per_speaker_f0.items():
        cat = np.concatenate(arrays, axis=0)
        all_f0.append(cat)
        speaker_stats[speaker] = (cat.mean(axis=0), cat.std(axis=0) + 1e-8)

    if all_f0:
        global_cat = np.concatenate(all_f0, axis=0)
        fallback = (global_cat.mean(axis=0), global_cat.std(axis=0) + 1e-8)
    else:
        fallback = (
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )

    return speaker_stats, fallback


def lookup_f0_stats(path, f0_speaker_stats, f0_fallback):
    speaker = os.path.basename(os.path.dirname(path))
    return f0_speaker_stats.get(speaker, f0_fallback)


def compute_fold_norm_stats(train_files):
    """
    MFCC stays normalized dataset-wide (unchanged) -- only F0 switches to
    per-speaker (see compute_per_speaker_f0_stats). Returns
    (mfcc_mean, mfcc_std, f0_speaker_stats, f0_fallback).
    """
    mfcc_all = []

    for path in tqdm(train_files, desc="Computing MFCC norm stats", leave=False):
        try:
            with np.load(path, allow_pickle=True) as d:
                mfcc_all.append(d["mfcc"])
        except Exception as e:
            print(f"WARNING: norm stats {os.path.basename(path)}: {e}")

    if not mfcc_all:
        raise RuntimeError("No valid training feature files for normalization.")

    mfcc_cat = np.concatenate(mfcc_all, axis=0)
    f0_speaker_stats, f0_fallback = compute_per_speaker_f0_stats(train_files)

    return (
        mfcc_cat.mean(axis=0),
        mfcc_cat.std(axis=0) + 1e-8,
        f0_speaker_stats,
        f0_fallback,
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def collate_with_paths(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
        torch.stack([b[3] for b in batch]),
        [b[4] for b in batch],
    )


def make_loader(dataset, shuffle, generator=None):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
        collate_fn=collate_with_paths,
    )


# -----------------------------------------------------------------------------
# Training and validation
# -----------------------------------------------------------------------------
def make_scaler():
    if not USE_AMP:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def build_optimizer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """AdamW optimizer shared by all three neural architectures."""
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for logmel, mfcc, f0, labels, _ in loader:
        logmel = logmel.to(device, non_blocking=True)
        mfcc = mfcc.to(device, non_blocking=True)
        f0 = f0.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(logmel, mfcc, f0)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(logmel, mfcc, f0)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.detach().item()
        preds = (torch.sigmoid(logits.detach()) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def validate_with_clip_f1(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    clip_probs = {}
    clip_labels = {}

    with torch.no_grad():
        for logmel, mfcc, f0, labels, paths in loader:
            logmel = logmel.to(device, non_blocking=True)
            mfcc = mfcc.to(device, non_blocking=True)
            f0 = f0.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(logmel, mfcc, f0)
                    loss = criterion(logits, labels)
            else:
                logits = model(logmel, mfcc, f0)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            probs_np = probs.squeeze(1).cpu().numpy()
            labels_np = labels.squeeze(1).cpu().numpy().astype(int)

            for path, prob, label in zip(paths, probs_np, labels_np):
                clip_probs.setdefault(path, []).append(float(prob))
                clip_labels[path] = int(label)

    clip_preds = []
    clip_targets = []
    clip_scores = []

    for path in clip_probs:
        score = float(np.mean(clip_probs[path]))
        clip_scores.append(score)
        clip_preds.append(1 if score >= 0.5 else 0)
        clip_targets.append(clip_labels[path])

    clip_f1 = f1_score(
        clip_targets, clip_preds, zero_division=0
    ) if clip_targets else 0.0

    return (
        total_loss / max(len(loader), 1),
        correct / max(total, 1),
        clip_f1,
        clip_preds,
        clip_targets,
        clip_scores,
    )


def soft_vote(model, val_files, mfcc_mean, mfcc_std, f0_speaker_stats, f0_fallback):
    model.eval()
    clip_probs = []
    clip_labels = []
    clip_scores = []

    mfcc_mean_t = torch.as_tensor(mfcc_mean, dtype=torch.float32, device=device)
    mfcc_std_t = torch.as_tensor(mfcc_std, dtype=torch.float32, device=device)

    with torch.no_grad():
        for path in tqdm(val_files, desc="Soft voting", leave=False):
            try:
                f0_mean, f0_std = lookup_f0_stats(path, f0_speaker_stats, f0_fallback)
                f0_mean_t = torch.as_tensor(f0_mean, dtype=torch.float32, device=device)
                f0_std_t = torch.as_tensor(f0_std, dtype=torch.float32, device=device)

                with np.load(path, allow_pickle=True) as d:
                    lm = torch.as_tensor(
                        d["logmel"], dtype=torch.float32, device=device
                    ).unsqueeze(1)
                    mc = (
                        torch.as_tensor(
                            d["mfcc"], dtype=torch.float32, device=device
                        ) - mfcc_mean_t
                    ) / mfcc_std_t
                    f0 = (
                        torch.as_tensor(
                            d["f0"], dtype=torch.float32, device=device
                        ) - f0_mean_t
                    ) / f0_std_t
                    label = int(d["label"])

                if USE_AMP:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(lm, mc, f0)
                else:
                    logits = model(lm, mc, f0)

                probs = torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
                score = float(np.mean(probs))

                clip_scores.append(score)
                clip_labels.append(label)
                clip_probs.append(1 if score >= 0.5 else 0)

            except Exception as e:
                print(f"WARNING: soft vote {os.path.basename(path)}: {e}")

    return (
        np.asarray(clip_probs),
        np.asarray(clip_labels),
        np.asarray(clip_scores),
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def calculate_eer(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    if len(np.unique(labels)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


# -----------------------------------------------------------------------------
# One model, four folds
# -----------------------------------------------------------------------------
def train_model(model_name, model_class):
    print(f"\n{'=' * 70}")
    print(f"  TRAINING: {model_name}")
    print(f"  Version : {TRAINING_VERSION}")
    print(f"{'=' * 70}")

    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    results_path = os.path.join(model_dir, "fold_results.npy")

    all_files = get_all_train_files()
    if not all_files:
        raise RuntimeError(
            f"No feature files found in {os.path.join(FEATURES_DIR, 'train')}"
        )

    fold_results = []
    completed_folds = []

    if os.path.exists(results_path):
        try:
            existing = list(np.load(results_path, allow_pickle=True))
            if existing and all(
                r.get("training_version") == TRAINING_VERSION
                for r in existing
            ):
                fold_results = existing
                completed_folds = [int(r["fold"]) for r in fold_results]
                print(f"  Resuming compatible run — completed folds: {completed_folds}")
            else:
                print("  Existing results use an older training version — starting fresh.")
        except Exception as e:
            print(f"  Existing fold results could not be reused: {e}")

    for fold in [1, 2, 3, 4]:
        if fold in completed_folds:
            print(f"\n  Skipping completed Fold {fold}")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {model_name} — FOLD {fold} / 4")
        print(f"{'=' * 70}")

        train_files, val_files = get_fold_files(all_files, fold)
        print(f"  Train files : {len(train_files):,}")
        print(f"  Val files   : {len(val_files):,}")

        mfcc_mean, mfcc_std, f0_speaker_stats, f0_fallback = compute_fold_norm_stats(
            train_files
        )

        # Val speakers are different people from train speakers (that's
        # the whole point of speaker-grouped CV) -- they don't exist in
        # f0_speaker_stats at all. Compute their own per-speaker F0 stats
        # from their own data, same as train. MFCC normalization is
        # unchanged and still uses the train-derived mfcc_mean/std.
        val_f0_speaker_stats, val_f0_fallback = compute_per_speaker_f0_stats(
            val_files
        )

        print("  Loading training windows into RAM...")
        train_ds = HybridDataset(
            train_files,
            mfcc_mean,
            mfcc_std,
            f0_speaker_stats,
            f0_fallback,
            desc="Train",
            max_windows=MAX_TRAIN_WINDOWS_PER_FILE,
        )

        print("  Loading validation windows into RAM...")
        val_ds = HybridDataset(
            val_files,
            mfcc_mean,
            mfcc_std,
            val_f0_speaker_stats,
            val_f0_fallback,
            desc="Val",
            max_windows=MAX_VAL_WINDOWS_PER_FILE,
        )

        generator = torch.Generator()
        generator.manual_seed(SEED + fold)

        train_loader = make_loader(train_ds, shuffle=True, generator=generator)
        val_loader = make_loader(val_ds, shuffle=False)

        print(f"  Train windows: {len(train_ds):,}")
        print(f"  Val windows  : {len(val_ds):,}")

        model = model_class(dropout=DROPOUT).to(device)
        optimizer = build_optimizer(model)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_FACTOR,
            patience=3,
            min_lr=1e-7,
        )
        scaler = make_scaler()

        best_val_f1 = -1.0
        best_epoch = 1
        patience_count = 0
        best_model_path = os.path.join(
            model_dir, f"best_fold{fold}.pt"
        )

        for epoch in range(1, MAX_EPOCHS + 1):
            start = time.time()

            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, scaler
            )
            val_loss, val_acc, val_clip_f1, _, _, _ = validate_with_clip_f1(
                model, val_loader, criterion
            )
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start

            improved = val_clip_f1 > best_val_f1 + 1e-6
            if improved:
                best_val_f1 = val_clip_f1
                best_epoch = epoch
                patience_count = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_count += 1

            print(
                f"  Epoch {epoch:02d}/{MAX_EPOCHS} | "
                f"Train {train_loss:.4f}/{train_acc:.4f} | "
                f"Val {val_loss:.4f}/{val_acc:.4f} | "
                f"Clip F1 {val_clip_f1:.4f} | "
                f"LR {current_lr:.2e} | "
                f"ES {patience_count}/{PATIENCE} | "
                f"{elapsed:.0f}s"
            )

            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        print(f"\n  Evaluating Fold {fold} with all validation windows...")
        state = torch.load(
            best_model_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)

        preds, labels, probs = soft_vote(
            model,
            val_files,
            mfcc_mean,
            mfcc_std,
            val_f0_speaker_stats,
            val_f0_fallback,
        )

        if len(labels) == 0:
            raise RuntimeError(f"Fold {fold} produced no validation predictions.")

        acc = float(np.mean(preds == labels))
        f1 = float(f1_score(labels, preds, zero_division=0))
        prec = float(precision_score(labels, preds, zero_division=0))
        rec = float(recall_score(labels, preds, zero_division=0))
        auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
        eer = calculate_eer(labels, probs)
        weights = model.get_fusion_weights()

        print(f"\n  RESULTS — {model_name} / Fold {fold}")
        print(f"     Accuracy  : {acc:.4f}")
        print(f"     F1-score  : {f1:.4f}")
        print(f"     Precision : {prec:.4f}")
        print(f"     Recall    : {rec:.4f}")
        print(f"     ROC-AUC   : {auc:.4f}")
        print(f"     EER       : {eer:.4f}")

        if weights:
            print("     Fusion weights:")
            for key, value in weights.items():
                print(f"       {key}: {value:.4f}")

        fold_results.append({
            "training_version": TRAINING_VERSION,
            "fold": fold,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "auc": auc,
            "eer": eer,
            "weights": weights,
            "best_epoch": best_epoch,
            "norm": {
                "mfcc_mean": mfcc_mean,
                "mfcc_std": mfcc_std,
                "f0_speaker_stats": f0_speaker_stats,
                "f0_fallback": f0_fallback,
            },
        })

        completed_folds.append(fold)
        fold_results.sort(key=lambda r: int(r["fold"]))
        np.save(results_path, fold_results)
        print(f"  Fold {fold} saved to {results_path}")

        del train_ds, val_ds, train_loader, val_loader, model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not fold_results:
        return []

    print(f"\n{'=' * 70}")
    print(f"  {model_name} — 4-FOLD CV SUMMARY")
    print(f"{'=' * 70}")

    accs = np.asarray([r["accuracy"] for r in fold_results], dtype=float)
    f1s = np.asarray([r["f1"] for r in fold_results], dtype=float)
    precs = np.asarray([r["precision"] for r in fold_results], dtype=float)
    recs = np.asarray([r["recall"] for r in fold_results], dtype=float)
    aucs = np.asarray([r["auc"] for r in fold_results], dtype=float)
    eers = np.asarray([r["eer"] for r in fold_results], dtype=float)

    print(f"  Accuracy  : {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  F1-score  : {f1s.mean():.4f} ± {f1s.std():.4f}")
    print(f"  Precision : {precs.mean():.4f} ± {precs.std():.4f}")
    print(f"  Recall    : {recs.mean():.4f} ± {recs.std():.4f}")
    print(f"  ROC-AUC   : {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"  EER       : {eers.mean():.4f} ± {eers.std():.4f}")

    print("\n  Per-fold F1:")
    for result in sorted(fold_results, key=lambda r: int(r["fold"])):
        print(f"    Fold {result['fold']}: {result['f1']:.4f}")

    return fold_results


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"AMP   : enabled")
    else:
        print("GPU   : CPU")
        print("AMP   : disabled")

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
        for model_name, model_class in MODELS.items():
            train_model(model_name, model_class)
    else:
        train_model(args.model, MODELS[args.model])

    print(f"\n{'=' * 85}")
    print("  FINAL COMPARISON — ALL COMPLETED MODELS")
    print(f"{'=' * 85}")
    print(
        f"  {'Model':<15} {'Acc':>8} {'F1':>8} {'Prec':>8} "
        f"{'Rec':>8} {'AUC':>8} {'EER':>8}"
    )
    print(f"  {'-' * 80}")

    all_results = {}
    for model_name in MODELS:
        results_path = os.path.join(
            MODELS_DIR, model_name, "fold_results.npy"
        )
        if not os.path.exists(results_path):
            continue

        try:
            results = list(np.load(results_path, allow_pickle=True))
            if not results:
                continue

            all_results[model_name] = results
            print(
                f"  {model_name:<15} "
                f"{np.mean([r.get('accuracy', 0) for r in results]):>8.4f} "
                f"{np.mean([r.get('f1', 0) for r in results]):>8.4f} "
                f"{np.mean([r.get('precision', 0) for r in results]):>8.4f} "
                f"{np.mean([r.get('recall', 0) for r in results]):>8.4f} "
                f"{np.mean([r.get('auc', 0) for r in results]):>8.4f} "
                f"{np.mean([r.get('eer', 0) for r in results]):>8.4f}"
            )
        except Exception as e:
            print(f"  {model_name:<15} [Error loading data: {e}]")

    if all_results:
        np.save(
            os.path.join(RESULTS_DIR, "all_results.npy"),
            all_results,
        )
        print(f"\n  Results saved to {os.path.join(RESULTS_DIR, 'all_results.npy')}")
