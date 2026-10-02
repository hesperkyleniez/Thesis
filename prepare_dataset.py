"""Prepare training/validation and clean held-out feature caches.

The fixed augmentation assignment proportions are preserved exactly.  This file
uses augmentation.py as the single degradation implementation so training and
robustness evaluation cannot silently drift apart.
"""

import hashlib
import os
import random
import sys
import time
import zipfile

import librosa
import numpy as np
from tqdm import tqdm

sys.path.insert(0, r"D:\Thesis\code")

from config import (
    TRAIN_DIR,
    TEST_DIR,
    FEATURES_DIR,
    SAMPLE_RATE,
    RESOURCES_DIR,
    OPENSLR_DIR,
    AIR_DIR,
    URBANSOUND_DIR,
    AUGMENTATION_SPLITS,
    LABEL_REAL,
    LABEL_AI,
    SEED,
)
from dataset import load_all_files, extract_windows
from augmentation import collect_rir_files, apply_augmentation, peak_normalize
from features import extract_logmel, extract_mfcc, extract_cqcc, extract_f0

CACHE_VERSION = "v11_robust_telephone"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_if_needed(zip_path, out_dir):
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        return
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing resource archive: {zip_path}")
    print(f"Extracting {zip_path}...")
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def ensure_resources():
    extract_if_needed(os.path.join(RESOURCES_DIR, "rirs_noises.zip"), OPENSLR_DIR)
    extract_if_needed(os.path.join(RESOURCES_DIR, "archive.zip"), URBANSOUND_DIR)
    extract_if_needed(os.path.join(RESOURCES_DIR, "AIR_wav_files.zip"), AIR_DIR)

    room_rirs, mobile_rirs, urban_files = collect_rir_files()
    print(f"Room RIRs   : {len(room_rirs)}")
    print(f"Mobile RIRs : {len(mobile_rirs)}")
    print(f"Urban noise : {len(urban_files)}")

    if not room_rirs:
        raise RuntimeError("No room RIRs found")
    if not mobile_rirs:
        raise RuntimeError("No mobile/device RIRs found")
    if not urban_files:
        raise RuntimeError("No UrbanSound8K WAV files found")
    return room_rirs, mobile_rirs, urban_files


def pair_key(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    return name[:-3] if name.endswith("_AI") else name


def pair_rng(filepath):
    """Stable per-pair RNG so paired real/AI clips get matched degradation draws."""
    digest = hashlib.sha256(f"{SEED}|{pair_key(filepath)}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def assign_fixed_conditions(samples):
    """Load the existing assignment map, or create it once with the same split."""
    conditions_path = os.path.join(FEATURES_DIR, "conditions.npy")
    if os.path.exists(conditions_path):
        print("Loading existing condition assignments (split preserved)...")
        return dict(np.load(conditions_path, allow_pickle=True).item())

    print("Assigning paired augmentation conditions...")
    names = list(AUGMENTATION_SPLITS.keys())
    probs = list(AUGMENTATION_SPLITS.values())

    pair_conditions = {}
    assignments = {}
    for filepath, _, _ in samples:
        key = pair_key(filepath)
        if key not in pair_conditions:
            pair_conditions[key] = np.random.choice(names, p=probs)
        assignments[filepath] = pair_conditions[key]

    np.save(conditions_path, assignments)

    from collections import Counter
    counts = Counter(assignments.values())
    print("Paired condition distribution:")
    for condition, count in sorted(counts.items()):
        print(f"  {condition:<20}: {count:>6} ({100*count/len(assignments):.1f}%)")
    print(f"Saved: {conditions_path}")
    return assignments


def _output_path(split_name, filepath, label, speaker_id):
    speaker_dir = os.path.join(FEATURES_DIR, split_name, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(filepath))[0]
    if label == LABEL_AI and not name.endswith("_AI"):
        name += "_AI"
    elif label == LABEL_REAL and not name.endswith("_Real"):
        name += "_Real"
    return os.path.join(speaker_dir, name + ".npz")


def process_split(samples, split_name, room_rirs, mobile_rirs, urban_files, conditions=None):
    success = skipped = 0
    errors = []
    print(f"\nProcessing {split_name}: {len(samples):,} files")

    for filepath, label, speaker_id in tqdm(samples):
        out_path = _output_path(split_name, filepath, label, speaker_id)

        if os.path.exists(out_path):
            try:
                with np.load(out_path, allow_pickle=True) as cached:
                    if (
                        "cqcc" in cached.files
                        and "cache_version" in cached.files
                        and str(cached["cache_version"]) == CACHE_VERSION
                    ):
                        skipped += 1
                        continue
            except Exception:
                pass

        try:
            audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
            audio = np.asarray(audio, dtype=np.float32)

            # Thesis preprocessing: per-audio z-score normalization.
            audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)

            condition = conditions.get(filepath, "clean") if conditions else "clean"
            if condition == "clean":
                # Preserve the existing clean preprocessing path exactly.
                audio = peak_normalize(audio)
            else:
                audio = apply_augmentation(
                    audio,
                    condition,
                    room_rirs,
                    mobile_rirs,
                    urban_files,
                    rng=pair_rng(filepath),
                )

            windows = extract_windows(audio)
            logmel = np.stack([extract_logmel(w) for w in windows])
            mfcc = np.stack([extract_mfcc(w) for w in windows])
            cqcc = np.stack([extract_cqcc(w) for w in windows])
            f0 = np.stack([extract_f0(w) for w in windows])

            np.savez_compressed(
                out_path,
                logmel=logmel,
                mfcc=mfcc,
                cqcc=cqcc,
                f0=f0,
                label=np.array(label),
                speaker=np.array(speaker_id),
                condition=np.array(condition),
                n_windows=np.array(len(windows)),
                cache_version=np.array(CACHE_VERSION),
            )
            success += 1
        except Exception as exc:
            errors.append((os.path.basename(filepath), str(exc)))

    print(f"  Saved   : {success}")
    print(f"  Skipped : {skipped}")
    print(f"  Errors  : {len(errors)}")
    for name, reason in errors[:5]:
        print(f"    {name}: {reason}")
    return success


def main():
    set_seed(SEED)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    room_rirs, mobile_rirs, urban_files = ensure_resources()

    train_samples = load_all_files(TRAIN_DIR)
    test_samples = load_all_files(TEST_DIR)
    print(f"Training/development files: {len(train_samples):,}")
    print(f"Held-out test files        : {len(test_samples):,}")

    conditions = assign_fixed_conditions(train_samples)

    t0 = time.time()
    process_split(
        train_samples, "train", room_rirs, mobile_rirs, urban_files, conditions
    )
    print(f"Training cache time: {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    process_split(
        test_samples, "test", room_rirs, mobile_rirs, urban_files, conditions=None
    )
    print(f"Clean test cache time: {(time.time()-t0)/60:.1f} min")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
