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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    PREP_WORKERS,
    COMPRESS_FEATURE_CACHE,
)
from dataset import load_all_files
from augmentation import collect_rir_files, apply_augmentation, peak_normalize
from features import extract_clip_features

CACHE_VERSION = "v14_fast_clip_cqcc_b12_normffv_yin"


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


def _cache_is_current(out_path):
    if not os.path.exists(out_path):
        return False
    try:
        with np.load(out_path, allow_pickle=True) as cached:
            return (
                all(k in cached.files for k in ("logmel", "mfcc", "cqcc", "f0", "ffv"))
                and "cache_version" in cached.files
                and str(cached["cache_version"]) == CACHE_VERSION
            )
    except Exception:
        return False


def _save_cache(out_path, payload):
    # Atomic replace means Ctrl+C cannot leave a half-written cache that looks valid.
    tmp_path = out_path + ".tmp.npz"
    saver = np.savez_compressed if COMPRESS_FEATURE_CACHE else np.savez
    saver(tmp_path, **payload)
    os.replace(tmp_path, out_path)


def _process_one(sample, split_name, room_rirs, mobile_rirs, urban_files, conditions):
    filepath, label, speaker_id = sample
    out_path = _output_path(split_name, filepath, label, speaker_id)
    if _cache_is_current(out_path):
        return "skipped", None

    try:
        audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)

        condition = conditions.get(filepath, "clean") if conditions else "clean"
        if condition == "clean":
            audio = peak_normalize(audio)
        else:
            audio = apply_augmentation(
                audio, condition, room_rirs, mobile_rirs, urban_files,
                rng=pair_rng(filepath),
            )

        feat = extract_clip_features(audio)
        n_windows = len(feat["cqcc"])
        payload = dict(
            **feat,
            label=np.array(label),
            speaker=np.array(speaker_id),
            condition=np.array(condition),
            n_windows=np.array(n_windows),
            cache_version=np.array(CACHE_VERSION),
        )
        _save_cache(out_path, payload)
        return "saved", None
    except Exception as exc:
        return "error", (os.path.basename(filepath), str(exc))


def process_split(samples, split_name, room_rirs, mobile_rirs, urban_files, conditions=None):
    success = skipped = 0
    errors = []
    workers = max(1, int(PREP_WORKERS))
    print(f"\nProcessing {split_name}: {len(samples):,} files with {workers} worker(s)")

    if workers == 1:
        iterator = (
            _process_one(s, split_name, room_rirs, mobile_rirs, urban_files, conditions)
            for s in tqdm(samples)
        )
        for status, err in iterator:
            success += status == "saved"
            skipped += status == "skipped"
            if err is not None:
                errors.append(err)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _process_one, s, split_name, room_rirs, mobile_rirs,
                    urban_files, conditions,
                )
                for s in samples
            ]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                status, err = fut.result()
                success += status == "saved"
                skipped += status == "skipped"
                if err is not None:
                    errors.append(err)

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
