"""Build the held-out degraded test cache.

The degradation assignment map is preserved.  If a previous
`test_degraded_conditions.npy` exists, it is reused verbatim.  If it does not,
the same seeded assignment rule used by the previous code is used: each paired
Real/AI clip is assigned one of the five existing degraded condition labels.
"""

import hashlib
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import librosa
import numpy as np
from tqdm import tqdm

sys.path.insert(0, r"D:\Thesis\code")

from config import (
    TEST_DIR,
    FEATURES_DIR,
    AUGMENTATION_SPLITS,
    SAMPLE_RATE,
    LABEL_REAL,
    LABEL_AI,
    SEED,
    PREP_WORKERS,
    COMPRESS_FEATURE_CACHE,
)
from dataset import load_all_files
from augmentation import collect_rir_files, apply_augmentation
from features import extract_clip_features

CACHE_VERSION = "v14_fast_clip_cqcc_b12_normffv_yin"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def pair_key(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    return name[:-3] if name.endswith("_AI") else name


def pair_rng(filepath):
    digest = hashlib.sha256(f"degraded-test|{SEED}|{pair_key(filepath)}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def load_or_create_condition_map(test_samples, cond_path):
    """Preserve the existing degraded assignment split whenever available."""
    if os.path.exists(cond_path):
        existing = dict(np.load(cond_path, allow_pickle=True).item())
        missing = [path for path, _, _ in test_samples if path not in existing]
        if not missing:
            print(f"Using existing degraded condition map: {cond_path}")
            return existing
        print(f"Existing condition map is missing {len(missing)} files; rebuilding deterministically.")

    # This intentionally matches the previous test assignment logic: equal
    # random choice among the five non-clean condition labels, paired by clip.
    condition_names = [c for c in AUGMENTATION_SPLITS if c != "clean"]
    pair_conditions = {}
    assignments = {}

    for filepath, _, _ in test_samples:
        key = pair_key(filepath)
        if key not in pair_conditions:
            pair_conditions[key] = random.choice(condition_names)
        assignments[filepath] = pair_conditions[key]

    np.save(cond_path, assignments)
    print(f"Saved degraded condition map: {cond_path}")
    return assignments


def output_path(out_dir, filepath, label, speaker_id):
    speaker_dir = os.path.join(out_dir, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(filepath))[0]
    if label == LABEL_AI and not name.endswith("_AI"):
        name += "_AI"
    elif label == LABEL_REAL and not name.endswith("_Real"):
        name += "_Real"
    return os.path.join(speaker_dir, name + ".npz")


def _cache_is_current(path):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as d:
            return (
                all(k in d.files for k in ("logmel", "mfcc", "cqcc", "f0", "ffv"))
                and "cache_version" in d.files
                and str(d["cache_version"]) == CACHE_VERSION
            )
    except Exception:
        return False


def _save_cache(path, payload):
    tmp = path + ".tmp.npz"
    saver = np.savez_compressed if COMPRESS_FEATURE_CACHE else np.savez
    saver(tmp, **payload)
    os.replace(tmp, path)


def _process_one(sample, out_dir, conditions, room_rirs, mobile_rirs, urban_files):
    filepath, label, speaker_id = sample
    out_path = output_path(out_dir, filepath, label, speaker_id)
    if _cache_is_current(out_path):
        return "skipped", None
    try:
        audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
        condition = conditions[filepath]
        audio = apply_augmentation(
            audio, condition, room_rirs, mobile_rirs, urban_files,
            rng=pair_rng(filepath),
        )
        feat = extract_clip_features(audio)
        _save_cache(out_path, dict(
            **feat, label=np.array(label), speaker=np.array(speaker_id),
            condition=np.array(condition), n_windows=np.array(len(feat["cqcc"])),
            cache_version=np.array(CACHE_VERSION),
        ))
        return "saved", None
    except Exception as exc:
        return "error", (os.path.basename(filepath), str(exc))


def main():
    set_seed(SEED)
    test_samples = load_all_files(TEST_DIR)
    print(f"Held-out test files: {len(test_samples):,}")

    room_rirs, mobile_rirs, urban_files = collect_rir_files()
    print(f"Room RIRs   : {len(room_rirs)}")
    print(f"Mobile RIRs : {len(mobile_rirs)}")
    print(f"Urban noise : {len(urban_files)}")

    out_dir = os.path.join(FEATURES_DIR, "test_degraded")
    os.makedirs(out_dir, exist_ok=True)
    cond_path = os.path.join(out_dir, "test_degraded_conditions.npy")
    conditions = load_or_create_condition_map(test_samples, cond_path)

    counts = Counter(conditions[path] for path, _, _ in test_samples)
    print("\nDegraded test condition distribution (assignment split preserved):")
    for condition, count in sorted(counts.items()):
        print(f"  {condition:<20}: {count:>5} ({100*count/len(test_samples):.1f}%)")

    success = skipped = 0
    errors = []
    t0 = time.time()
    workers = max(1, int(PREP_WORKERS))
    print(f"Using {workers} feature-preparation worker(s).")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _process_one, sample, out_dir, conditions,
                room_rirs, mobile_rirs, urban_files,
            )
            for sample in test_samples
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Degraded test"):
            status, err = fut.result()
            success += status == "saved"
            skipped += status == "skipped"
            if err is not None:
                errors.append(err)

    print(f"\nSaved   : {success}")
    print(f"Skipped : {skipped}")
    print(f"Errors  : {len(errors)}")
    print(f"Time    : {(time.time()-t0)/60:.1f} min")
    for name, reason in errors[:5]:
        print(f"  {name}: {reason}")
    print(f"\nDegraded test cache: {out_dir}")


if __name__ == "__main__":
    main()
