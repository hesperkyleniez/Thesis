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
)
from dataset import load_all_files, extract_windows
from augmentation import collect_rir_files, apply_augmentation
from features import extract_logmel, extract_mfcc, extract_cqcc, extract_f0

CACHE_VERSION = "v11_robust_telephone"


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

    for filepath, label, speaker_id in tqdm(test_samples, desc="Degraded test"):
        out_path = output_path(out_dir, filepath, label, speaker_id)

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
            audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)

            condition = conditions[filepath]
            audio = apply_augmentation(
                audio,
                condition,
                room_rirs,
                mobile_rirs,
                urban_files,
                rng=pair_rng(filepath),
            )

            windows = extract_windows(audio)
            np.savez_compressed(
                out_path,
                logmel=np.stack([extract_logmel(w) for w in windows]),
                mfcc=np.stack([extract_mfcc(w) for w in windows]),
                cqcc=np.stack([extract_cqcc(w) for w in windows]),
                f0=np.stack([extract_f0(w) for w in windows]),
                label=np.array(label),
                speaker=np.array(speaker_id),
                condition=np.array(condition),
                n_windows=np.array(len(windows)),
                cache_version=np.array(CACHE_VERSION),
            )
            success += 1
        except Exception as exc:
            errors.append((os.path.basename(filepath), str(exc)))

    print(f"\nSaved   : {success}")
    print(f"Skipped : {skipped}")
    print(f"Errors  : {len(errors)}")
    print(f"Time    : {(time.time()-t0)/60:.1f} min")
    for name, reason in errors[:5]:
        print(f"  {name}: {reason}")
    print(f"\nDegraded test cache: {out_dir}")


if __name__ == "__main__":
    main()
