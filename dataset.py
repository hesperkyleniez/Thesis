# dataset.py — data loading, speaker ID extraction, fold assignment
import os
import re
import numpy as np
import librosa
from config import (
    TRAIN_DIR, TEST_DIR, FOLD_ASSIGNMENTS, TEST_SPEAKERS,
    SAMPLE_RATE, WIN_SAMPLES, HOP_SAMPLES,
    LABEL_REAL, LABEL_AI
)


def get_speaker_id(filepath):
    """
    Extracts speaker ID from filepath.
    Handles both G-speaker and numeric speaker naming.

    Examples:
      .../G0004/G0004_1_S0001.wav        → G0004
      .../0001/0001.110816.083827.wav    → 0001
      .../generated_clips_0001/0001.wav  → 0001
    """
    folder = os.path.basename(os.path.dirname(filepath))

    # AI dataset folders start with generated_clips_
    if folder.startswith("generated_clips_"):
        return folder.replace("generated_clips_", "")

    # G-speaker folders
    if re.match(r"^G\d{4}$", folder):
        return folder

    # Numeric speaker folders
    if re.match(r"^\d{4}$", folder):
        return folder

    # Fallback — try filename prefix
    fname = os.path.basename(filepath)
    match = re.match(r"^(G?\d{4})", fname)
    if match:
        return match.group(1)

    raise ValueError(f"Cannot extract speaker ID from: {filepath}")


def load_all_files(base_dir):
    """
    Loads all WAV file paths and labels from a dataset folder.
    Returns list of (filepath, label, speaker_id) tuples.
    """
    samples = []

    for dataset_type, label in [
        ("Real_Dataset", LABEL_REAL),
        ("AI_Dataset",   LABEL_AI)
    ]:
        folder = os.path.join(base_dir, dataset_type)
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue

        for speaker_folder in os.listdir(folder):
            speaker_path = os.path.join(folder, speaker_folder)
            if not os.path.isdir(speaker_path):
                continue

            for fname in os.listdir(speaker_path):
                if not fname.lower().endswith(".wav"):
                    continue
                filepath   = os.path.join(speaker_path, fname)
                speaker_id = get_speaker_id(filepath)
                samples.append((filepath, label, speaker_id))

    return samples


def get_fold_splits(samples):
    """
    Splits samples into 4 folds based on speaker ID.
    Returns dict: fold_num → {train: [...], val: [...]}
    """
    folds = {}

    for fold_num in [1, 2, 3, 4]:
        train = []
        val   = []

        for filepath, label, speaker_id in samples:
            assigned_fold = FOLD_ASSIGNMENTS.get(speaker_id)

            if assigned_fold is None:
                print(f"⚠️  Speaker {speaker_id} not in fold assignments — skipping")
                continue

            if assigned_fold == fold_num:
                val.append((filepath, label, speaker_id))
            else:
                train.append((filepath, label, speaker_id))

        folds[fold_num] = {"train": train, "val": val}

    return folds


def load_audio(filepath):
    """Load WAV file — already 16kHz mono so no conversion needed."""
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def extract_windows(audio):
    """
    C.4: Segment audio into 1-second windows with 50% overlap.
    Zero-pad final incomplete window.
    Returns list of numpy arrays each of length WIN_SAMPLES.
    """
    windows = []
    start   = 0

    while start < len(audio):
        window = audio[start:start + WIN_SAMPLES]

        if len(window) < WIN_SAMPLES:
            window = np.pad(window, (0, WIN_SAMPLES - len(window)))

        windows.append(window.astype(np.float32))
        start += HOP_SAMPLES

    return windows


def assign_conditions(samples):
    """
    Randomly assigns augmentation condition to each file.
    Reassigned every epoch for diversity.

    Conditions:
      clean           50%
      telephone       25%
      room_gaussian   6.25%
      mobile_gaussian 6.25%
      mobile_urban    6.25%
      room_urban      6.25%
    """
    from config import AUGMENTATION_SPLITS

    conditions   = list(AUGMENTATION_SPLITS.keys())
    probabilities= list(AUGMENTATION_SPLITS.values())

    result = {}
    for filepath, label, speaker_id in samples:
        condition = np.random.choice(conditions, p=probabilities)
        result[filepath] = condition

    return result


if __name__ == "__main__":
    print("Loading dataset...")
    samples = load_all_files(TRAIN_DIR)
    print(f"Total training files: {len(samples)}")

    real_count = sum(1 for _, l, _ in samples if l == LABEL_REAL)
    ai_count   = sum(1 for _, l, _ in samples if l == LABEL_AI)
    print(f"  Real : {real_count}")
    print(f"  AI   : {ai_count}")

    # Check fold distribution
    folds = get_fold_splits(samples)
    for fold_num, split in folds.items():
        print(f"  Fold {fold_num} — Train: {len(split['train'])} | Val: {len(split['val'])}")

    # Test windowing
    test_audio = np.random.randn(SAMPLE_RATE * 5).astype(np.float32)
    windows    = extract_windows(test_audio)
    print(f"\nWindowing test: 5s audio → {len(windows)} windows")
    print(f"  Window shape: {windows[0].shape}")

    print("\n✅ dataset.py OK")