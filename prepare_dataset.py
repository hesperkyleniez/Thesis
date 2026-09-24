# prepare_dataset.py
# Run ONCE before training
# Does: augmentation → windowing → feature extraction → save NPZ cache

import os
import sys
import random
import zipfile
import urllib.request
import numpy as np
import librosa
from scipy.signal import fftconvolve, butter, filtfilt
from scipy.ndimage import zoom
from scipy.fftpack import dct
from tqdm import tqdm

def set_seed(seed=42):
    """Sets seed for strict reproducibility in augmentations and assignments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🌱 Random seed set to {seed}")


sys.path.insert(0, r"D:\Thesis\code")
from config import (
    TRAIN_DIR, TEST_DIR,
    SAMPLE_RATE, WIN_SAMPLES, HOP_SAMPLES,
    N_MELS, N_MFCC, N_CQCC, N_FFT, HOP_LENGTH, TARGET_SHAPE,
    CQCC_FMIN, CQCC_BINS, CQCC_BINS_PER_OCTAVE,
    F0_MIN, F0_MAX,
    OPENSLR_DIR, AIR_DIR, URBANSOUND_DIR, RESOURCES_DIR,
    FOLD_ASSIGNMENTS, AUGMENTATION_SPLITS,
    LABEL_REAL, LABEL_AI
)
from dataset import load_all_files, get_speaker_id

# ── Output directory ──────────────────────────────────────────────────────────
FEATURES_DIR = r"D:\Thesis\features"
CACHE_VERSION = "v9_cqcc"
os.makedirs(FEATURES_DIR, exist_ok=True)

# ── Step 1: Extract zip files if needed ──────────────────────────────────────
def extract_if_needed(zip_path, out_dir):
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        return
    print(f"Extracting {zip_path}...")
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    print(f"✅ Extracted to {out_dir}")

extract_if_needed(
    os.path.join(RESOURCES_DIR, "rirs_noises.zip"),
    OPENSLR_DIR
)
extract_if_needed(
    os.path.join(RESOURCES_DIR, "archive.zip"),
    URBANSOUND_DIR
)
extract_if_needed(
    os.path.join(RESOURCES_DIR, "AIR_wav_files.zip"),
    AIR_DIR
)

# ── Step 2: Collect RIR and noise files ───────────────────────────────────────
def collect_wav_files(folder):
    files = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(root, f))
    return files

room_rirs   = []
mobile_rirs = []
urban_files = []

for path in collect_wav_files(OPENSLR_DIR):
    lower = path.lower()
    if "real_rirs" in lower or "real_rir" in lower:
        room_rirs.append(path)
    elif "simulated" in lower or "pointsource" in lower:
        mobile_rirs.append(path)
    elif "rir" in lower:
        room_rirs.append(path)

for path in collect_wav_files(AIR_DIR):
    mobile_rirs.append(path)

for path in collect_wav_files(URBANSOUND_DIR):
    urban_files.append(path)

room_rirs   = sorted(set(room_rirs))
mobile_rirs = sorted(set(mobile_rirs))

print(f"Room RIRs   : {len(room_rirs)}")
print(f"Mobile RIRs : {len(mobile_rirs)}")
print(f"Urban noise : {len(urban_files)}")

if not room_rirs:
    raise RuntimeError("No room RIRs found")
if not mobile_rirs:
    raise RuntimeError("No mobile RIRs found")
if not urban_files:
    raise RuntimeError("No urban noise files found")

# ── Step 3: Augmentation functions ───────────────────────────────────────────
def peak_normalize(audio):
    mx = np.max(np.abs(audio))
    return audio / (mx + 1e-8) if mx > 0 else audio

def apply_gaussian(audio):
    return audio + np.random.normal(0, 0.003, len(audio)).astype(np.float32)

def apply_urban(audio, snr_db=30.0):
    noise, _ = librosa.load(random.choice(urban_files),
                            sr=SAMPLE_RATE, mono=True)
    noise = noise.astype(np.float32)
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio)/len(noise))))
    noise = noise[:len(audio)]
    p_s   = np.mean(audio**2) + 1e-8
    p_n   = np.mean(noise**2) + 1e-8
    scale = np.sqrt(p_s / (p_n * (10**(snr_db/10))))
    return peak_normalize(audio + noise * scale)

def apply_rir(audio, rir_list, wet=0.04):
    rir, _ = librosa.load(random.choice(rir_list),
                          sr=SAMPLE_RATE, mono=True)
    rir    = rir[:int(0.1*SAMPLE_RATE)]
    rir    = rir / (np.max(np.abs(rir)) + 1e-8)
    conv   = fftconvolve(audio, rir, mode='full')[:len(audio)]
    return peak_normalize((1-wet)*audio + wet*conv)

def apply_telephone(audio):
    # Soft-knee DRC
    threshold = 0.9
    ratio     = 1.2
    comp      = audio.copy()
    mask      = np.abs(comp) > threshold
    comp[mask]= (np.sign(comp[mask]) *
                 (threshold + (np.abs(comp[mask])-threshold)/ratio))
    # Telephone bandpass 300-3400Hz
    nyq      = SAMPLE_RATE / 2
    b, a     = butter(4, [300/nyq, 3400/nyq], btype='band')
    filtered = filtfilt(b, a, comp)
    return peak_normalize(filtered)

def augment(audio, condition):
    if condition == "clean":
        return audio
    elif condition == "telephone":
        return apply_telephone(audio)
    elif condition == "room_gaussian":
        return apply_rir(apply_gaussian(audio), room_rirs)
    elif condition == "mobile_gaussian":
        return apply_rir(apply_gaussian(audio), mobile_rirs)
    elif condition == "mobile_urban":
        return apply_rir(apply_urban(audio), mobile_rirs)
    elif condition == "room_urban":
        return apply_rir(apply_urban(audio), room_rirs)
    return audio

# ── Step 4: Feature extraction ────────────────────────────────────────────────
def extract_logmel(window):
    mel     = librosa.feature.melspectrogram(
        y=window, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, window='hamming', power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape != TARGET_SHAPE:
        factors = (TARGET_SHAPE[0]/log_mel.shape[0],
                   TARGET_SHAPE[1]/log_mel.shape[1])
        log_mel = zoom(log_mel, factors)
    mn, mx  = log_mel.min(), log_mel.max()
    return ((log_mel - mn) / (mx - mn + 1e-8)).astype(np.float32)

def extract_mfcc(window):
    mfcc = librosa.feature.mfcc(
        y=window, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return np.concatenate([mfcc.mean(axis=1),
                           mfcc.std(axis=1)]).astype(np.float32)

def extract_cqcc(window):
    """CQCC-style feature: log CQT -> DCT -> mean/std (80 dimensions)."""
    cqt = librosa.cqt(
        y=window,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=CQCC_FMIN,
        n_bins=CQCC_BINS,
        bins_per_octave=CQCC_BINS_PER_OCTAVE,
        window="hann",
        scale=True,
    )
    power = np.abs(cqt) ** 2
    log_cqt = np.log(power + 1e-10)
    cqcc = dct(log_cqt, type=2, axis=0, norm="ortho")[:N_CQCC]
    return np.concatenate(
        [cqcc.mean(axis=1), cqcc.std(axis=1)]
    ).astype(np.float32)

def extract_f0(window):
    """
    C.5: F0 using YIN.
    Frame: 2048 samples (128ms), Hop: 512 samples (32ms)
    ~31 frames per 1-second window.
    Log transform applied before mean/std.
    Returns [0.0, 0.0] if no voiced frames detected.
    """
    f0 = librosa.yin(
        window, fmin=F0_MIN, fmax=F0_MAX,
        sr=SAMPLE_RATE,
        hop_length=512,
        frame_length=2048
    )
    voiced = f0[(f0 > F0_MIN) & (f0 < F0_MAX)]
    if len(voiced) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    # Log transform — F0 perception is logarithmic
    voiced_log = np.log(voiced + 1e-8)
    return np.array(
        [np.mean(voiced_log), np.std(voiced_log)],
        dtype=np.float32
    )

def extract_windows(audio):
    windows = []
    start   = 0
    while start < len(audio):
        w = audio[start:start+WIN_SAMPLES]
        if len(w) < WIN_SAMPLES:
            w = np.pad(w, (0, WIN_SAMPLES-len(w)))
        windows.append(w.astype(np.float32))
        start += HOP_SAMPLES
    return windows

# ── Step 5: Assign fixed augmentation conditions ──────────────────────────────
def assign_fixed_conditions(samples):
    conditions_path = os.path.join(FEATURES_DIR, "conditions.npy")

    if os.path.exists(conditions_path):
        print("✅ Loading existing condition assignments...")
        return dict(np.load(conditions_path, allow_pickle=True).item())

    print("Assigning paired augmentation conditions...")
    condition_names = list(AUGMENTATION_SPLITS.keys())
    condition_probs = list(AUGMENTATION_SPLITS.values())

    # Build pair key → condition mapping
    # Real: G0008_1_S0001.wav      → key: G0008_1_S0001
    # AI:   G0008_1_S0001_AI.wav   → key: G0008_1_S0001
    # Real: 0029.110825.065356.0001.wav → key: 0029.110825.065356.0001
    # AI:   0029.110825.065356.0001.wav → key: same

    pair_conditions = {}   # pair_key → condition
    assignments     = {}   # filepath → condition

    for filepath, label, speaker_id in samples:
        fname    = os.path.splitext(os.path.basename(filepath))[0]
        # Remove _AI suffix to get the pair key
        pair_key = fname[:-3] if fname.endswith("_AI") else fname

        # Assign condition to pair key if not already assigned
        if pair_key not in pair_conditions:
            pair_conditions[pair_key] = np.random.choice(
                condition_names, p=condition_probs
            )
        assignments[filepath] = pair_conditions[pair_key]

    # Print distribution
    from collections import Counter
    counts = Counter(assignments.values())
    print("  Paired condition distribution:")
    for cond, count in sorted(counts.items()):
        pct = count / len(assignments) * 100
        print(f"    {cond:<20}: {count:>6} ({pct:.1f}%)")

    # Verify pairing — check a few pairs
    print("\n  Verifying paired conditions (first 3 pairs):")
    shown = 0
    pair_check = {}
    for filepath, label, speaker_id in samples:
        fname    = os.path.splitext(os.path.basename(filepath))[0]
        pair_key = fname[:-3] if fname.endswith("_AI") else fname
        if pair_key not in pair_check:
            pair_check[pair_key] = {}
        pair_check[pair_key]['AI' if label == LABEL_AI else 'Real'] = \
            assignments[filepath]

    for pair_key, pair_data in list(pair_check.items())[:3]:
        if 'Real' in pair_data and 'AI' in pair_data:
            match = "✅" if pair_data['Real'] == pair_data['AI'] else "❌"
            print(f"    {pair_key[:40]}")
            print(f"      Real: {pair_data['Real']} | AI: {pair_data['AI']} {match}")
            shown += 1

    np.save(conditions_path, assignments)
    print(f"\n✅ Conditions saved to {conditions_path}")
    return assignments
    
    
# ── Step 6: Process and save features ────────────────────────────────────────
def process_split(samples, split_name, conditions=None):
    """
    Process all files in a split:
    augment → window → extract features → save NPZ
    """
    out_dir = os.path.join(FEATURES_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)

    success  = 0
    skipped  = 0
    errors   = []

    print(f"\n{'='*55}")
    print(f"Processing: {split_name} ({len(samples)} files)")
    print(f"{'='*55}")

    for filepath, label, speaker_id in tqdm(samples):
        # Output path mirrors speaker folder structure
        speaker_dir = os.path.join(out_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)

        fname = os.path.splitext(os.path.basename(filepath))[0]

        # Avoid double _AI suffix for G-speaker AI files
        # that already contain _AI in their filename
        if label == LABEL_AI and not fname.endswith("_AI"):
            fname = fname + "_AI"
        elif label == LABEL_REAL and not fname.endswith("_Real"):
            fname = fname + "_Real"

        out_path = os.path.join(speaker_dir, fname + ".npz")

        # Reuse only v9 caches. Older caches do not contain CQCC.
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
                print(f"  Rebuilding cache with CQCC: {os.path.basename(out_path)}")
            except Exception:
                print(f"  Rebuilding unreadable cache: {os.path.basename(out_path)}")

        try:
            # Load audio
            audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
            audio    = audio.astype(np.float32)

            # Z-score normalization
            mu     = np.mean(audio)
            sigma  = np.std(audio)
            audio  = (audio - mu) / (sigma + 1e-8)

            # Apply augmentation (fixed condition)
            if conditions is not None:
                condition = conditions.get(filepath, "clean")
            else:
                condition = "clean"   # test set always clean
            audio = augment(audio, condition)

            # Final peak normalization
            audio = peak_normalize(audio)

            # Extract windows
            windows = extract_windows(audio)

            # Extract features per window
            logmel_list = []
            mfcc_list   = []
            cqcc_list  = []
            f0_list     = []

            for window in windows:
                logmel_list.append(extract_logmel(window))
                mfcc_list.append(extract_mfcc(window))
                cqcc_list.append(extract_cqcc(window))
                f0_list.append(extract_f0(window))

            # Save as NPZ
            np.savez_compressed(
                out_path,
                logmel    = np.stack(logmel_list),   # (n_win, 128, 128)
                mfcc      = np.stack(mfcc_list),     # (n_win, 80)
                cqcc      = np.stack(cqcc_list),      # (n_win, 80)
                f0        = np.stack(f0_list),       # (n_win, 2)
                label     = np.array(label),
                speaker   = np.array(speaker_id),
                condition = np.array(condition),
                n_windows = np.array(len(windows)),
                cache_version = np.array(CACHE_VERSION)
            )
            success += 1

        except Exception as e:
            errors.append((os.path.basename(filepath), str(e)))

    print(f"\n  ✅ Saved   : {success}")
    print(f"  ⏭️  Skipped : {skipped}")
    print(f"  ❌ Errors  : {len(errors)}")
    if errors:
        for fname, reason in errors[:5]:
            print(f"    {fname}: {reason}")

    return success

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    
    set_seed(42)
    
    print("="*55)
    print("DATASET PREPARATION")
    print("="*55)

    # Load all files
    print("\nLoading training files...")
    train_samples = load_all_files(TRAIN_DIR)
    print(f"Total training files: {len(train_samples)}")

    print("\nLoading test files...")
    test_samples = load_all_files(TEST_DIR)
    print(f"Total test files: {len(test_samples)}")

    # Assign fixed conditions to training files
    conditions = assign_fixed_conditions(train_samples)

    # Process training set
    t0 = time.time()
    process_split(train_samples, "train", conditions)
    print(f"\nTraining features done in {(time.time()-t0)/60:.1f} minutes")

    # Process test set (no augmentation — always clean)
    t0 = time.time()
    process_split(test_samples, "test", conditions=None)
    print(f"\nTest features done in {(time.time()-t0)/60:.1f} minutes")

    # Summary
    def count_npz(folder):
        n = 0
        for _, _, files in os.walk(folder):
            n += sum(1 for f in files if f.endswith(".npz"))
        return n

    print("\n" + "="*55)
    print("FEATURE CACHE SUMMARY")
    print("="*55)
    print(f"  Train NPZ files: {count_npz(os.path.join(FEATURES_DIR, 'train'))}")
    print(f"  Test NPZ files : {count_npz(os.path.join(FEATURES_DIR, 'test'))}")
    print(f"\n✅ Dataset preparation complete")
    print(f"   Features saved to: {FEATURES_DIR}")