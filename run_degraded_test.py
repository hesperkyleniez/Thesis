import sys
sys.path.insert(0, r"D:\Thesis\code")

import os
import random
import zipfile
import numpy as np
import librosa
from scipy.signal import fftconvolve, butter, filtfilt
from scipy.ndimage import zoom
from tqdm import tqdm
from collections import Counter

def set_seed(seed=42):
    """Sets seed for strict reproducibility in augmentations and assignments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🌱 Random seed set to {seed}")
    
from config import (
    TEST_DIR, FEATURES_DIR, AUGMENTATION_SPLITS,
    SAMPLE_RATE, WIN_SAMPLES, HOP_SAMPLES,
    N_MELS, N_MFCC, N_FFT, HOP_LENGTH, TARGET_SHAPE,
    F0_MIN, F0_MAX, OPENSLR_DIR, AIR_DIR, URBANSOUND_DIR,
    LABEL_REAL, LABEL_AI
)
from dataset import load_all_files

# ── Collect RIR and noise files ───────────────────────────────────────────────
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

# ── Augmentation ──────────────────────────────────────────────────────────────
def peak_normalize(audio):
    mx = np.max(np.abs(audio))
    return audio / (mx + 1e-8) if mx > 0 else audio

def apply_gaussian(audio):
    return audio + np.random.normal(0, 0.003,
                                    len(audio)).astype(np.float32)

def apply_urban(audio, snr_db=30.0):
    noise, _ = librosa.load(random.choice(urban_files),
                            sr=SAMPLE_RATE, mono=True)
    noise = noise.astype(np.float32)
    if len(noise) < len(audio):
        noise = np.tile(noise,
                        int(np.ceil(len(audio)/len(noise))))
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
    from scipy.signal import fftconvolve
    conv   = fftconvolve(audio, rir, mode='full')[:len(audio)]
    return peak_normalize((1-wet)*audio + wet*conv)

def apply_telephone(audio):
    threshold = 0.9
    ratio     = 1.2
    comp      = audio.copy()
    mask      = np.abs(comp) > threshold
    comp[mask]= (np.sign(comp[mask]) *
                 (threshold +
                  (np.abs(comp[mask])-threshold)/ratio))
    nyq      = SAMPLE_RATE / 2
    b, a     = butter(4, [300/nyq, 3400/nyq], btype='band')
    from scipy.signal import filtfilt
    filtered = filtfilt(b, a, comp)
    return peak_normalize(filtered)

def augment(audio, condition):
    if condition == "telephone":
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

# ── Feature extraction ────────────────────────────────────────────────────────
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
    return ((log_mel - mn) /
            (mx - mn + 1e-8)).astype(np.float32)

def extract_mfcc(window):
    mfcc = librosa.feature.mfcc(
        y=window, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return np.concatenate([mfcc.mean(axis=1),
                           mfcc.std(axis=1)]).astype(np.float32)

def extract_f0(window):
    """
    F0 using YIN. Frame: 2048 samples, Hop: 512 samples.
    Log transform applied before mean/std.
    Returns [0.0, 0.0] if no voiced frames.
    """
    from config import F0_FRAME_LENGTH, F0_HOP_LENGTH, F0_LOG
    f0 = librosa.yin(
        window, fmin=F0_MIN, fmax=F0_MAX,
        sr=SAMPLE_RATE,
        hop_length=F0_HOP_LENGTH,
        frame_length=F0_FRAME_LENGTH
    )
    voiced = f0[(f0 > F0_MIN) & (f0 < F0_MAX)]
    if len(voiced) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    if F0_LOG:
        voiced = np.log(voiced + 1e-8)
    return np.array(
        [np.mean(voiced), np.std(voiced)],
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

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    
    set_seed(42)
    
    test_samples = load_all_files(TEST_DIR)
    print(f"Test files: {len(test_samples)}")

    # Assign random degradation conditions
    # No clean — all degraded for robustness evaluation
    condition_names = [c for c in AUGMENTATION_SPLITS.keys()
                       if c != "clean"]

    test_conditions = {}
    pair_conditions = {}   # pair_key → condition
    for filepath, label, speaker_id in test_samples:
        fname    = os.path.splitext(os.path.basename(filepath))[0]
        pair_key = fname[:-3] if fname.endswith("_AI") else fname
        if pair_key not in pair_conditions:
            pair_conditions[pair_key] = random.choice(condition_names)
        test_conditions[filepath] = pair_conditions[pair_key]

    counts = Counter(test_conditions.values())
    print("\nDegraded test condition distribution:")
    for cond, count in sorted(counts.items()):
        pct = count / len(test_conditions) * 100
        print(f"  {cond:<20}: {count:>5} ({pct:.1f}%)")

    # Process
    out_dir = os.path.join(FEATURES_DIR, "test_degraded")
    os.makedirs(out_dir, exist_ok=True)
    cond_path = os.path.join(out_dir, "test_degraded_conditions.npy")
    np.save(cond_path, test_conditions)
    print(f"✅ Condition map saved to {cond_path}")

    success = 0
    skipped = 0
    errors  = []

    print(f"\nProcessing degraded test set...")
    t0 = time.time()

    for filepath, label, speaker_id in tqdm(test_samples):
        speaker_dir = os.path.join(out_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)

        fname = os.path.splitext(os.path.basename(filepath))[0]
        if label == LABEL_AI and not fname.endswith("_AI"):
            fname = fname + "_AI"
        elif label == LABEL_REAL and not fname.endswith("_Real"):
            fname = fname + "_Real"
        out_path = os.path.join(speaker_dir, fname + ".npz")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            audio, _ = librosa.load(filepath,
                                    sr=SAMPLE_RATE,
                                    mono=True)
            audio    = audio.astype(np.float32)

            # Z-score normalization
            mu    = np.mean(audio)
            sigma = np.std(audio)
            audio = (audio - mu) / (sigma + 1e-8)

            # Apply degradation
            condition = test_conditions[filepath]
            audio     = augment(audio, condition)

            # Final peak normalize
            audio = peak_normalize(audio)

            # Windows
            windows = extract_windows(audio)

            # Features
            logmel_list = []
            mfcc_list   = []
            f0_list     = []

            for window in windows:
                logmel_list.append(extract_logmel(window))
                mfcc_list.append(extract_mfcc(window))
                f0_list.append(extract_f0(window))

            np.savez_compressed(
                out_path,
                logmel    = np.stack(logmel_list),
                mfcc      = np.stack(mfcc_list),
                f0        = np.stack(f0_list),
                label     = np.array(label),
                speaker   = np.array(speaker_id),
                condition = np.array(condition),
                n_windows = np.array(len(windows))
            )
            success += 1

        except Exception as e:
            errors.append((os.path.basename(filepath), str(e)))

    elapsed = (time.time() - t0) / 60
    print(f"\n  ✅ Saved   : {success}")
    print(f"  ⏭️  Skipped : {skipped}")
    print(f"  ❌ Errors  : {len(errors)}")
    print(f"  Time      : {elapsed:.1f} minutes")

    if errors:
        for fname, reason in errors[:5]:
            print(f"    {fname}: {reason}")

    print(f"\n✅ Degraded test set saved to {out_dir}")