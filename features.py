# features.py — Log-Mel, MFCC, F0 extraction
import numpy as np
import librosa
from scipy.ndimage import zoom
from config import (
    SAMPLE_RATE, N_MELS, N_MFCC, N_FFT,
    HOP_LENGTH, TARGET_SHAPE, F0_MIN, F0_MAX
)


def extract_logmel(window):
    """
    C.5: Log-Mel spectrogram
    STFT 25ms frame / 10ms hop / Hamming window
    → Mel filter bank → power to dB → resize 128×128 → normalize 0-1
    """
    mel = librosa.feature.melspectrogram(
        y          = window,
        sr         = SAMPLE_RATE,
        n_fft      = N_FFT,
        hop_length = HOP_LENGTH,
        n_mels     = N_MELS,
        window     = 'hamming',
        power      = 2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Resize to 128×128
    if log_mel.shape != TARGET_SHAPE:
        factors = (
            TARGET_SHAPE[0] / log_mel.shape[0],
            TARGET_SHAPE[1] / log_mel.shape[1]
        )
        log_mel = zoom(log_mel, factors)

    # Normalize 0-1
    mn, mx  = log_mel.min(), log_mel.max()
    log_mel = (log_mel - mn) / (mx - mn + 1e-8)

    return log_mel.astype(np.float32)


def extract_mfcc(window):
    """
    C.5: MFCC → 80-dim statistical vector
    40 coefficients × frames → mean + std → 80-dim
    """
    mfcc     = librosa.feature.mfcc(
        y          = window,
        sr         = SAMPLE_RATE,
        n_mfcc     = N_MFCC,
        n_fft      = N_FFT,
        hop_length = HOP_LENGTH
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std( mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)


def extract_f0(window):
    """
    C.5: F0 estimation using YIN algorithm.
    - Frame length: 2048 samples (128ms at 16kHz)
    - Hop length: 512 samples (32ms at 16kHz)
    - ~31 frames per 1-second window
    - Unvoiced frames excluded (below F0_MIN threshold)
    - Log transform applied before mean/std
      (F0 perception is logarithmic — musical pitch)
    - If no voiced frames found: returns [0.0, 0.0]
      These zero-F0 windows are included in training
      and the model learns that silence/noise has no pitch
    """
    from config import F0_FRAME_LENGTH, F0_HOP_LENGTH, F0_LOG

    f0 = librosa.yin(
        window,
        fmin         = F0_MIN,
        fmax         = F0_MAX,
        sr           = SAMPLE_RATE,
        hop_length   = F0_HOP_LENGTH,
        frame_length = F0_FRAME_LENGTH
    )

    # Exclude unvoiced frames
    voiced = f0[(f0 > F0_MIN) & (f0 < F0_MAX)]

    if len(voiced) == 0:
        # No voiced frames — return zeros
        # Model learns this as "no pitch information"
        return np.array([0.0, 0.0], dtype=np.float32)

    if F0_LOG:
        # Log transform before statistics
        # Captures perceptually meaningful pitch differences
        voiced = np.log(voiced + 1e-8)

    return np.array(
        [np.mean(voiced), np.std(voiced)],
        dtype=np.float32
    )


def extract_features(window):
    """Extract all three features from one 1-second window."""
    return {
        "logmel": extract_logmel(window),
        "mfcc"  : extract_mfcc(window),
        "f0"    : extract_f0(window)
    }


if __name__ == "__main__":
    import time
    test_window = np.random.randn(16000).astype(np.float32)

    t0       = time.perf_counter()
    features = extract_features(test_window)
    elapsed  = (time.perf_counter() - t0) * 1000

    print(f"Log-Mel : {features['logmel'].shape}")
    print(f"MFCC    : {features['mfcc'].shape}")
    print(f"F0      : {features['f0'].shape}")
    print(f"Time    : {elapsed:.2f}ms per window")
    print("\n✅ features.py OK")