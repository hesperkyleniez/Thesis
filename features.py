# features.py — Log-Mel, MFCC, CQCC, and F0 extraction
import numpy as np
import librosa
from scipy.ndimage import zoom
from scipy.fftpack import dct
from config import (
    SAMPLE_RATE, N_MELS, N_MFCC, N_CQCC, N_FFT,
    HOP_LENGTH, TARGET_SHAPE, F0_MIN, F0_MAX,
    CQCC_FMIN, CQCC_BINS, CQCC_BINS_PER_OCTAVE,
    F0_FRAME_LENGTH, F0_HOP_LENGTH, F0_LOG,
)


def extract_logmel(window):
    """128x128 normalized Log-Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=window, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, window="hamming", power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape != TARGET_SHAPE:
        factors = (
            TARGET_SHAPE[0] / log_mel.shape[0],
            TARGET_SHAPE[1] / log_mel.shape[1],
        )
        log_mel = zoom(log_mel, factors)

    mn, mx = log_mel.min(), log_mel.max()
    return ((log_mel - mn) / (mx - mn + 1e-8)).astype(np.float32)


def extract_mfcc(window):
    """40 MFCC coefficients pooled as mean + std -> 80 dimensions."""
    mfcc = librosa.feature.mfcc(
        y=window, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1)]
    ).astype(np.float32)


def extract_cqcc(window):
    """
    CQCC-style representation:
      CQT magnitude -> log power -> DCT along frequency -> 40 coefficients
      -> mean + std -> 80 dimensions.

    This is intentionally kept as a compact statistical vector so the
    auxiliary CQCC model stays cheap to train and easy to fuse at prediction
    level with the CNN-GRU model.
    """
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

    # DCT over the frequency axis.
    cqcc = dct(log_cqt, type=2, axis=0, norm="ortho")
    cqcc = cqcc[:N_CQCC]

    return np.concatenate(
        [cqcc.mean(axis=1), cqcc.std(axis=1)]
    ).astype(np.float32)


def extract_f0(window):
    """Log-F0 mean + std using YIN."""
    f0 = librosa.yin(
        window,
        fmin=F0_MIN,
        fmax=F0_MAX,
        sr=SAMPLE_RATE,
        hop_length=F0_HOP_LENGTH,
        frame_length=F0_FRAME_LENGTH,
    )

    voiced = f0[(f0 > F0_MIN) & (f0 < F0_MAX)]

    if len(voiced) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    if F0_LOG:
        voiced = np.log(voiced + 1e-8)

    return np.array(
        [np.mean(voiced), np.std(voiced)],
        dtype=np.float32,
    )


def extract_features(window):
    return {
        "logmel": extract_logmel(window),
        "mfcc": extract_mfcc(window),
        "cqcc": extract_cqcc(window),
        "f0": extract_f0(window),
    }


if __name__ == "__main__":
    import time

    test_window = np.random.randn(16000).astype(np.float32)
    t0 = time.perf_counter()
    features = extract_features(test_window)
    elapsed = (time.perf_counter() - t0) * 1000

    for name, value in features.items():
        print(f"{name:<7}: {value.shape}")

    print(f"Time    : {elapsed:.2f}ms per window")
    print("\nfeatures.py OK")
