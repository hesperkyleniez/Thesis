# features.py — Log-Mel, MFCC, canonical CQCC, and F0 extraction
import numpy as np
import librosa
from scipy.ndimage import zoom
from scipy.fft import dct
from scipy.interpolate import CubicSpline

from config import (
    SAMPLE_RATE, N_MELS, N_MFCC, N_FFT,
    HOP_LENGTH, TARGET_SHAPE, F0_MIN, F0_MAX,
    CQCC_BINS_PER_OCTAVE, CQCC_FMAX, CQCC_FMIN,
    CQCC_N_BINS, CQCC_UNIFORM_SAMPLES_FIRST_OCTAVE,
    CQCC_NUM_STATIC, CQCC_HOP_LENGTH,
    F0_FRAME_LENGTH, F0_HOP_LENGTH, F0_LOG, F0_VECTOR_DIM,
    FFV_FRAME_MS, FFV_HOP_MS, FFV_SEPARATION_MS, FFV_N_FFT, FFV_NUM_FILTERS,
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


def _cqcc_frequency_axes():
    """Geometric CQT centres and Todisco-style linearized frequency grid."""
    geometric = CQCC_FMIN * (
        2.0 ** (np.arange(CQCC_N_BINS, dtype=np.float64) / CQCC_BINS_PER_OCTAVE)
    )

    # Todisco et al. define
    #   kl = B*log2(1 + 1/d)
    # and resample at 1/[fmin*(2^(kl/B)-1)] samples/Hz.
    # Algebraically, the corresponding frequency step is fmin/d.
    step_hz = CQCC_FMIN / CQCC_UNIFORM_SAMPLES_FIRST_OCTAVE
    uniform = np.arange(
        CQCC_FMIN,
        geometric[-1] + 0.5 * step_hz,
        step_hz,
        dtype=np.float64,
    )
    return geometric, uniform


_CQCC_GEOMETRIC_FREQS, _CQCC_UNIFORM_FREQS = _cqcc_frequency_axes()


def extract_cqcc_frames(window):
    """Extract canonical static CQCC frames before fixed-length pooling.

    The front-end follows Todisco, Delgado & Evans (Odyssey 2016 / CSL 2017):
      CQT -> log power -> uniform frequency resampling -> DCT.

    Reference defaults retained here:
      * 96 bins per octave
      * fmax = Nyquist
      * fmin chosen near 20 Hz to span an integer number of octaves
      * d = 16 uniform samples in the first octave
      * 20 static coefficients including C0 (equivalent to cf=19, Zs)

    librosa.vqt is used instead of librosa.cqt because it exposes the non-zero
    gamma bandwidth term used by the reference CQCC implementation.  The
    resulting time-frequency matrix is then linearized with cubic-spline
    interpolation before the DCT, matching the defining CQCC resampling step.

    Returns
    -------
    np.ndarray, shape (20, n_frames)
        Static CQCC coefficients including C0.
    """
    x = np.asarray(window, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("CQCC input window is empty.")

    # Reference CQCC bandwidth offset:
    # gamma = 228.7 * (2^(1/B) - 2^(-1/B))
    B = float(CQCC_BINS_PER_OCTAVE)
    gamma = 228.7 * (2.0 ** (1.0 / B) - 2.0 ** (-1.0 / B))

    cqt = librosa.vqt(
        y=x,
        sr=SAMPLE_RATE,
        hop_length=CQCC_HOP_LENGTH,
        fmin=CQCC_FMIN,
        n_bins=CQCC_N_BINS,
        bins_per_octave=CQCC_BINS_PER_OCTAVE,
        gamma=gamma,
        window="hann",
        scale=True,
    )

    log_power = np.log(np.abs(cqt).astype(np.float64) ** 2 + np.finfo(float).eps)

    # Canonical CQCC step that the previous implementation omitted:
    # linearize the geometrically spaced CQT frequency axis before the DCT.
    spline = CubicSpline(
        _CQCC_GEOMETRIC_FREQS,
        log_power,
        axis=0,
        bc_type="not-a-knot",
        extrapolate=False,
    )
    uniform_log_power = spline(_CQCC_UNIFORM_FREQS)

    if not np.all(np.isfinite(uniform_log_power)):
        raise FloatingPointError("Non-finite values produced during CQCC resampling.")

    cepstrum = dct(uniform_log_power, type=2, axis=0, norm="ortho")
    static = cepstrum[:CQCC_NUM_STATIC]
    return static.astype(np.float32)


def extract_cqcc(window):
    """Canonical CQCC pooled as frame mean + std -> 40 dimensions.

    The CQCC *extraction* follows Todisco et al.  The mean+std summarization is
    the fixed-length strategy used by Akıncı & Özbay (Applied Sciences, 2026)
    for cepstral deepfake-speech features.  Pooling is applied only after the
    canonical CQCC resampling + DCT stages have been completed.
    """
    cqcc = extract_cqcc_frames(window)
    return np.concatenate(
        [cqcc.mean(axis=1), cqcc.std(axis=1)]
    ).astype(np.float32)


def extract_f0(window):
    """Richer F0/prosody summary using probabilistic YIN (pYIN) -> 8-D.

    Retains the thesis F0 cue but avoids collapsing the contour to only two
    numbers. Features are: log-F0 mean/std/median/IQR, delta-log-F0 mean
    absolute change/std, voiced fraction, and voicing-transition rate.
    """
    x = np.asarray(window, dtype=np.float32)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        x, fmin=F0_MIN, fmax=F0_MAX, sr=SAMPLE_RATE,
        frame_length=F0_FRAME_LENGTH, hop_length=F0_HOP_LENGTH,
        fill_na=np.nan,
    )
    valid = np.isfinite(f0) & (f0 >= F0_MIN) & (f0 <= F0_MAX)
    voiced_fraction = float(np.mean(valid)) if len(valid) else 0.0
    vf = np.asarray(voiced_flag, dtype=bool) if voiced_flag is not None else valid
    transition_rate = float(np.mean(vf[1:] != vf[:-1])) if len(vf) > 1 else 0.0
    vals = f0[valid]
    if len(vals) == 0:
        return np.zeros(F0_VECTOR_DIM, dtype=np.float32)
    if F0_LOG:
        vals = np.log(vals + 1e-8)
    q25, q75 = np.percentile(vals, [25, 75])
    delta = np.diff(vals)
    d_abs = float(np.mean(np.abs(delta))) if len(delta) else 0.0
    d_std = float(np.std(delta)) if len(delta) else 0.0
    return np.array([
        np.mean(vals), np.std(vals), np.median(vals), q75-q25,
        d_abs, d_std, voiced_fraction, transition_rate,
    ], dtype=np.float32)


def _ffv_spectrum_for_frame(x, center):
    """Approximate the normative FFV spectrum via spectral dilation matching.

    Laskowski's FFV compares magnitude spectra from the left and right sides of
    an analysis frame. A candidate log-frequency dilation corresponds to an
    instantaneous F0 change; normalized spectral correlation gives the FFV
    spectrum. This avoids explicit F0 tracking and remains defined in unvoiced
    regions.
    """
    half = int(round(FFV_FRAME_MS * SAMPLE_RATE / 2000.0))
    sep = int(round(FFV_SEPARATION_MS * SAMPLE_RATE / 2000.0))
    l0, l1 = center-sep-half, center-sep
    r0, r1 = center+sep, center+sep+half
    if l0 < 0 or r1 > len(x):
        return None
    win = np.hanning(half).astype(np.float64)
    L = np.abs(np.fft.rfft(x[l0:l1] * win, n=FFV_N_FFT))[1:]
    R = np.abs(np.fft.rfft(x[r0:r1] * win, n=FFV_N_FFT))[1:]
    freqs = np.fft.rfftfreq(FFV_N_FFT, 1.0/SAMPLE_RATE)[1:]
    keep = (freqs >= 70.0) & (freqs <= 4000.0)
    f, L, R = freqs[keep], L[keep], R[keep]
    L = np.log1p(L); R = np.log1p(R)
    # ±2 octaves/frame covers the normative FFV extremities. Dense sampling is
    # reduced to seven prosodically interpretable bands below.
    rates = np.linspace(-2.0, 2.0, 81)
    spec = np.empty_like(rates)
    Ln = np.linalg.norm(L) + 1e-10
    for i, rate in enumerate(rates):
        # Dilation maps right spectrum back to left-frequency coordinates.
        scale = 2.0 ** rate
        Rd = np.interp(f * scale, f, R, left=0.0, right=0.0)
        spec[i] = np.dot(L, Rd) / (Ln * (np.linalg.norm(Rd)+1e-10))
    return rates, spec


def extract_ffv_frames(window):
    """Seven-dimensional frame-synchronous FFV representation.

    The seven filters summarize extreme fall, rapid fall, slow fall, flat,
    slow rise, rapid rise, and extreme rise, following the seven-coefficient
    FFV representation described by Laskowski et al.
    """
    x = np.asarray(window, dtype=np.float64).reshape(-1)
    hop = int(round(FFV_HOP_MS * SAMPLE_RATE / 1000.0))
    margin = int(round((FFV_FRAME_MS/2 + FFV_SEPARATION_MS) * SAMPLE_RATE/1000.0))
    centers = range(margin, max(margin+1, len(x)-margin), hop)
    # Seven overlapping triangular/trapezoidal regions over the FFV rate axis.
    centers_r = np.array([-1.5, -0.65, -0.25, 0.0, 0.25, 0.65, 1.5])
    widths = np.array([0.75, 0.55, 0.35, 0.20, 0.35, 0.55, 0.75])
    out=[]
    for c in centers:
        z=_ffv_spectrum_for_frame(x,c)
        if z is None: continue
        rates,spec=z
        vals=[]
        for mu,w in zip(centers_r,widths):
            weight=np.maximum(1.0-np.abs(rates-mu)/w,0.0)
            vals.append(float(np.sum(weight*spec)/(np.sum(weight)+1e-10)))
        out.append(vals)
    if not out:
        return np.zeros((FFV_NUM_FILTERS,1),dtype=np.float32)
    return np.asarray(out,dtype=np.float32).T


def extract_ffv(window):
    """FFV pooled as frame mean + std -> 14 dimensions."""
    ffv=extract_ffv_frames(window)
    return np.concatenate([ffv.mean(axis=1),ffv.std(axis=1)]).astype(np.float32)

def extract_features(window):
    return {
        "logmel": extract_logmel(window),
        "mfcc": extract_mfcc(window),
        "cqcc": extract_cqcc(window),
        "f0": extract_f0(window),
        "ffv": extract_ffv(window),
    }


if __name__ == "__main__":
    import time

    test_window = np.random.randn(SAMPLE_RATE).astype(np.float32)
    t0 = time.perf_counter()
    features = extract_features(test_window)
    elapsed = (time.perf_counter() - t0) * 1000

    for name, value in features.items():
        print(f"{name:<7}: {value.shape}")

    frames = extract_cqcc_frames(test_window)
    print(f"cqcc_frames: {frames.shape}")
    print(f"Time       : {elapsed:.2f}ms per window")
    print("\nfeatures.py OK")
