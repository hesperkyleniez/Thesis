"""Fast acoustic feature extraction for the thesis pipeline.

Design goals
------------
* Keep the defining Todisco CQCC stages (CQT, linearized frequency axis,
  DCT, 20 static coefficients including C0) with a runtime-oriented B=12 grid.
* Use the normative 7-filter FFV construction described by Laskowski et al.
  rather than the previous nested Python approximation.
* Use YIN for the CQCC+F0 comparison because it is dramatically cheaper than
  pYIN while still providing a frame-level F0 contour.
* Extract expensive auxiliary frame streams ONCE per clip, then pool them into
  the existing 1.0 s / 0.5 s-overlap windows.  This is the main runtime win.
"""

import numpy as np
import librosa
from scipy.ndimage import zoom
from scipy.fft import dct
from scipy.interpolate import CubicSpline

from config import (
    SAMPLE_RATE, N_MELS, N_MFCC, N_FFT, HOP_LENGTH, TARGET_SHAPE,
    WIN_SAMPLES, HOP_SAMPLES, F0_MIN, F0_MAX,
    CQCC_BINS_PER_OCTAVE, CQCC_FMAX, CQCC_FMIN, CQCC_N_BINS,
    CQCC_UNIFORM_SAMPLES_FIRST_OCTAVE, CQCC_NUM_STATIC, CQCC_HOP_LENGTH,
    F0_FRAME_LENGTH, F0_HOP_LENGTH, F0_LOG, F0_VECTOR_DIM,
    FFV_N_FFT, FFV_NUM_FILTERS,
)

_EPS32 = np.finfo(np.float32).eps


# -----------------------------------------------------------------------------
# Log-Mel / MFCC (cheap enough to keep exact per-window behavior)
# -----------------------------------------------------------------------------
def extract_logmel(window):
    """128x128 normalized Log-Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(window, dtype=np.float32), sr=SAMPLE_RATE,
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
    mn, mx = float(log_mel.min()), float(log_mel.max())
    return ((log_mel - mn) / (mx - mn + 1e-8)).astype(np.float32)


def extract_mfcc(window):
    """40 MFCC coefficients pooled as mean + std -> 80 dimensions."""
    mfcc = librosa.feature.mfcc(
        y=np.asarray(window, dtype=np.float32), sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


# -----------------------------------------------------------------------------
# CQCC (research-backed runtime configuration)
# -----------------------------------------------------------------------------
def _cqcc_frequency_axes():
    geometric = CQCC_FMIN * (
        2.0 ** (np.arange(CQCC_N_BINS, dtype=np.float64) / CQCC_BINS_PER_OCTAVE)
    )
    # Todisco default d=16 => uniform frequency step fmin/d.
    step_hz = CQCC_FMIN / CQCC_UNIFORM_SAMPLES_FIRST_OCTAVE
    uniform = np.arange(
        CQCC_FMIN, geometric[-1] + 0.5 * step_hz, step_hz, dtype=np.float64
    )
    return geometric, uniform


_CQCC_GEOMETRIC_FREQS, _CQCC_UNIFORM_FREQS = _cqcc_frequency_axes()


def _build_cqcc_projection():
    """Precompute spline-resampling + first-20-DCT as one linear projection.

    Cubic-spline interpolation and the DCT are both linear in the log-power
    values.  Collapsing them into a compact 20xN_CQT matrix is numerically equivalent to
    constructing the uniformly-resampled spectrum on every call, but
    avoids that allocation and DCT at runtime.
    """
    eye = np.eye(CQCC_N_BINS, dtype=np.float32)
    interp = CubicSpline(
        _CQCC_GEOMETRIC_FREQS, eye, axis=0,
        bc_type="not-a-knot", extrapolate=False,
    )(_CQCC_UNIFORM_FREQS).astype(np.float32)
    return dct(interp, type=2, axis=0, norm="ortho")[:CQCC_NUM_STATIC].astype(np.float32)


_CQCC_PROJECTION = _build_cqcc_projection()
_CQCC_B = float(CQCC_BINS_PER_OCTAVE)
_CQCC_GAMMA = 228.7 * (2.0 ** (1.0 / _CQCC_B) - 2.0 ** (-1.0 / _CQCC_B))


def extract_cqcc_frames(audio):
    """Static CQCC frame stream, shape (20, frames)."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("CQCC input is empty.")

    cqt = librosa.vqt(
        y=x, sr=SAMPLE_RATE, hop_length=CQCC_HOP_LENGTH,
        fmin=CQCC_FMIN, n_bins=CQCC_N_BINS,
        bins_per_octave=CQCC_BINS_PER_OCTAVE,
        gamma=_CQCC_GAMMA, window="hann", scale=True,
    )
    power = np.abs(cqt).astype(np.float32)
    log_power = np.log(power * power + _EPS32)
    return (_CQCC_PROJECTION @ log_power).astype(np.float32)


def extract_cqcc(window):
    cqcc = extract_cqcc_frames(window)
    return np.concatenate([cqcc.mean(axis=1), cqcc.std(axis=1)]).astype(np.float32)


# -----------------------------------------------------------------------------
# Fast F0/prosody comparison front-end (YIN)
# -----------------------------------------------------------------------------
def extract_f0_frames(audio):
    """Return YIN F0, an energy-derived voiced mask, and frame centers.

    YIN avoids pYIN's Viterbi decoding cost.  An RMS gate prevents low-energy
    padded/silent frames from being treated as reliable pitch observations.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    f0 = librosa.yin(
        x, fmin=F0_MIN, fmax=F0_MAX, sr=SAMPLE_RATE,
        frame_length=F0_FRAME_LENGTH, hop_length=F0_HOP_LENGTH,
        center=True,
    ).astype(np.float32)
    rms = librosa.feature.rms(
        y=x, frame_length=F0_FRAME_LENGTH,
        hop_length=F0_HOP_LENGTH, center=True,
    )[0].astype(np.float32)
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    # Relative energy gate: robust to per-file gain because preprocessing is
    # normalized.  -35 dB keeps ordinary voiced speech while rejecting padding.
    peak_rms = float(np.max(rms))
    if peak_rms <= 1e-7:
        voiced = np.zeros_like(f0, dtype=bool)
    else:
        rms_db = 20.0 * np.log10((rms + 1e-8) / (peak_rms + 1e-8))
        voiced = (
            np.isfinite(f0) & (f0 >= F0_MIN) & (f0 <= F0_MAX) & (rms_db >= -35.0)
        )
    centers = np.arange(n, dtype=np.int64) * F0_HOP_LENGTH
    return f0, voiced, centers


def _summarize_f0(f0, voiced):
    voiced = np.asarray(voiced, dtype=bool)
    if len(f0) == 0:
        return np.zeros(F0_VECTOR_DIM, dtype=np.float32)
    voiced_fraction = float(np.mean(voiced))
    transition_rate = float(np.mean(voiced[1:] != voiced[:-1])) if len(voiced) > 1 else 0.0
    vals = np.asarray(f0, dtype=np.float32)[voiced]
    if vals.size == 0:
        return np.zeros(F0_VECTOR_DIM, dtype=np.float32)
    if F0_LOG:
        vals = np.log(vals + 1e-8)
    q25, q75 = np.percentile(vals, [25, 75])
    delta = np.diff(vals)
    return np.asarray([
        np.mean(vals), np.std(vals), np.median(vals), q75 - q25,
        np.mean(np.abs(delta)) if delta.size else 0.0,
        np.std(delta) if delta.size else 0.0,
        voiced_fraction, transition_rate,
    ], dtype=np.float32)


def extract_f0(window):
    f0, voiced, _ = extract_f0_frames(window)
    return _summarize_f0(f0, voiced)


# -----------------------------------------------------------------------------
# Normative 7-filter FFV, vectorized
# -----------------------------------------------------------------------------
# Laskowski/Edlund defaults at 16 kHz:
# tFra=8 ms, tSep=14 ms, tInt=11 ms, tExt=9 ms, Ng=512, tSepRef=8 ms.
_FFV_TFRA = 0.008
_FFV_TSEP = 0.014
_FFV_TINT = 0.011
_FFV_TEXT = 0.009
_FFV_TSEP_REF = 0.008
_FFV_NG = 512
_FFV_HOP = int(round(_FFV_TFRA * SAMPLE_RATE))
_FFV_FRAME = int(round((2 * _FFV_TEXT + _FFV_TSEP) * SAMPLE_RATE)) + 1  # inclusive 32 ms support
_FFV_CENTER = _FFV_FRAME // 2
_FFV_SEP_HALF = int(round((_FFV_TSEP / 2.0) * SAMPLE_RATE))
_FFV_INT = int(round(_FFV_TINT * SAMPLE_RATE))
_FFV_EXT = int(round(_FFV_TEXT * SAMPLE_RATE))


def _half_window(length, kind):
    # Generate a monotonic half-window ending at 1.0.
    if kind == "hann":
        full = np.hanning(max(2 * length - 1, 3))
    else:
        full = np.hamming(max(2 * length - 1, 3))
    return full[:length].astype(np.float32)


def _build_ffv_windows():
    hL = np.zeros(_FFV_FRAME, dtype=np.float32)
    hR = np.zeros(_FFV_FRAME, dtype=np.float32)
    l_peak = _FFV_CENTER - _FFV_SEP_HALF
    r_peak = _FFV_CENTER + _FFV_SEP_HALF

    # Away from center: Hamming. Toward center: Hann (normative defaults).
    ext_rise = _half_window(_FFV_EXT + 1, "hamming")
    int_fall = _half_window(_FFV_INT + 1, "hann")[::-1]
    hL[l_peak - _FFV_EXT:l_peak + 1] = ext_rise
    hL[l_peak:l_peak + _FFV_INT + 1] = np.maximum(
        hL[l_peak:l_peak + _FFV_INT + 1], int_fall
    )

    int_rise = _half_window(_FFV_INT + 1, "hann")
    ext_fall = _half_window(_FFV_EXT + 1, "hamming")[::-1]
    hR[r_peak - _FFV_INT:r_peak + 1] = int_rise
    hR[r_peak:r_peak + _FFV_EXT + 1] = np.maximum(
        hR[r_peak:r_peak + _FFV_EXT + 1], ext_fall
    )
    return hL, hR


_FFV_HL, _FFV_HR = _build_ffv_windows()
_FFV_FREQS = np.fft.rfftfreq(FFV_N_FFT, 1.0 / SAMPLE_RATE).astype(np.float32)
# Avoid DC and the very top of the spectrum, where dilation has little common
# support.  The same support is used for every rho hypothesis.
_FFV_FREQ_KEEP = (_FFV_FREQS >= 50.0) & (_FFV_FREQS <= 0.45 * SAMPLE_RATE)
_FFV_F = _FFV_FREQS[_FFV_FREQ_KEEP]


def _build_ffv_filterbank():
    fb = np.zeros((7, _FFV_NG + 1), dtype=np.float32)
    fb[0, 117:140] = 1.0
    fb[1, 246:251] = 1.0; fb[1, 245] = fb[1, 251] = 0.5
    fb[2, 250:255] = 1.0; fb[2, 249] = fb[2, 255] = 0.5
    fb[3, 255:258] = 1.0; fb[3, 254] = fb[3, 258] = 0.5
    fb[4, 258:263] = 1.0; fb[4, 257] = fb[4, 263] = 0.5
    fb[5, 262:267] = 1.0; fb[5, 261] = fb[5, 267] = 0.5
    fb[6, 373:396] = 1.0
    return fb


_FFV_FB_FULL = _build_ffv_filterbank()
_FFV_ACTIVE_IDX = np.flatnonzero(np.any(_FFV_FB_FULL > 0, axis=0)).astype(np.int32)
_FFV_FB = _FFV_FB_FULL[:, _FFV_ACTIVE_IDX]
_FFV_FB /= np.maximum(_FFV_FB.sum(axis=1, keepdims=True), 1e-8)
_FFV_DELTA_RHO = (4.0 / _FFV_NG) * (_FFV_TSEP_REF / _FFV_TSEP)
_FFV_RHO = ((_FFV_ACTIVE_IDX - _FFV_NG // 2) * _FFV_DELTA_RHO).astype(np.float32)


def _interp_plan(scale):
    target = _FFV_F[None, :] * scale[:, None]
    hi = np.searchsorted(_FFV_F, target, side="left")
    valid = (hi > 0) & (hi < len(_FFV_F))
    hi = np.clip(hi, 1, len(_FFV_F) - 1)
    lo = hi - 1
    x0, x1 = _FFV_F[lo], _FFV_F[hi]
    w = np.where(valid, (target - x0) / (x1 - x0 + 1e-12), 0.0)
    return lo.astype(np.int32), hi.astype(np.int32), w.astype(np.float32), valid


# rho<0: dilate left by 2^rho. rho>0: dilate right by 2^-rho.
_FFV_NEG = _FFV_RHO < 0
_FFV_POS = _FFV_RHO > 0
_FFV_ZERO = ~(_FFV_NEG | _FFV_POS)
_FFV_NEG_PLAN = _interp_plan((2.0 ** _FFV_RHO[_FFV_NEG]).astype(np.float32))
_FFV_POS_PLAN = _interp_plan((2.0 ** (-_FFV_RHO[_FFV_POS])).astype(np.float32))


def _cosine_with_dilated(A, B, plan):
    """Cosine similarity of A with frequency-dilated B for many rho values."""
    lo, hi, w, valid = plan
    Bd = (B[:, lo] * (1.0 - w)[None, :, :] + B[:, hi] * w[None, :, :])
    Bd *= valid[None, :, :]
    # Compare only common support after dilation.
    Av = A[:, None, :] * valid[None, :, :]
    num = np.einsum("brf,brf->br", Av, Bd, optimize=True)
    den = np.linalg.norm(Av, axis=2) * np.linalg.norm(Bd, axis=2) + 1e-10
    return (num / den).astype(np.float32)


def extract_ffv_frames(audio, return_centers=False):
    """Normative-style seven-dimensional FFV frame stream.

    The expensive rho comparisons are vectorized across frames and only rho
    samples touched by the published seven-filter default bank are evaluated.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(x) < _FFV_FRAME:
        x = np.pad(x, (0, _FFV_FRAME - len(x)))

    starts = np.arange(0, len(x) - _FFV_FRAME + 1, _FFV_HOP, dtype=np.int64)
    if starts.size == 0:
        out = np.zeros((FFV_NUM_FILTERS, 1), dtype=np.float32)
        return (out, np.array([len(x) // 2], dtype=np.int64)) if return_centers else out

    idx = starts[:, None] + np.arange(_FFV_FRAME, dtype=np.int64)[None, :]
    frames = x[idx]
    L = np.abs(np.fft.rfft(frames * _FFV_HL[None, :], n=FFV_N_FFT, axis=1)).astype(np.float32)
    R = np.abs(np.fft.rfft(frames * _FFV_HR[None, :], n=FFV_N_FFT, axis=1)).astype(np.float32)
    L = np.log1p(L[:, _FFV_FREQ_KEEP])
    R = np.log1p(R[:, _FFV_FREQ_KEEP])

    spec = np.empty((len(starts), len(_FFV_RHO)), dtype=np.float32)
    if np.any(_FFV_NEG):
        # rho<0 dilates LEFT; compare R against dilated L.
        spec[:, _FFV_NEG] = _cosine_with_dilated(R, L, _FFV_NEG_PLAN)
    if np.any(_FFV_POS):
        # rho>0 dilates RIGHT; compare L against dilated R.
        spec[:, _FFV_POS] = _cosine_with_dilated(L, R, _FFV_POS_PLAN)
    if np.any(_FFV_ZERO):
        num = np.sum(L * R, axis=1)
        den = np.linalg.norm(L, axis=1) * np.linalg.norm(R, axis=1) + 1e-10
        spec[:, _FFV_ZERO] = (num / den)[:, None]

    filtered = (spec @ _FFV_FB.T).T.astype(np.float32)
    centers = starts + _FFV_CENTER
    return (filtered, centers) if return_centers else filtered


def extract_ffv(window):
    ffv = extract_ffv_frames(window)
    return np.concatenate([ffv.mean(axis=1), ffv.std(axis=1)]).astype(np.float32)


# -----------------------------------------------------------------------------
# Clip-level extraction: expensive streams once, then pool into thesis windows
# -----------------------------------------------------------------------------
def _window_starts(n_samples):
    if n_samples <= 0:
        return np.array([0], dtype=np.int64)
    return np.arange(0, n_samples, HOP_SAMPLES, dtype=np.int64)


def _pool_mean_std(frames, centers, starts, dim):
    out = np.zeros((len(starts), dim * 2), dtype=np.float32)
    for j, start in enumerate(starts):
        mask = (centers >= start) & (centers < start + WIN_SAMPLES)
        z = frames[:, mask]
        if z.shape[1] == 0:
            continue
        out[j, :dim] = z.mean(axis=1)
        out[j, dim:] = z.std(axis=1)
    return out


def extract_clip_features(audio):
    """Extract all cached features while avoiding overlap recomputation.

    Returns arrays with exactly one row per existing 1 s / 0.5 s window.
    Log-Mel/MFCC retain their original per-window implementation. CQCC, F0 and
    FFV are computed once over a zero-padded clip and pooled by frame center.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    starts = _window_starts(len(x))
    padded_len = int(starts[-1] + WIN_SAMPLES)
    xp = np.pad(x, (0, max(0, padded_len - len(x))))

    windows = np.stack([xp[s:s + WIN_SAMPLES] for s in starts]).astype(np.float32)
    logmel = np.stack([extract_logmel(w) for w in windows])
    mfcc = np.stack([extract_mfcc(w) for w in windows])

    cqcc_frames = extract_cqcc_frames(xp)
    cqcc_centers = np.arange(cqcc_frames.shape[1], dtype=np.int64) * CQCC_HOP_LENGTH
    cqcc = _pool_mean_std(cqcc_frames, cqcc_centers, starts, CQCC_NUM_STATIC)

    f0_frames, voiced, f0_centers = extract_f0_frames(xp)
    f0 = np.zeros((len(starts), F0_VECTOR_DIM), dtype=np.float32)
    for j, start in enumerate(starts):
        mask = (f0_centers >= start) & (f0_centers < start + WIN_SAMPLES)
        f0[j] = _summarize_f0(f0_frames[mask], voiced[mask])

    ffv_frames, ffv_centers = extract_ffv_frames(xp, return_centers=True)
    ffv = _pool_mean_std(ffv_frames, ffv_centers, starts, FFV_NUM_FILTERS)

    return {
        "logmel": logmel,
        "mfcc": mfcc,
        "cqcc": cqcc,
        "f0": f0,
        "ffv": ffv,
    }


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
    rng = np.random.default_rng(42)
    x = rng.normal(size=SAMPLE_RATE * 10).astype(np.float32)
    t0 = time.perf_counter()
    feat = extract_clip_features(x)
    elapsed = time.perf_counter() - t0
    print(f"10 s clip -> {len(feat['cqcc'])} overlapping windows in {elapsed:.3f} s")
    for name, value in feat.items():
        print(f"{name:<7}: {value.shape}")
