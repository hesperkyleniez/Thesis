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

from improvedCQCC_config import (
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


# -----------------------------------------------------------------------------
# Final proposed CNN-GRU-CQCC-F0 fusion: paper-inspired CQCC ONLY.
# This supplements, rather than changes, the original all-model feature cache.
# The exact published parameters and window pooling are used by the separate
# final-CQCC extraction scripts (improvedCQCC_prepare_paper_*).
# -----------------------------------------------------------------------------


# Akıncı–Özbay (2026)-inspired CQCC implementation from verified project.
"""Development-only CQCC extraction comparison inspired by Akıncı & Özbay (2026).

This is an independently implemented approximation of the *reported* extraction
settings, NOT a bit-for-bit reproduction.  It never reads held-out/test folders.

Stage self-test:    inspect parameter/dimensionality differences on synthetic audio.
Stage benchmark:    CQCC-only warmed time for synthetic 1-s audio on this machine.
Stage pilot:        speaker-disjoint, condition-matched, development-only linear-SVM
                    probe using frozen existing .npz as the current-CQCC arm,
                    and re-extracted paper-inspired CQCC from the same original
                    WAV and deterministic degradation.

Do not use pilot scores to replace the v28 fusion or report final test results.
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa
from scipy.fft import dct
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PAPER_FMIN = 32.70
PAPER_N_BINS = 84
PAPER_BPO = 12
PAPER_CEPSTRA = 20
PAPER_HOP = 512 # The uploaded paper does not specify its CQCC hop; match thesis CQCC hop.
PAPER_EPS = 1e-10
VERSION = 'akinci_ozbay_2026_dev_cqcc_screen_v1'
CONDITIONS = ('clean', 'telephone', 'mobile_gaussian',
              'mobile_urban', 'room_gaussian', 'room_urban')


def paper_cqcc_frames(audio, sr=16000):
    """Paper-inspired: CQT -> log power -> DCT -> first 20 static coefficients.

    Based on reported fmin/CQT bins/BPO/coefficients. The publication does not
    fully fix all numerical implementation details, so this is NOT a replication.
    """
    x = np.asarray(audio, np.float32).reshape(-1)
    if len(x) < 1:
        raise ValueError('Empty input')
    cqt = librosa.cqt(y=x, sr=sr, hop_length=PAPER_HOP,
                      fmin=PAPER_FMIN, n_bins=PAPER_N_BINS,
                      bins_per_octave=PAPER_BPO, window='hann', scale=True)
    log_power = np.log(np.square(np.abs(cqt)) + PAPER_EPS)
    return dct(log_power, type=2, axis=0, norm='ortho')[:PAPER_CEPSTRA].astype(np.float32)


def pool_by_centers(audio, extractor, win_samples=16000, hop_samples=8000,
                    frame_hop=512):
    """Same 1-s, 0.5-s-overlap clip pooling convention as thesis features.py."""
    x = np.asarray(audio, np.float32).reshape(-1)
    starts = np.arange(0, max(len(x), 1), hop_samples, dtype=np.int64)
    padded = int(starts[-1] + win_samples)
    xp = np.pad(x, (0, max(padded - len(x), 0)))
    frames = extractor(xp)
    centers = np.arange(frames.shape[1], dtype=np.int64) * frame_hop
    if frames.shape[0] != 20:
        raise ValueError(f'Expected 20 coefficients, found {frames.shape[0]}')
    pooled = np.zeros((len(starts), 40), np.float32)
    for k, s in enumerate(starts):
        take = (centers >= s) & (centers < s + win_samples)
        subset = frames[:, take]
        if subset.size:
            pooled[k, :20] = subset.mean(axis=1)
            pooled[k, 20:] = subset.std(axis=1)
    if not np.isfinite(pooled).all():
        raise ValueError('Non-finite CQCC feature vector')
    return pooled


def thesis_modules():
    # Imported only by development-only stages. These imports may initialize
    # the user's established project folders but must not update training data.
    import improvedCQCC_config as config
    import improvedCQCC_features as features
    return config, features


def self_test():
    cfg, features = thesis_modules()
    rng = np.random.default_rng(42)
    t = np.arange(cfg.SAMPLE_RATE, dtype=np.float32) / cfg.SAMPLE_RATE
    synthetic = (0.13*np.sin(2*np.pi*190*t) + 0.04*rng.normal(size=len(t))).astype(np.float32)
    a = pool_by_centers(synthetic, features.extract_cqcc_frames,
                        cfg.WIN_SAMPLES, cfg.HOP_SAMPLES, cfg.CQCC_HOP_LENGTH)
    b = pool_by_centers(synthetic, paper_cqcc_frames,
                        cfg.WIN_SAMPLES, cfg.HOP_SAMPLES, PAPER_HOP)
    assert a.shape == b.shape == (2,40), (a.shape,b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    print('SELF-TEST PASSED: both pipelines produce finite (2, 40) window arrays')
    print(f'  Thesis/current: bins={cfg.CQCC_N_BINS}, BPO={cfg.CQCC_BINS_PER_OCTAVE}, '
          f'fmin={cfg.CQCC_FMIN}, approx max center={cfg.CQCC_FMIN*2**((cfg.CQCC_N_BINS-1)/cfg.CQCC_BINS_PER_OCTAVE):.1f} Hz')
    print(f'  Paper-inspired: bins={PAPER_N_BINS}, BPO={PAPER_BPO}, '
          f'fmin={PAPER_FMIN}, approx max center={PAPER_FMIN*2**((PAPER_N_BINS-1)/PAPER_BPO):.1f} Hz')
    print('  Paper recipe: librosa.cqt + direct frequency-axis DCT (20 coefficients)')
    print('  Current recipe: librosa.vqt + cubic-spline frequency resampling + DCT')
    print(f'  Synthetic feature cosine similarity: '
          f'{np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9):.4f}')
    print('  WARNING: Features differ. DO NOT apply new features to an old CQCC checkpoint.')
    print('  Original test recordings, model checkpoints, and caches untouched.')


def benchmark(runs):
    cfg, features = thesis_modules()
    rng = np.random.default_rng(42)
    y = rng.normal(0, .1, cfg.SAMPLE_RATE).astype(np.float32)
    output = {}
    for name, fn in [('thesis_current', features.extract_cqcc),
                     ('paper_inspired', lambda v: np.r_[paper_cqcc_frames(v).mean(axis=1),
                                                         paper_cqcc_frames(v).std(axis=1)])]:
        # Note: use one call per benchmark; direct function below avoids double CQT.
        if name == 'paper_inspired':
            def fn(v):
                z = paper_cqcc_frames(v)
                return np.r_[z.mean(axis=1), z.std(axis=1)]
        for _ in range(3):
            fn(y)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            res = fn(y)
            times.append((time.perf_counter()-t0)*1000)
            assert res.shape == (40,) and np.isfinite(res).all()
        output[name] = (np.mean(times), np.quantile(times,.95), np.max(times))
    print('Warmed CQCC-only extraction latency (one 1-second waveform):')
    for name,(mean,p95,mx) in output.items():
        print(f'  {name:<16} mean={mean:.2f} ms, p95={p95:.2f} ms, max={mx:.2f} ms')
    print('NOTE: excludes models, fusion, waveform acquisition, I/O and resampling.')
    print('NOTE: Python benchmark is not evidence of accuracy improvement.')


def source_cache_path(cfg, source, label, speaker):
    name = os.path.splitext(os.path.basename(source))[0]
    if label == cfg.LABEL_AI and not name.endswith('_AI'):
        name += '_AI'
    elif label == cfg.LABEL_REAL and not name.endswith('_Real'):
        name += '_Real'
    return os.path.join(cfg.FEATURES_DIR, 'train', speaker, name+'.npz')


def read_dev_candidates(cfg, fold, per_cell, seed):
    from improvedCQCC_dataset import load_all_files
    samples = load_all_files(cfg.TRAIN_DIR)
    cond_file = os.path.join(cfg.FEATURES_DIR, 'conditions.npy')
    if not os.path.isfile(cond_file):
        raise FileNotFoundError(f'Frozen development condition map missing: {cond_file}')
    assigned = np.load(cond_file, allow_pickle=True).item()
    buckets = defaultdict(list)
    for source, label, speaker in samples:
        if speaker not in cfg.FOLD_ASSIGNMENTS:
            raise RuntimeError(f'Unassigned development speaker: {speaker}')
        if speaker in cfg.TEST_SPEAKERS:
            raise RuntimeError(f'Test speaker in development: {speaker}')
        if source not in assigned:
            raise RuntimeError(f'Frozen condition assignment missing for: {source}')
        cond = str(assigned[source])
        if cond not in CONDITIONS:
            raise RuntimeError(f'Unexpected development condition: {cond}')
        cache = source_cache_path(cfg,source,label,speaker)
        if not os.path.isfile(cache):
            raise FileNotFoundError(f'Missing frozen original dev cache: {cache}')
        part = 'val' if cfg.FOLD_ASSIGNMENTS[speaker] == fold else 'train'
        buckets[(part,cond,int(label))].append((source,cache,cond,int(label),speaker))
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(buckets):
        items = buckets[key]
        rng.shuffle(items)
        chosen = items[:per_cell]
        print(f'  {key}: sampled {len(chosen):>3} of {len(items):>5} clips')
        (val if key[0]=='val' else train).extend(chosen)
    if len(train)<20 or len(val)<10:
        raise RuntimeError('Too few development clips for the pilot; increase --per-cell')
    assert set(x[4] for x in train).isdisjoint(set(x[4] for x in val))
    return train,val


def compute_metrics(labels, score):
    score=np.asarray(score,float)
    labels=np.asarray(labels,int)
    preds=(score>=0).astype(int)
    fpr,tpr,thresh=roc_curve(labels,score)
    fnr=1-tpr
    i=int(np.argmin(np.abs(fpr-fnr)))
    tn,fp,fn,tp=confusion_matrix(labels,preds,labels=[0,1]).ravel()
    return {'N':len(labels),'acc':float(accuracy_score(labels,preds)),
            'f1':float(f1_score(labels,preds,zero_division=0)),
            'precision':float(precision_score(labels,preds,zero_division=0)),
            'recall':float(recall_score(labels,preds,zero_division=0)),
            'auc':float(roc_auc_score(labels,score)),
            'eer':float((fpr[i]+fnr[i])/2),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def get_degraded_audio(source, condition, resources, cfg):
    import improvedCQCC_augmentation as aug
    from improvedCQCC_original_prepare_dataset import pair_rng
    audio,_=librosa.load(source,sr=cfg.SAMPLE_RATE,mono=True)
    audio=np.asarray(audio,np.float32)
    audio=(audio-audio.mean())/(audio.std()+1e-8)
    if condition=='clean':
        audio=aug.peak_normalize(audio)
    else:
        room,mobile,urban=resources
        audio=aug.apply_augmentation(audio,condition,room,mobile,urban,
                                     rng=pair_rng(source),training=True)
    return audio


def pilot(fold,per_cell,save):
    cfg,features = thesis_modules()
    print('DEVELOPMENT ONLY — test and test_degraded NOT ACCESSED')
    print('A compact linear-SVM screening experiment, NOT a retrained v28 fusion')
    train,val=read_dev_candidates(cfg,fold,per_cell,cfg.SEED)
    print('  Loading augmentation files for exact condition replay ...')
    import improvedCQCC_augmentation as augmentation
    resources=augmentation.collect_rir_files()
    sampled=train+val
    xs_old=[];xs_new=[];ys=[];parts=[];conds=[];fidelities=[]
    fidelity_by_condition=defaultdict(int)
    for i,(source,cache,cond,label,speaker) in enumerate(sampled,1):
        with np.load(cache,allow_pickle=False) as d:
            old=np.asarray(d['cqcc'],np.float32)
            if old.ndim!=2 or old.shape[1]!=40:
                raise RuntimeError(f'Unexpected frozen current CQCC shape {old.shape}: {cache}')
            if 'condition' not in d or str(d['condition'].item())!=cond:
                raise RuntimeError(f'Condition mismatch between frozen cache and assignment: {cache}')
            if int(d['label'])!=label:
                raise RuntimeError(f'Label mismatch: {cache}')
        wav=get_degraded_audio(source,cond,resources,cfg)
        newer=pool_by_centers(wav,paper_cqcc_frames,
                              cfg.WIN_SAMPLES,cfg.HOP_SAMPLES,PAPER_HOP)
        if len(old)!=len(newer):
            raise RuntimeError(f'Window count mismatch: cached={len(old)} new={len(newer)}: {cache}')
        # Check a few source reconstructions against the ORIGINAL frontend/cache.
        if fidelity_by_condition[cond]<2:
            fidelity_by_condition[cond]+=1
            current=pool_by_centers(wav,features.extract_cqcc_frames,
                                   cfg.WIN_SAMPLES,cfg.HOP_SAMPLES,cfg.CQCC_HOP_LENGTH)
            reldiff=float(np.linalg.norm(current-old)/(np.linalg.norm(old)+1e-8))
            fidelities.append((cache,cond,reldiff))
        xs_old.append(old.mean(axis=0)); xs_new.append(newer.mean(axis=0))
        ys.append(label);parts.append('train' if i<=len(train) else 'val');conds.append(cond)
        if i%25==0 or i==len(sampled):
            print(f'  reconstructed {i}/{len(sampled)} dev clips')
    drift=np.array([v for _,_,v in fidelities]); print(f'  Current-cache replay relative error ({len(drift)} across conditions): '
       f'median={np.median(drift):.5f}, max={np.max(drift):.5f}')
    if float(np.median(drift))>.02 or float(np.max(drift))>.1:
        raise RuntimeError('Existing current cache could not be faithfully reproduced '
                           'from the present source code/augmentations. Stop: no fair paired comparison.')
    xs={'current':np.stack(xs_old), 'paper_inspired':np.stack(xs_new)}
    y=np.asarray(ys); tr=np.asarray(parts)=='train'; va=~tr
    results={}
    for name,x in xs.items():
        pipe=make_pipeline(StandardScaler(),SVC(kernel='linear',C=1.0,probability=False,random_state=cfg.SEED))
        pipe.fit(x[tr],y[tr])
        scores=pipe.decision_function(x[va])
        rows={'overall':compute_metrics(y[va],scores),'conditions':{}}
        for condition in CONDITIONS:
            mask=va & (np.asarray(conds)==condition)
            if mask.sum() and len(np.unique(y[mask]))==2:
                rows['conditions'][condition]=compute_metrics(y[mask],pipe.decision_function(x[mask]))
        results[name]=rows
        stats=rows['overall']
        print(f'\n{name}: fold={fold} val N={stats["N"]} AUC={stats["auc"]:.4f} EER={stats["eer"]:.4f} '
              f'F1={stats["f1"]:.4f} Acc={stats["acc"]:.4f}')
        for k,v in rows['conditions'].items():
            print(f'    {k:<16} N={v["N"]:3d} AUC={v["auc"]:.4f} F1={v["f1"]:.4f}')
    print('\nDelta paper-minus-current on validation (same linear SVM protocol): '
          f'AUC={results["paper_inspired"]["overall"]["auc"]-results["current"]["overall"]["auc"]:+.4f}, '
          f'F1={results["paper_inspired"]["overall"]["f1"]-results["current"]["overall"]["f1"]:+.4f}')
    print('EXPLORATORY ONLY: small subset, one speaker fold, linear SVM rather than final CQCC MLP.')
    print('DO NOT alter any model or held-out report based on this pilot.')
    if save:
        path=Path(cfg.RESULTS_DIR)/'akinci_ozbay_dev_pilot'/f'pilot_fold{fold}_percell{per_cell}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        report={'version':VERSION,'fold':fold,'per_cell':per_cell,'n_train':int(tr.sum()),'n_val':int(va.sum()),
                'reference':'Akıncı & Özbay (2026), Applied Sciences 16(13), 6707',
                'not_exact_paper_replication':True,'clip_aggregation':'mean of 1s pooled CQCC windows',
                'classifier':'linear SVM C=1 on development speaker-disjoint fold',
                'current_cache_replay_relative_errors':[{'cache':p,'condition':c,'error':v} for p,c,v in fidelities],
                'results':results}
        with open(path,'x',encoding='utf-8') as f: json.dump(report,f,indent=2)
        print('Saved NEW report (no original files overwritten):',path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['self-test','benchmark','pilot'],default='self-test')
    p.add_argument('--runs',type=int,default=30)
    p.add_argument('--fold',type=int,choices=[1,2,3,4],default=1)
    p.add_argument('--per-cell',type=int,default=12,
                   help='Max development recordings per train/val, condition and label (12=>up to 288).')
    p.add_argument('--save',action='store_true',help='Write a NEW JSON pilot report using exclusive-create mode.')
    a=p.parse_args()
    if a.stage=='self-test':self_test()
    elif a.stage=='benchmark':benchmark(a.runs)
    else:pilot(a.fold,a.per_cell,a.save)

if __name__=='__main__':main()


# Names used by the paper sidecars and live-window benchmark.
extract_improved_cqcc_frames = paper_cqcc_frames

def extract_improved_cqcc(window):
    x=np.asarray(window,dtype=np.float32)
    m=paper_cqcc_frames(x)
    return np.concatenate((m.mean(axis=1),m.std(axis=1))).astype(np.float32)
