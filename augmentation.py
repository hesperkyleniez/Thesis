"""Centralized audio degradation pipeline.

All training and degraded-test preprocessing imports this module so the same
implementation is used everywhere.  The clean path is intentionally untouched.
"""

import os
import random
import shutil
import subprocess
import tempfile
import zipfile
import urllib.request

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, butter, filtfilt

from config import (
    SAMPLE_RATE,
    OPENSLR_DIR,
    AIR_DIR,
    URBANSOUND_DIR,
    RESOURCES_DIR,
    URBAN_SNR_DB,
    GAUSSIAN_NOISE_STD,
    RIR_WET_RATIO,
    RIR_MAX_SEC,
    TELEPHONE_CODEC_MODES,
)


def setup_resources():
    """Download/extract augmentation resources if they are missing."""
    os.makedirs(RESOURCES_DIR, exist_ok=True)

    openslr_zip = os.path.join(RESOURCES_DIR, "rirs_noises.zip")
    if not os.path.exists(OPENSLR_DIR) or not os.listdir(OPENSLR_DIR):
        print("Downloading OpenSLR RIRs...")
        if not os.path.exists(openslr_zip):
            urllib.request.urlretrieve(
                "https://www.openslr.org/resources/28/rirs_noises.zip",
                openslr_zip,
            )
        os.makedirs(OPENSLR_DIR, exist_ok=True)
        with zipfile.ZipFile(openslr_zip, "r") as zf:
            zf.extractall(OPENSLR_DIR)

    return collect_rir_files()


def _collect_wavs(folder):
    paths = []
    if not os.path.isdir(folder):
        return paths
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".wav"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def collect_rir_files():
    """Return room RIRs, mobile/device RIRs, and UrbanSound8K WAV files."""
    room_rirs = []
    mobile_rirs = []

    for path in _collect_wavs(OPENSLR_DIR):
        lower = path.lower()
        if "real_rirs" in lower or "real_rir" in lower:
            room_rirs.append(path)
        elif "simulated" in lower or "pointsource" in lower:
            mobile_rirs.append(path)
        elif "rir" in lower:
            room_rirs.append(path)

    # The existing project treats AIR responses as mobile/device responses.
    mobile_rirs.extend(_collect_wavs(AIR_DIR))
    urban_files = _collect_wavs(URBANSOUND_DIR)

    return sorted(set(room_rirs)), sorted(set(mobile_rirs)), urban_files


def peak_normalize(audio):
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 0.0:
        return audio
    return (audio / (peak + 1e-8)).astype(np.float32)


def _choice(seq, rng):
    if not seq:
        return None
    return seq[rng.randrange(len(seq))]


def apply_gaussian_noise(audio, rng):
    """Gaussian white noise using the thesis scaling factor (0.003)."""
    np_rng = np.random.default_rng(rng.randrange(0, 2**32))
    noise = np_rng.normal(0.0, GAUSSIAN_NOISE_STD, len(audio)).astype(np.float32)
    return (np.asarray(audio, dtype=np.float32) + noise).astype(np.float32)


def apply_urban_noise(audio, urban_files, rng, snr_db=URBAN_SNR_DB):
    """Mix a real acoustic-scene recording at the thesis target SNR."""
    noise_path = _choice(urban_files, rng)
    if noise_path is None:
        return np.asarray(audio, dtype=np.float32)

    noise, _ = librosa.load(noise_path, sr=SAMPLE_RATE, mono=True)
    noise = np.asarray(noise, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)

    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / max(len(noise), 1))))
    noise = noise[: len(audio)]

    p_signal = float(np.mean(audio**2)) + 1e-8
    p_noise = float(np.mean(noise**2)) + 1e-8
    scale = np.sqrt(p_signal / (p_noise * (10.0 ** (snr_db / 10.0))))
    return peak_normalize(audio + noise * scale)


def apply_rir(audio, rir_files, rng, wet_ratio=RIR_WET_RATIO):
    """Apply room/device impulse-response convolution."""
    rir_path = _choice(rir_files, rng)
    if rir_path is None:
        return np.asarray(audio, dtype=np.float32)

    rir, _ = librosa.load(rir_path, sr=SAMPLE_RATE, mono=True)
    rir = np.asarray(rir, dtype=np.float32)
    rir = rir[: int(RIR_MAX_SEC * SAMPLE_RATE)]
    if len(rir) == 0:
        return np.asarray(audio, dtype=np.float32)

    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    audio = np.asarray(audio, dtype=np.float32)
    convolved = fftconvolve(audio, rir, mode="full")[: len(audio)]
    return peak_normalize((1.0 - wet_ratio) * audio + wet_ratio * convolved)


def _dynamic_range_compression(audio):
    """Lightweight soft-knee DRC used by the existing methodology."""
    threshold = 0.90
    ratio = 1.20
    out = np.asarray(audio, dtype=np.float32).copy()
    mask = np.abs(out) > threshold
    out[mask] = np.sign(out[mask]) * (
        threshold + (np.abs(out[mask]) - threshold) / ratio
    )
    return out


def _telephone_bandpass(audio):
    """Conventional narrow-band telephone passband (300-3400 Hz)."""
    nyquist = SAMPLE_RATE / 2.0
    b, a = butter(4, [300.0 / nyquist, 3400.0 / nyquist], btype="band")
    return filtfilt(b, a, np.asarray(audio, dtype=np.float32)).astype(np.float32)


def _codec_roundtrip_ffmpeg(audio, mode):
    """Encode/decode through a low-bitrate telecommunication codec with FFmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    audio = np.asarray(audio, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="thesis_codec_") as tmp:
        src = os.path.join(tmp, "input.wav")
        sf.write(src, audio, SAMPLE_RATE, subtype="PCM_16")

        if mode == "gsm":
            encoded = os.path.join(tmp, "channel_gsm.wav")
            enc = [
                ffmpeg, "-y", "-loglevel", "error", "-i", src,
                "-ac", "1", "-ar", "8000", "-c:a", "libgsm_ms", encoded,
            ]
        elif mode == "opus12":
            encoded = os.path.join(tmp, "channel_12k.ogg")
            enc = [
                ffmpeg, "-y", "-loglevel", "error", "-i", src,
                "-ac", "1", "-ar", "16000", "-c:a", "libopus",
                "-application", "voip", "-b:a", "12k", "-vbr", "off", encoded,
            ]
        else:  # opus16
            encoded = os.path.join(tmp, "channel_16k.ogg")
            enc = [
                ffmpeg, "-y", "-loglevel", "error", "-i", src,
                "-ac", "1", "-ar", "16000", "-c:a", "libopus",
                "-application", "voip", "-b:a", "16k", "-vbr", "off", encoded,
            ]

        try:
            subprocess.run(enc, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            decoded = os.path.join(tmp, "decoded.wav")
            dec = [
                ffmpeg, "-y", "-loglevel", "error", "-i", encoded,
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", decoded,
            ]
            subprocess.run(dec, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            y, _ = librosa.load(decoded, sr=SAMPLE_RATE, mono=True)
            return np.asarray(y, dtype=np.float32)
        except (subprocess.CalledProcessError, OSError):
            return None


def _mulaw_fallback(audio):
    """Deterministic G.711-style companding fallback when FFmpeg is unavailable."""
    audio = peak_normalize(audio)
    mu = 255.0
    encoded = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    encoded = np.round((encoded + 1.0) * 127.5) / 127.5 - 1.0
    decoded = np.sign(encoded) * (np.expm1(np.abs(encoded) * np.log1p(mu)) / mu)
    return np.asarray(decoded, dtype=np.float32)


def apply_telephone(audio, rng):
    """Telephone simulation: DRC -> 300-3400 Hz -> low-bitrate codec round-trip.

    Codec variability is intentionally confined to the already-assigned telephone
    condition.  It does not alter the clean/degraded assignment proportions.
    """
    x = _dynamic_range_compression(audio)
    x = peak_normalize(_telephone_bandpass(x))

    mode = _choice(list(TELEPHONE_CODEC_MODES), rng) or "gsm"
    coded = _codec_roundtrip_ffmpeg(x, mode)
    if coded is None:
        # Still preserve narrow-band/quantization effects if FFmpeg is absent.
        narrow = librosa.resample(x, orig_sr=SAMPLE_RATE, target_sr=8000)
        narrow = _mulaw_fallback(narrow)
        coded = librosa.resample(narrow, orig_sr=8000, target_sr=SAMPLE_RATE)

    # Match the original sample length after codec delay/resampling.
    if len(coded) < len(audio):
        coded = np.pad(coded, (0, len(audio) - len(coded)))
    coded = coded[: len(audio)]
    return peak_normalize(coded)


def apply_volume_scaling(audio, rng):
    """Thesis volume-scaling range a_vol in [0.8, 1.2]."""
    scale = rng.uniform(0.8, 1.2)
    return (np.asarray(audio, dtype=np.float32) * scale).astype(np.float32)


def apply_augmentation(
    audio,
    condition,
    room_rirs,
    mobile_rirs,
    urban_files,
    rng=None,
):
    """Apply one of the project's fixed degradation conditions.

    The assignment split is unchanged.  Only the implementation inside each
    existing condition is centralized here.  Clean audio is returned untouched.
    """
    rng = rng or random
    audio = np.asarray(audio, dtype=np.float32)

    if condition == "clean":
        return audio

    if condition == "telephone":
        out = apply_telephone(audio, rng)
    elif condition == "room_gaussian":
        out = apply_rir(apply_gaussian_noise(audio, rng), room_rirs, rng)
    elif condition == "mobile_gaussian":
        out = apply_rir(apply_gaussian_noise(audio, rng), mobile_rirs, rng)
    elif condition == "mobile_urban":
        out = apply_rir(apply_urban_noise(audio, urban_files, rng), mobile_rirs, rng)
    elif condition == "room_urban":
        out = apply_rir(apply_urban_noise(audio, urban_files, rng), room_rirs, rng)
    else:
        raise ValueError(f"Unknown augmentation condition: {condition}")

    # Volume scaling is part of the thesis degradation methodology.  It is
    # applied after normalization so it is not immediately cancelled out.
    return apply_volume_scaling(peak_normalize(out), rng)


if __name__ == "__main__":
    room, mobile, urban = collect_rir_files()
    print(f"Room RIRs   : {len(room)}")
    print(f"Mobile RIRs : {len(mobile)}")
    print(f"Urban noise : {len(urban)}")
    print("augmentation.py OK")
