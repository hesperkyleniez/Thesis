"""Degradation pipelines with the thesis condition split unchanged.

Runtime optimizations are implementation-only: cached audio resources, a
precomputed SOS telephone filter, polyphase fallback resampling, and in-memory
FFmpeg codec pipes instead of per-file temporary WAV/codec files.
"""

import os
import random
import shutil
import subprocess
import urllib.request
import zipfile
from functools import lru_cache

import librosa
import numpy as np
from scipy.signal import fftconvolve, butter, sosfiltfilt, resample_poly

from config import (
    SAMPLE_RATE, OPENSLR_DIR, AIR_DIR, URBANSOUND_DIR, RESOURCES_DIR,
    URBAN_SNR_DB, GAUSSIAN_NOISE_STD, RIR_WET_RATIO, RIR_MAX_SEC,
    TELEPHONE_CODEC_MODES,
)

_TELEPHONE_SOS = butter(
    4, [300.0, 3400.0], btype="bandpass", fs=SAMPLE_RATE, output="sos"
)


def setup_resources():
    os.makedirs(RESOURCES_DIR, exist_ok=True)
    openslr_zip = os.path.join(RESOURCES_DIR, "rirs_noises.zip")
    if not os.path.exists(OPENSLR_DIR) or not os.listdir(OPENSLR_DIR):
        print("Downloading OpenSLR RIRs...")
        if not os.path.exists(openslr_zip):
            urllib.request.urlretrieve(
                "https://www.openslr.org/resources/28/rirs_noises.zip", openslr_zip
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
    room_rirs, mobile_rirs = [], []
    for path in _collect_wavs(OPENSLR_DIR):
        lower = path.lower()
        if "real_rirs" in lower or "real_rir" in lower:
            room_rirs.append(path)
        elif "simulated" in lower or "pointsource" in lower:
            mobile_rirs.append(path)
        elif "rir" in lower:
            room_rirs.append(path)
    mobile_rirs.extend(_collect_wavs(AIR_DIR))
    urban_files = _collect_wavs(URBANSOUND_DIR)
    return sorted(set(room_rirs)), sorted(set(mobile_rirs)), urban_files


@lru_cache(maxsize=96)
def _load_audio_cached(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(y, dtype=np.float32)


def peak_normalize(audio):
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio if peak <= 0.0 else (audio / (peak + 1e-8)).astype(np.float32)


def _choice(seq, rng):
    return None if not seq else seq[rng.randrange(len(seq))]


def apply_gaussian_noise(audio, rng):
    np_rng = np.random.default_rng(rng.randrange(0, 2**32))
    noise = np_rng.normal(0.0, GAUSSIAN_NOISE_STD, len(audio)).astype(np.float32)
    return (np.asarray(audio, dtype=np.float32) + noise).astype(np.float32)


def apply_urban_noise(audio, urban_files, rng, snr_db=URBAN_SNR_DB):
    noise_path = _choice(urban_files, rng)
    if noise_path is None:
        return np.asarray(audio, dtype=np.float32)
    noise = _load_audio_cached(noise_path)
    audio = np.asarray(audio, dtype=np.float32)
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / max(len(noise), 1))))
    noise = noise[:len(audio)]
    p_signal = float(np.mean(audio**2)) + 1e-8
    p_noise = float(np.mean(noise**2)) + 1e-8
    scale = np.sqrt(p_signal / (p_noise * (10.0 ** (snr_db / 10.0))))
    return peak_normalize(audio + noise * scale)


def apply_rir(audio, rir_files, rng, wet_ratio=RIR_WET_RATIO):
    rir_path = _choice(rir_files, rng)
    if rir_path is None:
        return np.asarray(audio, dtype=np.float32)
    rir = _load_audio_cached(rir_path)[:int(RIR_MAX_SEC * SAMPLE_RATE)]
    if len(rir) == 0:
        return np.asarray(audio, dtype=np.float32)
    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    audio = np.asarray(audio, dtype=np.float32)
    convolved = fftconvolve(audio, rir, mode="full")[:len(audio)]
    return peak_normalize((1.0 - wet_ratio) * audio + wet_ratio * convolved)


def _dynamic_range_compression(audio):
    threshold, ratio = 0.90, 1.20
    out = np.asarray(audio, dtype=np.float32).copy()
    mask = np.abs(out) > threshold
    out[mask] = np.sign(out[mask]) * (
        threshold + (np.abs(out[mask]) - threshold) / ratio
    )
    return out


def _telephone_bandpass(audio):
    return sosfiltfilt(_TELEPHONE_SOS, np.asarray(audio, dtype=np.float32)).astype(np.float32)


def _codec_roundtrip_ffmpeg(audio, mode):
    """Low-bitrate codec round-trip entirely through pipes.

    This preserves the GSM/Opus codec degradation but removes creation,
    filesystem I/O and decoding of three temporary files per telephone clip.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    x = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2", copy=False).tobytes()
    common_in = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
    ]

    if mode == "gsm":
        enc = common_in + ["-ar", "8000", "-ac", "1", "-c:a", "libgsm_ms", "-f", "wav", "pipe:1"]
        input_fmt = "wav"
    elif mode == "opus12":
        enc = common_in + [
            "-ar", "16000", "-ac", "1", "-c:a", "libopus", "-application", "voip",
            "-b:a", "12k", "-vbr", "off", "-f", "ogg", "pipe:1",
        ]
        input_fmt = "ogg"
    else:
        enc = common_in + [
            "-ar", "16000", "-ac", "1", "-c:a", "libopus", "-application", "voip",
            "-b:a", "16k", "-vbr", "off", "-f", "ogg", "pipe:1",
        ]
        input_fmt = "ogg"

    try:
        encoded = subprocess.run(
            enc, input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True,
        ).stdout
        dec = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", input_fmt, "-i", "pipe:0", "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-f", "f32le", "pipe:1",
        ]
        raw = subprocess.run(
            dec, input=encoded, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True,
        ).stdout
        if not raw:
            return None
        return np.frombuffer(raw, dtype="<f4").copy()
    except (subprocess.CalledProcessError, OSError):
        return None


def _mulaw_fallback(audio):
    audio = peak_normalize(audio)
    mu = 255.0
    encoded = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    encoded = np.round((encoded + 1.0) * 127.5) / 127.5 - 1.0
    decoded = np.sign(encoded) * (np.expm1(np.abs(encoded) * np.log1p(mu)) / mu)
    return np.asarray(decoded, dtype=np.float32)


def apply_telephone(audio, rng):
    """DRC -> 300-3400 Hz -> existing GSM/Opus low-bitrate codec variants."""
    x = peak_normalize(_telephone_bandpass(_dynamic_range_compression(audio)))
    mode = _choice(list(TELEPHONE_CODEC_MODES), rng) or "gsm"
    coded = _codec_roundtrip_ffmpeg(x, mode)
    if coded is None:
        narrow = resample_poly(x, up=1, down=2).astype(np.float32)
        narrow = _mulaw_fallback(narrow)
        coded = resample_poly(narrow, up=2, down=1).astype(np.float32)
    if len(coded) < len(audio):
        coded = np.pad(coded, (0, len(audio) - len(coded)))
    return peak_normalize(coded[:len(audio)])


def apply_volume_scaling(audio, rng):
    return (np.asarray(audio, dtype=np.float32) * rng.uniform(0.8, 1.2)).astype(np.float32)


def apply_augmentation(audio, condition, room_rirs, mobile_rirs, urban_files, rng=None):
    """Apply one fixed thesis degradation condition; proportions are untouched."""
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
    return apply_volume_scaling(peak_normalize(out), rng)


if __name__ == "__main__":
    room, mobile, urban = collect_rir_files()
    print(f"Room RIRs   : {len(room)}")
    print(f"Mobile RIRs : {len(mobile)}")
    print(f"Urban noise : {len(urban)}")
    print("augmentation.py OK")
