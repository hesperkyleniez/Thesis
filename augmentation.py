# augmentation.py — all degradation pipelines
import os
import random
import zipfile
import urllib.request
import numpy as np
import librosa
import soundfile as sf
import subprocess
import tempfile
from scipy.signal import fftconvolve, butter, filtfilt
from config import SAMPLE_RATE, OPENSLR_DIR, AIR_DIR, URBANSOUND_DIR, RESOURCES_DIR


def setup_resources():
    """
    Downloads and extracts RIR and noise resources if not present.
    Run once before training.
    """
    os.makedirs(RESOURCES_DIR, exist_ok=True)

    # OpenSLR RIRs
    openslr_zip = os.path.join(RESOURCES_DIR, "rirs_noises.zip")
    if not os.path.exists(OPENSLR_DIR) or len(os.listdir(OPENSLR_DIR)) == 0:
        print("Downloading OpenSLR RIRs...")
        if not os.path.exists(openslr_zip):
            urllib.request.urlretrieve(
                "https://www.openslr.org/resources/28/rirs_noises.zip",
                openslr_zip
            )
        print("Extracting OpenSLR RIRs...")
        os.makedirs(OPENSLR_DIR, exist_ok=True)
        with zipfile.ZipFile(openslr_zip, 'r') as z:
            z.extractall(OPENSLR_DIR)
        print("✅ OpenSLR RIRs ready")
    else:
        print("✅ OpenSLR RIRs already present")

    # AIR database from GitHub
    air_zip = os.path.join(RESOURCES_DIR, "air_rir.zip")
    if not os.path.exists(AIR_DIR) or len(os.listdir(AIR_DIR)) == 0:
        print("Downloading AIR database...")
        if not os.path.exists(air_zip):
            urllib.request.urlretrieve(
                "https://github.com/Graphi07/room-impulse-responses/"
                "archive/refs/heads/master.zip",
                air_zip
            )
        print("Extracting AIR database...")
        os.makedirs(AIR_DIR, exist_ok=True)
        with zipfile.ZipFile(air_zip, 'r') as z:
            z.extractall(AIR_DIR)
        print("✅ AIR database ready")
    else:
        print("✅ AIR database already present")

    return collect_rir_files()


def collect_rir_files():
    """Collect room and mobile RIR file paths."""
    room_rirs   = []
    mobile_rirs = []
    urban_files = []

    # OpenSLR — real_rirs → room, simulated → mobile
    for root, _, files in os.walk(OPENSLR_DIR):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            path  = os.path.join(root, f)
            lower = path.lower()
            if "real_rirs" in lower or "real_rir" in lower:
                room_rirs.append(path)
            elif "simulated" in lower or "pointsource" in lower:
                mobile_rirs.append(path)
            elif "rir" in lower:
                room_rirs.append(path)

    # AIR database → mobile/device IRs
    for root, _, files in os.walk(AIR_DIR):
        for f in files:
            if f.lower().endswith(".wav"):
                mobile_rirs.append(os.path.join(root, f))

    # UrbanSound8K
    for root, _, files in os.walk(URBANSOUND_DIR):
        for f in files:
            if f.lower().endswith(".wav"):
                urban_files.append(os.path.join(root, f))

    print(f"   Room RIRs   : {len(room_rirs)}")
    print(f"   Mobile RIRs : {len(mobile_rirs)}")
    print(f"   Urban noise : {len(urban_files)}")

    return room_rirs, mobile_rirs, urban_files


def apply_gaussian_noise(audio):
    """Gaussian white noise at scaling factor 0.003."""
    noise = np.random.normal(0.0, 0.003, len(audio)).astype(np.float32)
    return (audio + noise)


def apply_urban_noise(audio, urban_files, snr_db=30.0):
    """Mix with UrbanSound8K at target SNR."""
    if not urban_files:
        return audio

    noise, _ = librosa.load(random.choice(urban_files),
                             sr=SAMPLE_RATE, mono=True)
    noise = noise.astype(np.float32)

    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    noise = noise[:len(audio)]

    p_signal = np.mean(audio ** 2) + 1e-8
    p_noise  = np.mean(noise ** 2) + 1e-8
    scale    = np.sqrt(p_signal / (p_noise * (10 ** (snr_db / 10))))
    mixed    = audio + noise * scale

    max_val  = np.max(np.abs(mixed))
    return mixed / (max_val + 1e-8) if max_val > 0 else mixed


def apply_rir(audio, rir_files, wet_ratio=0.04):
    """Convolve with room impulse response. y(n) = x(n) * h(n)"""
    if not rir_files:
        return audio

    rir, _ = librosa.load(random.choice(rir_files),
                           sr=SAMPLE_RATE, mono=True)
    rir    = rir.astype(np.float32)

    # Limit RIR length to 0.1 seconds
    max_len = int(0.1 * SAMPLE_RATE)
    rir     = rir[:max_len]
    rir     = rir / (np.max(np.abs(rir)) + 1e-8)

    convolved = fftconvolve(audio, rir, mode='full')[:len(audio)]
    mixed     = (1 - wet_ratio) * audio + wet_ratio * convolved

    max_val   = np.max(np.abs(mixed))
    return mixed / (max_val + 1e-8) if max_val > 0 else mixed


def apply_telephone(audio):
    """
    Telephone codec simulation:
    DRC + 300-3400Hz bandpass (telephone bandwidth)
    """
    from scipy.signal import butter, filtfilt

    # Soft-knee dynamic range compression
    threshold  = 0.9
    ratio      = 1.2
    compressed = audio.copy()
    mask       = np.abs(compressed) > threshold
    compressed[mask] = (
        np.sign(compressed[mask]) *
        (threshold + (np.abs(compressed[mask]) - threshold) / ratio)
    )

    # Telephone bandwidth: 300-3400Hz
    nyq  = SAMPLE_RATE / 2
    low  = 300  / nyq
    high = 3400 / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, compressed)

    max_val = np.max(np.abs(filtered))
    return filtered / (max_val + 1e-8) if max_val > 0 else filtered


def apply_augmentation(audio, condition, room_rirs, mobile_rirs, urban_files):
    """
    Apply augmentation based on assigned condition.

    Conditions:
      clean           — no change
      telephone       — DRC + bandpass
      room_gaussian   — Gaussian noise + room RIR
      mobile_gaussian — Gaussian noise + mobile RIR
      mobile_urban    — UrbanSound8K + mobile RIR
      room_urban      — UrbanSound8K + room RIR
    """
    if condition == "clean":
        return audio

    elif condition == "telephone":
        return apply_telephone(audio)

    elif condition == "room_gaussian":
        audio = apply_gaussian_noise(audio)
        audio = apply_rir(audio, room_rirs)
        return audio

    elif condition == "mobile_gaussian":
        audio = apply_gaussian_noise(audio)
        audio = apply_rir(audio, mobile_rirs)
        return audio

    elif condition == "mobile_urban":
        audio = apply_urban_noise(audio, urban_files)
        audio = apply_rir(audio, mobile_rirs)
        return audio

    elif condition == "room_urban":
        audio = apply_urban_noise(audio, urban_files)
        audio = apply_rir(audio, room_rirs)
        return audio

    return audio


if __name__ == "__main__":
    print("Setting up resources...")
    room_rirs, mobile_rirs, urban_files = setup_resources()
    print("✅ augmentation.py OK")