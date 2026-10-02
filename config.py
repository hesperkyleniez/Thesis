# config.py — all paths and parameters in one place
import os
import math

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = r"D:\Thesis"
TRAIN_DIR     = os.path.join(BASE_DIR, "Training and Validation Set")
TEST_DIR      = os.path.join(BASE_DIR, "Held out Test set")
CODE_DIR      = os.path.join(BASE_DIR, "code")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
FEATURES_DIR  = os.path.join(BASE_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

# RIR and noise sources
OPENSLR_DIR   = os.path.join(RESOURCES_DIR, "rirs_noises")
AIR_DIR       = os.path.join(RESOURCES_DIR, "air_rir")
URBANSOUND_DIR= os.path.join(RESOURCES_DIR, "urbansound8k")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Fold assignments ────────────────────────────────────────────────────────────
# Original G-speakers
FOLD_ASSIGNMENTS = {
    # Fold 1
    "G0009": 1, "G3011": 1,
    # Fold 2
    "G0005": 2, "G3002": 2,
    # Fold 3
    "G0007": 3, "G0010": 3,
    # Fold 4
    "G0006": 4, "G0004": 4,
    # New numeric speakers
    "0003": 1, "0006": 1, "0011": 1, "0012": 1, "0016": 1, "0021": 1,
    "0001": 2, "0008": 2, "0013": 2, "0018": 2, "0023": 2, "0027": 2,
    "0002": 3, "0009": 3, "0014": 3, "0019": 3, "0025": 3, "0028": 3,
    "0005": 4, "0010": 4, "0015": 4, "0020": 4, "0026": 4, "0029": 4,
}

# Test speakers — never used in training
TEST_SPEAKERS = {
    "G0008", "G3001",
    "0000", "0004", "0007", "0017", "0022", "0024"
}

# ── Audio parameters ────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
WINDOW_SEC    = 1.0
HOP_SEC       = 0.5
WIN_SAMPLES   = int(WINDOW_SEC * SAMPLE_RATE)   # 16000
HOP_SAMPLES   = int(HOP_SEC   * SAMPLE_RATE)   # 8000

# ── Feature parameters ──────────────────────────────────────────────────────────
N_MELS        = 128
N_MFCC        = 40

# Runtime-optimized CQCC front-end. The defining CQCC stages are unchanged:
# CQT -> log power -> uniform-frequency spline resampling -> DCT.
# B=12 is used by later ASVspoof CQCC baselines and a 2026 deepfake-speech
# cepstral comparison; 62.5 Hz..8 kHz gives 7 octaves / 84 CQT bins. This is
# far cheaper than the original B=96 setting while retaining the full speech
# band and the same 20 static coefficients used by the compact model.
CQCC_BINS_PER_OCTAVE = 12
CQCC_FMAX = SAMPLE_RATE / 2.0
CQCC_FMIN = 62.5
CQCC_OCTAVES = int(round(math.log2(CQCC_FMAX / CQCC_FMIN)))
CQCC_N_BINS = CQCC_BINS_PER_OCTAVE * CQCC_OCTAVES  # 84
CQCC_UNIFORM_SAMPLES_FIRST_OCTAVE = 16
CQCC_NUM_STATIC = 20
CQCC_HOP_LENGTH = 512
CQCC_VECTOR_DIM = CQCC_NUM_STATIC * 2  # frame mean + std => 40-D

N_FFT         = 400    # 25ms at 16kHz
HOP_LENGTH    = 160    # 10ms at 16kHz
TARGET_SHAPE  = (128, 128)
F0_MIN        = 50.0
F0_MAX        = 500.0




# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42

# ── F0 parameters ────────────────────────────────────────────────────────────
# YIN is used for the CQCC+F0 comparison to keep preprocessing inexpensive.
# The contour is summarized with robust level/dynamics/voicing statistics.
F0_FRAME_LENGTH = 2048
F0_HOP_LENGTH   = 512
F0_LOG          = True
F0_VECTOR_DIM   = 8

# FFV (Fundamental Frequency Variation) front-end. features.py follows the
# normative Laskowski defaults: 32 ms support, 8 ms frame step, 14 ms peak
# separation, 11/9 ms inner/outer window extents, Ng=512 and the published
# seven-filter bank. Only the filterbank-active rho samples are evaluated.
FFV_FRAME_MS = 32.0
FFV_HOP_MS = 8.0
FFV_SEPARATION_MS = 14.0
FFV_N_FFT = 1024
FFV_NUM_FILTERS = 7
FFV_VECTOR_DIM = FFV_NUM_FILTERS * 2  # frame mean + std => 14-D






# ── Degradation parameters ────────────────────────────────────────────────────
# Kept separate from AUGMENTATION_SPLITS so the assignment proportions never
# change when degradation strength is tuned. Values follow the current thesis.
GAUSSIAN_NOISE_STD = 0.003
URBAN_SNR_DB       = 5.0
RIR_WET_RATIO      = 0.04
RIR_MAX_SEC        = 0.10
# Low-bitrate telephone/VoIP channel variants. The condition assignment remains
# exactly "telephone"; this only varies the codec inside that assigned condition.
TELEPHONE_CODEC_MODES = ("gsm", "opus12", "opus16")

# Fixed augmentation conditions assigned once before training
# 50%   clean
# 25%   telephone
# 6.25% room RIR + Gaussian
# 6.25% mobile RIR + Gaussian
# 6.25% mobile RIR + UrbanSound8K
# 6.25% room RIR + UrbanSound8K
AUGMENTATION_SPLITS = {
    "clean"              : 0.50,
    "telephone"          : 0.25,
    "room_gaussian"      : 0.0625,
    "mobile_gaussian"    : 0.0625,
    "mobile_urban"       : 0.0625,
    "room_urban"         : 0.0625,
}


# ── Dataset preparation runtime ───────────────────────────────────────────────
# Two file-level workers is a conservative laptop default. Increase to 3-4 only
# if CPU/RAM headroom is available; feature libraries already use native code.
PREP_WORKERS = 2
# Uncompressed NPZ is substantially faster to write/read. Set True only when
# disk space matters more than preparation/training startup time.
COMPRESS_FEATURE_CACHE = False

# ── Model hyperparameters ───────────────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 50
PATIENCE      = 7
LR_FACTOR     = 0.5
DROPOUT       = 0.5

# ── Per-file window cap for HybridDataset ───────────────────────────────────────
# None = keep every window from every file (recommended). HybridDataset already
# decompresses each file's full window array regardless of how many windows are
# kept, so capping this does not save any I/O -- it only removes training
# diversity. Set an integer here only if you hit a genuine RAM ceiling; if so,
# use the highest number you can afford rather than a small one.
# Cap the number of cached windows contributed by each original training file.
# This is applied AFTER feature extraction, during training dataset assembly.
# It prevents unusually long recordings from dominating the optimization while
# preserving every cached window for validation/test evaluation.
MAX_TRAIN_WINDOWS_PER_FILE = 6
MAX_VAL_WINDOWS_PER_FILE   = None

# Final retraining budget policy. The median of the four CV best epochs is used
# instead of the maximum, which avoids letting one noisy fold determine a long
# final training run.
RETRAIN_EPOCH_POLICY = "median"

# CQCC expert used only inside CNN-GRU + CQCC prediction-level late fusion.
CQCC_BATCH_SIZE = 128
CQCC_HIDDEN = 64
CQCC_DROPOUT = 0.35
CQCC_MAX_EPOCHS = 50
CQCC_PATIENCE = 7
FFV_BATCH_SIZE = 128
FFV_HIDDEN = 64
FFV_DROPOUT = 0.35
FFV_MAX_EPOCHS = 50
FFV_PATIENCE = 7

# ── CQCC auxiliary-pair models ────────────────────────────────────────────────
AUX_PAIR_BATCH_SIZE = 256
AUX_PAIR_HIDDEN = 64
AUX_PAIR_DROPOUT = 0.30
AUX_PAIR_MAX_EPOCHS = 40
AUX_PAIR_PATIENCE = 6

# ── Labels ──────────────────────────────────────────────────────────────────────
LABEL_REAL = 0
LABEL_AI   = 1
