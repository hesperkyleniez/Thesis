# config.py — all paths and parameters in one place
import os

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
# CQCC uses 40 coefficients, matching MFCC dimensionality after pooling.
N_CQCC        = 40
CQCC_FMIN     = 32.70319566257483   # C1, Hz
CQCC_BINS     = 84                  # 7 octaves × 12 bins/octave
CQCC_BINS_PER_OCTAVE = 12
N_FFT         = 400    # 25ms at 16kHz
HOP_LENGTH    = 160    # 10ms at 16kHz
TARGET_SHAPE  = (128, 128)
F0_MIN        = 50.0
F0_MAX        = 500.0




# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42

# ── F0 parameters (updated for acoustic validity) ───────────────────────────
# Standard speech analysis: 25ms frame, 10ms hop
# At 16kHz: frame=400 samples, hop=160 samples
# We use slightly larger for efficiency: frame=2048, hop=512
F0_FRAME_LENGTH = 2048   # 128ms at 16kHz — acoustically valid
F0_HOP_LENGTH   = 512    # 32ms at 16kHz — ~31 frames per 1-second window
# Log F0 is used before computing mean/std (perceptually meaningful)
F0_LOG          = True





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

# ── Fusion gate entropy regularization ──────────────────────────────────────────
# CNNGRUFusion's gate is a softmax over 3 branches (CNN-GRU / MFCC / F0) with no
# balancing term, which makes it susceptible to "modality laziness" / gate
# collapse: one branch dominates early and starves the others of gradient
# signal (Wang, Tran & Feiszli, CVPR 2020, "What Makes Training Multi-modal
# Classification Networks Hard?"; Peng et al., CVPR 2022, "Balanced Multimodal
# Learning via On-the-fly Gradient Modulation"; the underlying mechanic is the
# same one Mixture-of-Experts load-balancing losses exist to prevent, e.g.
# Shazeer et al. 2017). GATE_ENTROPY_WEIGHT adds a small penalty during
# training (train.py / retrain.py only, not validation) that discourages the
# gate from collapsing onto a single branch. 0.0 disables it. Start small
# (0.01-0.05) and watch get_fusion_weights() move away from a degenerate
# [~1, ~0, ~0] split across training.
GATE_ENTROPY_WEIGHT = 0.005

# Final retraining budget policy. The median of the four CV best epochs is used
# instead of the maximum, which avoids letting one noisy fold determine a long
# final training run.
RETRAIN_EPOCH_POLICY = "median"

# Auxiliary prediction-fusion training.
AUX_BATCH_SIZE = 128
AUX_HIDDEN = 64
AUX_DROPOUT = 0.35
AUX_MAX_EPOCHS = 50
AUX_PATIENCE = 7

# ── Labels ──────────────────────────────────────────────────────────────────────
LABEL_REAL = 0
LABEL_AI   = 1