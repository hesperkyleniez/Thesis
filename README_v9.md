# v9 Fusion Experiment Pipeline

This version keeps the existing CNN-GRU as the primary learned Log-Mel model and
adds a controlled prediction-level fusion experiment using independent MFCC,
CQCC, and F0 classifiers.

## Experiment matrix

| ID | Configuration | Representation(s) | Combination |
|---|---|---|---|
| A | CNN-GRU Baseline | Log-Mel | None |
| B | MFCC Baseline | MFCC | None |
| C | CQCC Baseline | CQCC | None |
| D | F0 Baseline | F0 | None |
| E | CNN-GRU + MFCC | Log-Mel + MFCC | Prediction level |
| F | CNN-GRU + CQCC | Log-Mel + CQCC | Prediction level |
| G | CNN-GRU + F0 | Log-Mel + F0 | Prediction level |
| H | CNN-GRU + MFCC + CQCC + F0 | Log-Mel + MFCC + CQCC + F0 | Prediction level |
| I | CNN-GRU + MFCC + F0 | Log-Mel + MFCC + F0 | Feature-level gated fusion |

A is the primary baseline. B-D establish the independent auxiliary branches.
E-G test whether each auxiliary representation adds complementary evidence.
H tests full prediction fusion. I preserves the existing feature-level gated
fusion experiment as a direct comparison against prediction-level fusion.

## Important methodological choices

### 1. Speaker independent evaluation remains unchanged

The existing four speaker-grouped CV folds and six held-out speakers are kept.
The held-out speakers are not used to train the fusion weights.

### 2. Window cap

Training datasets use:

    MAX_TRAIN_WINDOWS_PER_FILE = 6

This cap is applied during training dataset assembly, not during feature-cache
creation. Every window remains available in the NPZ cache.

The six windows are selected deterministically and evenly across each long file.
This prevents unusually long recordings from dominating training while retaining
coverage of the beginning, middle, and end of the recording.

Validation and test sets keep all windows.

### 3. Median final retraining epoch

Final retraining uses the median of the four CV best epochs rather than the
maximum. This prevents one unusually late fold from unnecessarily extending
the final training run.

The policy is controlled by:

    RETRAIN_EPOCH_POLICY = "median"

### 4. CQCC

CQCC is added as an independent auxiliary representation. The implementation is
a compact CQCC-style pipeline:

    CQT -> log power -> DCT -> first 40 coefficients -> mean + std

This produces an 80-dimensional vector like the MFCC branch.

### 5. Prediction-level fusion

Each branch is trained independently. Each produces a clip-level probability
by averaging its per-window probabilities.

The fusion weights are learned from validation predictions only:

    P_final = sum(w_i * P_i)

with:

    w_i >= 0
    sum(w_i) = 1

Weights are optimized using validation binary log-loss. The learned weights are
then frozen before evaluating held-out speakers.

The test set is never used to learn fusion weights.

## Recommended run order

From:

    D:\Thesis\code

### Step 1. Rebuild feature caches

This is required because old NPZ files do not contain CQCC.

    python prepare_dataset.py

The script automatically rebuilds old caches and writes `cache_version=v9_cqcc`.

### Step 2. Train the original CNN family

    python train.py --model CNN-GRU
    python train.py --model CNN-only
    python train.py --model CNN-GRU-F

If you want the complete CV matrix for the CNN family:

    python train.py --model all

### Step 3. Train independent auxiliary models

    python aux_train.py --model all

This trains:

    MFCC
    CQCC
    F0

with the same four speaker-grouped folds.

### Step 4. Final retraining

For the CNN family:

    python retrain.py --model all

For the auxiliary models:

    python aux_retrain.py --model all

Final retraining uses the median CV best epoch.

### Step 5. Generate degraded test cache

    python run_degraded_test.py

The degraded cache now also contains CQCC.

### Step 6. Evaluate CNN family

    python evaluate.py --model all

### Step 7. Evaluate auxiliary baselines

    python evaluate_aux.py --model all

### Step 8. Run prediction fusion

    python prediction_fusion.py --fusion all

This evaluates:

    CNN-GRU + MFCC
    CNN-GRU + CQCC
    CNN-GRU + F0
    CNN-GRU + MFCC + CQCC
    CNN-GRU + MFCC + F0
    CNN-GRU + CQCC + F0
    CNN-GRU + MFCC + CQCC + F0

for both:

    test
    test_degraded

## Runtime notes

The expensive part is feature extraction and CNN-GRU training. The auxiliary
MLPs are deliberately small and use a larger auxiliary batch size.

Prediction fusion itself is cheap because it does not retrain the base experts.
It only runs the trained experts over the same validation/test files and learns
a small set of fusion weights.

## Outputs

CNN CV:

    models\CNN-GRU\fold_results.npy
    models\CNN-GRU\best_fold*.pt
    models\CNN-GRU\final_model.pt

Auxiliary CV:

    models\MFCC\fold_results.npy
    models\CQCC\fold_results.npy
    models\F0\fold_results.npy

Fusion:

    models\Prediction-Fusion\*.npy
    results\prediction_fusion_results.npy

## Important comparison rule

Do not compare the paper's reported 99% result directly with this pipeline.
The paper uses a different dataset and split design. The purpose of this
experiment is to determine whether auxiliary representations improve the
speaker-independent Filipino deepfake detector under the same evaluation
protocol used throughout this thesis.
