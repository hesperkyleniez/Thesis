# Filipino Speech Deepfake Detection — Clean v11

This codebase contains only the four model configurations retained for the thesis:

1. **CNN-only** — Log-Mel spectrogram -> CNN classifier.
2. **CNN-GRU** — Log-Mel spectrogram -> three CNN blocks -> one 128-unit GRU -> classifier.
3. **CNN-GRU-F** — thesis feature-fusion model using the CNN-GRU Log-Mel embedding plus pooled MFCC and F0, weighted by learnable scalar parameters `alpha`, `beta`, and `gamma` before classification.
4. **CNN-GRU-CQCC-F** — prediction-level late fusion of an independent Log-Mel CNN-GRU expert and an independent CQCC expert. A regularized logistic fusion classifier is trained only from speaker-grouped out-of-fold predictions.

There are no standalone MFCC or F0 classifiers and no alternate prediction-fusion combinations in this version. CQCC exists only as the auxiliary expert required by model 4.

## Degradation handling

The original augmentation assignment proportions in `config.py` are unchanged:

- 50% clean
- 25% telephone
- 6.25% room + Gaussian
- 6.25% mobile + Gaussian
- 6.25% mobile + UrbanSound8K
- 6.25% room + UrbanSound8K

`prepare_dataset.py` reuses the existing `features/conditions.npy` assignment map when it exists. `run_degraded_test.py` likewise reuses the existing `features/test_degraded/test_degraded_conditions.npy` map, so the held-out degraded condition assignments are not reshuffled.

The degradation implementation was centralized in `augmentation.py` so training and degraded testing use the same code. The clean path remains the same: per-audio z-score normalization followed by peak normalization and the same feature extraction.

Robustness changes are limited to degraded audio:

- Gaussian noise remains at standard deviation `0.003`.
- UrbanSound8K mixing now uses the thesis value of **5 dB SNR**.
- RIR augmentation retains the existing room/mobile conditions and 0.04 wet ratio.
- Telephone audio now uses **DRC -> 300–3400 Hz telephone bandpass -> low-bitrate codec round-trip**.
- The telephone condition varies among GSM, Opus 12 kbps, and Opus 16 kbps to reduce dependence on one codec realization.
- FFmpeg is used for the real codec round-trip when available. A narrow-band companding fallback is included if FFmpeg is unavailable.
- Degraded audio receives the thesis volume scaling range `[0.8, 1.2]`.
- Paired Real/AI recordings use the same deterministic degradation draw within their assigned condition.

The classification threshold is **0.50 for both clean and degraded evaluation**. There is no degraded-test threshold tuning and no held-out-test calibration.

The clean evaluation pipeline is not modified by the robustness changes. Because the models are retrained on updated degraded training examples, however, the learned weights can still change and therefore identical clean-test metrics cannot be guaranteed.

## Recommended full run

Run these commands from `D:\Thesis\code` in this order:

```text
python prepare_dataset.py
python train.py --model all
python cqcc_train.py
python retrain.py --model all
python cqcc_retrain.py
python run_degraded_test.py
python evaluate.py --model all
python cqcc_fusion.py --rebuild-cache
python latency_test.py
```

`train.py --model all` trains only CNN-only, CNN-GRU, and CNN-GRU-F. `cqcc_train.py` and `cqcc_retrain.py` train the internal CQCC expert required for CNN-GRU-CQCC-F. `cqcc_fusion.py` fits the OOF late-fusion classifier and evaluates it on both clean and degraded held-out test data.

If the CNN-GRU and CQCC fold/final checkpoints have already been regenerated with this v11 feature cache, the fusion alone can be checked with:

```text
python cqcc_fusion.py --rebuild-cache
```

After the OOF cache has been created once, later fusion evaluation can be run with:

```text
python cqcc_fusion.py --eval-only
```

## Files

- `config.py` — paths, folds, augmentation split, feature parameters, training parameters.
- `augmentation.py` — single source of truth for all degraded-audio transformations.
- `dataset.py` — file discovery, speaker IDs, fold handling, 1-second/50%-overlap windowing.
- `features.py` — Log-Mel, MFCC, F0, and CQCC extraction.
- `prepare_dataset.py` — fixed training augmentation and clean held-out feature-cache generation.
- `run_degraded_test.py` — degraded held-out cache generation while preserving the existing assignment map.
- `train.py` — 4-fold CV for CNN-only, CNN-GRU, and CNN-GRU-F only.
- `retrain.py` — final development-set retraining for those three models.
- `evaluate.py` — clean/degraded evaluation for those three models.
- `cqcc_train.py` — 4-fold CV for the internal CQCC expert used by model 4.
- `cqcc_retrain.py` — final retraining of that CQCC expert.
- `cqcc_fusion.py` — OOF CNN-GRU + CQCC prediction-level late fusion and held-out evaluation.
- `latency_test.py` — per-window latency benchmark for all four retained model configurations.

## Important cache behavior

The feature cache version is now `v11_robust_telephone`. Old NPZ feature files are automatically rebuilt when `prepare_dataset.py` or `run_degraded_test.py` sees an older cache version. The condition-assignment `.npy` maps are intentionally retained so the assignment split stays fixed.
