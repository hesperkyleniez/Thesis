# Filipino Deepfake Speech Detection — v13

This package retains four reported model configurations:

1. **CNN-only** — Log-Mel -> CNN.
2. **CNN-GRU** — Log-Mel -> CNN -> GRU.
3. **CNN-GRU-F** — original feature-fusion comparison, now using Log-Mel CNN-GRU + MFCC + a richer F0/prosody vector.
4. **CNN-GRU-CQCC-FFV-F** — refined prediction-level late fusion of independent CNN-GRU, canonical CQCC, and FFV experts.

## Feature definitions

### Canonical CQCC
CQCC follows Todisco, Delgado & Evans: CQT -> log power -> uniform frequency resampling -> DCT. Twenty static coefficients including C0 are summarized by frame mean+std (40-D). The resampling stage is intentionally retained; this is not the earlier "CQCC-style" shortcut.

### FFV
FFV follows the Laskowski FFV principle: left/right half-frame magnitude spectra are compared under log-frequency dilation and summarized with seven pitch-change regions. Frame mean+std gives a 14-D vector. FFV is trained as an independent expert. This is motivated by Pal, Paul & Saha (2018), who used FFV as complementary pitch-variation evidence with CQCC for synthetic-speech detection.

### Improved F0 in CNN-GRU-F
The original model still uses MFCC + F0, but F0 is no longer only mean/std. pYIN is used to obtain a frame-level contour and the fixed vector contains log-F0 mean, std, median, IQR, delta absolute mean, delta std, voiced fraction, and voicing-transition rate (8-D). This preserves the original F0 concept while retaining substantially more temporal/voicing information.

## Degradation methodology

The thesis assignment split is unchanged:
- clean 50%
- telephone 25%
- room+Gaussian 6.25%
- mobile+Gaussian 6.25%
- mobile+UrbanSound8K 6.25%
- room+UrbanSound8K 6.25%

Existing `conditions.npy` and degraded-test condition maps are reused. The v12 thesis-aligned degradation implementation is retained: Gaussian std 0.003, UrbanSound8K 5 dB SNR, RIR wet ratio 0.04 / max 0.10 s, and telephone DRC + 300–3400 Hz band limiting + low-bitrate GSM/Opus codec simulation. No degradation-specific transform is applied to the clean held-out test set.

## Selective retraining / overwrite behavior

Architecture versions are tracked independently.
- **CNN-only** keeps version `v11_clean_four_models_robust_telephone`.
- **CNN-GRU** keeps version `v11_clean_four_models_robust_telephone`.
- **CNN-GRU-F** is version `v13_mfcc_rich_f0` because its F0 input changed from 2-D to 8-D.
- **CQCC** keeps the canonical v12 expert version, so a compatible v12 CQCC expert can be reused.
- **FFV** is new and must train.
- **CNN-GRU-CQCC-FFV-F** is new and must fit its OOF late-fusion model.

Therefore, rerunning `train.py --model all` skips compatible existing CNN-only/CNN-GRU folds and retrains only incompatible/new architectures. `retrain.py` applies the same per-model version check to final checkpoints.

## Run order

```text
python prepare_dataset.py
python train.py --model all
python cqcc_train.py
python ffv_train.py
python retrain.py --model all
python cqcc_retrain.py
python ffv_retrain.py
python run_degraded_test.py
python evaluate.py --model all
python cqcc_ffv_fusion.py --rebuild-cache
python latency_test.py
```

`prepare_dataset.py` rebuilds feature caches because F0/FFV changed, but it preserves the existing degradation assignment map. `run_degraded_test.py` likewise preserves the existing held-out degraded condition map.

## Important

The code changes are methodologically motivated; they do **not** guarantee improved degraded F1. Performance must be established by the new speaker-disjoint CV and held-out clean/degraded evaluations. The fixed 0.50 classification threshold is retained.
