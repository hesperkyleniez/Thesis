# Filipino Deepfake Speech Detection — runtime-optimized CQCC/FFV comparison

This revision prioritizes preprocessing and training runtime while keeping the thesis dataset split, 1.0 s windows with 50% overlap, speaker-grouped 4-fold CV, and degradation-condition proportions unchanged.

## New comparison models

### CQCC-FFV
A compact two-branch neural fusion model:

- CQCC: 40-D pooled vector -> compact MLP branch
- FFV: 14-D pooled vector -> compact MLP branch
- learned positive branch gates
- concatenation -> small MLP classifier -> fake/real logit

The motivation is Pal, Paul & Saha (Computer Speech & Language, 2018), who showed that CQCC spectral evidence and FFV pitch-variation evidence are complementary for synthetic-speech detection. The model here keeps that complementary pairing but uses a lightweight neural fusion backend instead of their GMM score fusion.

### CQCC-F0
The exact same fusion architecture and training protocol, except FFV is replaced by an 8-D YIN-derived F0/prosody vector. This makes the CQCC+FFV versus CQCC+F0 comparison controlled: CQCC, folds, optimizer, batch size, classifier capacity, early stopping, and 0.50 decision threshold are shared.

Run both:

```text
python cqcc_aux_models.py --model all --stage all
```

Or separately:

```text
python cqcc_aux_models.py --model CQCC-FFV --stage all
python cqcc_aux_models.py --model CQCC-F0 --stage all
```

## Why preparation is much faster

### CQCC
The defining CQCC pipeline is still CQT -> log power -> uniform-frequency spline resampling -> DCT. The CQT grid is changed from the extremely expensive original-default B=96 configuration to a research-backed B=12 / 84-bin configuration (62.5 Hz to 8 kHz), while retaining d=16 and 20 static coefficients including C0. The spline-resampling plus first 20 DCT coefficients are precomputed as one linear projection.

Most importantly, CQCC is computed once over each padded clip and then pooled into the existing overlapping 1-second windows. The old v13 implementation recomputed the full CQT independently for every overlapping window.

### FFV
FFV now follows the published Laskowski defaults more closely: 32 ms support, 8 ms frame step, 14 ms separation, 11/9 ms inner/outer window extents, Ng=512, and the published seven-filter bank. The code evaluates only rho values actually touched by those seven filters and vectorizes the dilation comparisons across frames.

### F0
The comparison model uses YIN rather than pYIN. This is deliberate for runtime: YIN gives the required frame-level F0 contour without pYIN's additional probabilistic candidate processing and Viterbi decoding. The contour is summarized with log-F0 level, spread, robust range, dynamics, voiced fraction, and voicing-transition rate.

### Clip-level extraction
`extract_clip_features()` computes expensive CQCC, FFV and F0 frame streams once per clip and pools them into the unchanged 1.0 s / 0.5 s-overlap windows. Log-Mel and MFCC remain per-window because they are already cheap.

## Degradation: same split, lower overhead

The assignment proportions are unchanged:

- clean 50%
- telephone 25%
- room + Gaussian 6.25%
- mobile + Gaussian 6.25%
- mobile + UrbanSound8K 6.25%
- room + UrbanSound8K 6.25%

The degradation implementation is optimized without changing those conditions:

- RIR and UrbanSound audio loads use an LRU cache.
- Telephone bandpass coefficients are designed once and reused.
- GSM / Opus codec simulation is retained, but FFmpeg now uses in-memory pipes instead of writing and rereading temporary source/encoded/decoded files.
- The fallback 8 kHz telephone path uses polyphase resampling.
- Feature preparation uses two file-level workers by default.
- Feature NPZ files are uncompressed by default for faster write/read. Set `COMPRESS_FEATURE_CACHE = True` in `config.py` if disk space matters more than speed.

Existing `conditions.npy` and degraded-test condition maps are still reused, so rerunning preparation does not reshuffle the degradation split.

## Recommended run order

```text
python prepare_dataset.py
python run_degraded_test.py
python cqcc_aux_models.py --model all --stage all
python latency_test.py --model CQCC-FFV
python latency_test.py --model CQCC-F0
```

The original Log-Mel CNN models are still present if they are needed for the thesis baseline comparison:

```text
python train.py --model all
python retrain.py --model all
python evaluate.py --model all
```

## Cache/version note

The optimized features intentionally use a new cache version because CQCC density, FFV implementation, and F0 extraction changed. Old v13 caches cannot be treated as equivalent. Current caches are skipped safely on reruns, and cache writes use an atomic temporary-file replacement so interrupting preparation does not leave a partially written file marked as complete.

## Research basis

- Todisco, Delgado & Evans (2016/2017): CQCC definition and anti-spoofing motivation.
- ASVspoof later CQCC baselines: lower B configurations demonstrate that B is a front-end parameter rather than a requirement that must always equal 96.
- Pal, Paul & Saha (2018): complementary CQCC and FFV evidence for synthetic-speech detection.
- Laskowski & Edlund (2010), plus Laskowski et al. (2008/2009): normative FFV framing, dilation spectrum and seven-filter representation.
- de Cheveigne & Kawahara (2002): YIN F0 estimator.

Performance is still an empirical question. The code therefore keeps speaker-disjoint CV, clean held-out evaluation, degraded held-out evaluation, per-condition degraded metrics, and latency measurement rather than assuming that a feature change must improve F1.

## Prediction-level fusion experiment (corrected)

The controlled comparison is:

- CNN-GRU + CQCC + FFV -> OOF prediction-level logistic fusion
- CNN-GRU + CQCC + F0  -> OOF prediction-level logistic fusion

Run in this order:

```bash
python prepare_dataset.py
python run_degraded_test.py
python train.py --model CNN-GRU
python retrain.py --model CNN-GRU
python cqcc_train.py
python cqcc_retrain.py
python ffv_train.py
python ffv_retrain.py
python f0_train.py
python f0_retrain.py
python cqcc_ffv_fusion.py
python cqcc_f0_fusion.py
python latency_test.py --model CNN-GRU-CQCC-FFV-F
python latency_test.py --model CNN-GRU-CQCC-F0-F
```

`cqcc_aux_models.py` is not part of the final prediction-fusion comparison; it is retained only as an earlier feature/embedding-fusion experiment.
