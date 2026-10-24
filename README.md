# Improved CQCC–F0 Thesis System (16 Python files)

**Proposed model:** Filipino speech deepfake detection using a Log-Mel CNN–GRU expert, a separate **Akıncı–Özbay (2026)-inspired CQCC** expert, and a separate F0 expert, with **prediction-level late fusion and causal temporal evidence correction**.

This is the **same externally visible file structure** as the group's original code: 16 Python files, one README, and `run_all.bat`. Older experimental source implementations (formerly named v19, v22, v25, v27, v28) are consolidated as *readable, uncompressed source sections* **inside** `improvedCQCC_cqcc_f0_fusion.py`, rather than being distributed as additional Python files. The original F0 AUC-selected training routine is included inside `improvedCQCC_f0_train.py`; the paper-based CQCC feature extraction is inside `improvedCQCC_features.py`; original and paper-based preparation implementations are inside the two preparation scripts.

## Contents


| File                                | Purpose                                                                                                                             |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `improvedCQCC_config.py`            | Dataset locations, four speaker folds, processing parameters, original and new CQCC definitions                                     |
| `improvedCQCC_dataset.py`           | WAV discovery, speaker identification, audio windowing                                                                              |
| `improvedCQCC_augmentation.py`      | Original deterministic augmentations, telephone and noise/RIR effects                                                               |
| `improvedCQCC_features.py`          | Log-Mel/MFCC/F0, legacy baseline CQCC, **new 40-D CQT CQCC**                                                                        |
| `improvedCQCC_prepare_dataset.py`   | Original shared training/clean-test feature caches **and** new development/clean CQCC sidecars                                      |
| `improvedCQCC_run_degraded_test.py` | Original five-condition degraded test cache **and** new degraded CQCC sidecars                                                      |
| `improvedCQCC_train.py`             | CNN-only, CNN–GRU, and MFCC–F0 comparison baseline cross-validation                                                                 |
| `improvedCQCC_retrain.py`           | Final baseline CNN model retraining                                                                                                 |
| `improvedCQCC_evaluate.py`          | Clean/degraded tests for those three baselines                                                                                      |
| `improvedCQCC_cqcc_train.py`        | **Paper-based CQCC MLP** four speaker-held-out folds, train-only normalization, clip-AUC checkpoint selection                       |
| `improvedCQCC_cqcc_retrain.py`      | Final paper-based CQCC expert, 46 epochs in the recorded run                                                                        |
| `improvedCQCC_f0_train.py`          | F0 MLP plus the **AUC-selected** F0 fold training used by the proposed fusion                                                       |
| `improvedCQCC_f0_retrain.py`        | Final AUC-selected F0 expert                                                                                                        |
| `improvedCQCC_cqcc_f0_fusion.py`    | **Three-expert OOF prediction creation**, calibration, aligned-window fusion, causal temporal correction, clean/degraded evaluation |
| `improvedCQCC_latency_test.py`      | Four-model latency benchmark; includes actual new CQCC frontend in proposed-model timing                                            |
| `improvedCQCC_robustness.py`        | Condition-aware reporting and threshold utilities                                                                                   |
| `run_all.bat`                       | Stepwise start-from-audio development workflow; doesn't automatically evaluate on held-out data                                     |




## Exactly what is new

The final proposed CQCC implementation uses `librosa.cqt`, 16 kHz, **84 CQT bins**, **12 bins per octave**, **32.70 Hz minimum frequency**, log-power, DCT-II, **20 cepstral coefficients**, and mean/std pooling over the existing 1-second windows (0.5-second hop), yielding **40 features/window**. These are settings based on Akıncı and Özbay (2026), not a claim of bit-identical reproduction of their paper's entire processing code. 

The *three independent model outputs* are aligned per window, calibrated and fed to a clip-level logistic reference plus the **same causal temporal evidence-correction network as the tested Akıncı–Özbay temporal experiment**. The selected threshold was **0.43** in that recorded run, selected on development data.

### Why is the old CQCC field still in `features/`?

The three baseline models use the original shared feature cache. The same original cache also contains the older CQCC field, used in previously recorded experimental scaffolding. **The proposed system does not use that older CQCC as its actual acoustic input.** Its new `cqcc` values live in protected, aligned sidecars:

```
E:\Thesis\features\{train,test,test_degraded}\...
E:\Thesis\features_experiments\akinci_ozbay_cqcc\{train,test,test_degraded}\...
```

Keeping both caches is intentional: it makes the previous CNN–GRU and F0 recordings and model inputs reproducible, while the final classifier receives the new 40-D CQCC values. Neither cache is silently relabeled or mixed with another extraction version.

## Software/data requirements

- Python 3.11, NumPy, SciPy, scikit-learn, PyTorch, librosa, soundfile, tqdm, joblib; suitable FFmpeg executable for supported telephone-codec effects.
- Original training and held-out WAV folders plus noise/impulse-response resources.
- The original speaker split, seed, augmentation assignment maps, matching feature caches and model weights if using already trained models.
- Default root path is `E:\Thesis`; on another machine use Windows Command Prompt: `set THESIS_BASE_DIR=D:\Thesis`, *then* invoke Python in the extracted folder. The saved training run used an RTX 2060; CPU execution may be much slower.

**The files in this package no longer prepend the old E:/Thesis/code or D:/Thesis/code to sys.path.** This prevents accidentally importing scripts from another copy when testing in a separate directory. The underlying algorithms remain unchanged.

**Before rebuilding** an existing thesis workspace, back up `features/`, `models/`, and `results/`. Fresh cache extraction can rewrite the base cache. The paper-sidecar code protects existing incompatible sidecars and refuses unsafe overwrite. Retraining is stochastic; code/seed equivalence is **not** a promise of numerically identical checkpoints on different setups.

Note: put the conditions npz here: 

```cmd
E:\Thesis\
└── features\
    ├── conditions.npy
    └── test_degraded\
        └── test_degraded_conditions.npy
```



## Training from scratch (review each stage)

```cmd
cd /d E:\Thesis\code

REM 1. Original baseline features PLUS paper-based CQCC sidecars
python improvedCQCC_prepare_dataset.py --stage all --workers 6
python improvedCQCC_run_degraded_test.py --stage all --workers 6
python improvedCQCC_prepare_dataset.py --stage verify
python improvedCQCC_run_degraded_test.py --stage verify

REM 2. CNN comparison baselines
python improvedCQCC_train.py --model all
python improvedCQCC_retrain.py --model all

REM 3. Two improved prediction-fusion experts, four speaker-held-out folds
python improvedCQCC_cqcc_train.py --stage check
python improvedCQCC_cqcc_train.py --stage cv
python improvedCQCC_f0_train.py --stage cv

REM 4. OOF predictions + new temporal fusion (DEVELOPMENT ONLY)
python improvedCQCC_cqcc_f0_fusion.py --stage self-test
python improvedCQCC_cqcc_f0_fusion.py --stage oof
python improvedCQCC_cqcc_f0_fusion.py --stage fit

REM 5. Final expert weights (full development, epochs from CV)
python improvedCQCC_cqcc_retrain.py
python improvedCQCC_f0_retrain.py
```

The fusion script constructs the aligned OOF reference using pretrained CNN–GRU, the AUC-selected F0 and the new CQCC expert. If the original saved predecessor OOF is present, it is reused read-only for the precise CNN/F0 historical inputs. Otherwise the script generates a new reference OOF without training an obsolete CQCC classifier. This new reconstruction has compatible architecture/metadata but **cannot be promised byte-identical** to the saved reference OOF, so its resulting performance should be checked again.

### Evaluate only after development is complete

```cmd
python improvedCQCC_evaluate.py --model all
python improvedCQCC_cqcc_f0_fusion.py --stage evaluate
```

The held-out data had informed prior research iterations, so the resulting comparison remains **exploratory**. We do not use test/degraded-test labels to choose the new fusion threshold.

### Latency (with the new CQCC extraction)

```cmd
python improvedCQCC_latency_test.py --model CNN-only
python improvedCQCC_latency_test.py --model CNN-GRU
python improvedCQCC_latency_test.py --model CNN-GRU-F
python improvedCQCC_latency_test.py --model CNN-GRU-CQCC-F0-F --stage self-test
python improvedCQCC_latency_test.py --model CNN-GRU-CQCC-F0-F --stage check
python improvedCQCC_latency_test.py --model CNN-GRU-CQCC-F0-F --stage benchmark --runs 100 --history-windows 8
```

The proposed-model benchmark times feature extraction, the three experts, and temporal correction for **an already available one-second 16 kHz input**. Acquisition, disk I/O and resampling are excluded. The 1-second live-window CQCC and F0 representations can differ from the full-recording pooled features used during training/testing near boundaries; a latency measurement alone **does not prove accuracy equivalence** or deployed end-to-end real-time performance.

## Recorded performance targets (NOT newly re-measured by this source ZIP)


| Metric             | Clean (7,031 clips) | Degraded (7,031 paired clips) |
| ------------------ | ------------------- | ----------------------------- |
| Accuracy           | 93.02%              | 89.32%                        |
| F1                 | 92.74%              | 89.33%                        |
| Precision          | 96.17%              | 88.89                         |
| Recall             | 89.55%              | 89.78%                        |
| AUC                | 98.72%              | 95.90%                        |
| EER (lower better) | 7.11%               | 10.48%                        |


The ZIP contains **source code only**; to reproduce these specific checkpoint outputs, transfer the corresponding original CNN–GRU and F0 weights/normalization, CQCC cross-validation and final weights/normalization, experimental fusion artifacts and sidecar caches. Always validate data version and speaker assignment before comparing numerical results. This source package has passed Python syntax/import and synthetic CQCC/temporal/latency tests; a full training and 14,062-recording evaluation run was **not possible here** without the datasets and checkpoints.