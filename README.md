# Filipino Deepfake Speech Detection

This project compares four models:

| Model | Command name |
|---|---|
| CNN-only | `CNN-only` |
| CNN-GRU | `CNN-GRU` |
| CNN-GRU with MFCC and F0 (late feature fusion) | `CNN-GRU-F` |
| CNN-GRU with CQCC and F0 (late prediction fusion) | `CNN-GRU-CQCC-F0-F` |

## Setup

Extract the `code` folder into `D:\Thesis`. Run commands from `D:\Thesis\code`.
The existing paths in `config.py` expect datasets under `D:\Thesis\Training and Validation Set` and `D:\Thesis\Held out Test set`, with `Real_Dataset` and `AI_Dataset` subfolders.

Use Python with NumPy, SciPy, librosa, PyTorch, scikit-learn, and tqdm. Install FFmpeg and put it on PATH for GSM/Opus telephone simulation. Noise and room-response resources belong under `D:\Thesis\resources`.

## Prepare features

For a first run:

```text
python prepare_dataset.py
python run_degraded_test.py
```

For a robustness rerun with existing clean and degraded held-out caches, use only:

```text
python prepare_dataset.py --train-only
```

## Train and evaluate the four models

Run in this order:

```text
python train.py --model all
python retrain.py --model all
python evaluate.py --model all
python cqcc_train.py
python cqcc_retrain.py
python f0_train.py
python f0_retrain.py
python cqcc_f0_fusion.py --rebuild-cache
```

The first three commands cover the three CNN models. CQCC and F0 are supporting experts for the fourth model, not additional comparison models. The fusion command fits the prediction fusion and evaluates both held-out splits.

## Measure latency

```text
python latency_test.py --model CNN-only
python latency_test.py --model CNN-GRU
python latency_test.py --model CNN-GRU-F
python latency_test.py --model CNN-GRU-CQCC-F0-F
```

`python latency_test.py --model all` measures exactly these four models.

Outputs are saved under `D:\Thesis\features`, `models`, and `results`.
Only the four comparison models are retained, along with their supporting CQCC and F0 experts. Latency testing was updated to remove the other model variants; training, evaluation, and feature extraction are unchanged. Existing feature preparation still extracts FFV as well.

For an automatic run, double-click `run_all.bat` inside the code folder. It checks Python and FFmpeg, runs the stages above in order, reuses existing held-out caches, and stops on an error.
