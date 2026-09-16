# evaluate.py — Test set evaluation with soft voting
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

sys.path.insert(0, r"D:\Thesis\code")
from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, SEED
from train import (
    CNNOnly, CNNGRUOnly, CNNGRUFusion, set_seed,
    compute_per_speaker_f0_stats, lookup_f0_stats,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "CNN-only"  : CNNOnly,
    "CNN-GRU"   : CNNGRUOnly,
    "CNN-GRU-F" : CNNGRUFusion,
}

def get_test_files(split="test"):
    """Get all NPZ files from test or test_degraded split."""
    return glob.glob(
        os.path.join(FEATURES_DIR, split, "**", "*.npz"),
        recursive=True
    )

def load_fold_norm_stats(model_name, fold):
    """
    Load MFCC normalization stats saved during training. NOTE: this no
    longer returns F0 stats -- F0 is per-speaker normalized, and this
    function's whole fold is from train/val speakers, not test speakers,
    so those stats would be meaningless here anyway. evaluate_on_test()
    computes F0 stats fresh from the test set's own speakers instead.
    """
    results_path = os.path.join(
        MODELS_DIR, model_name, "fold_results.npy")
    results = list(np.load(results_path, allow_pickle=True))
    for r in results:
        if int(r['fold']) == fold:
            return (
                r['norm']['mfcc_mean'],
                r['norm']['mfcc_std'],
            )
    raise ValueError(f"Fold {fold} not found in results")

def evaluate_on_test(model_name, model_class, test_split="test"):
    """
    Evaluate a trained model on the test set.
    Uses the final model trained on all folds.
    Falls back to best fold model if final not available.
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {model_name} on {test_split}")
    print(f"{'='*60}")

    model_dir = os.path.join(MODELS_DIR, model_name)

    # Load fold results to get norm stats
    # Use fold 1 norm stats as approximation for final model
    # (ideally final model would have its own norm stats)
    results_path = os.path.join(model_dir, "fold_results.npy")
    if not os.path.exists(results_path):
        print(f"  ❌ No results found for {model_name}")
        return None

    results = list(np.load(results_path, allow_pickle=True))

    # Load final norm stats saved by retrain.py
    norm_path = os.path.join(model_dir, "final_norm_stats.npy")
    if os.path.exists(norm_path):
        norm      = np.load(norm_path, allow_pickle=True).item()
        mfcc_mean = norm['mfcc_mean']
        mfcc_std  = norm['mfcc_std']
        print(f"  Using: final_norm_stats.npy (MFCC only -- F0 is computed fresh below)")
    else:
        print(f"  ⚠️  No final norm stats — averaging CV fold stats")
        mfcc_means = np.stack([r['norm']['mfcc_mean'] for r in results])
        mfcc_stds  = np.stack([r['norm']['mfcc_std']  for r in results])
        mfcc_mean  = mfcc_means.mean(axis=0)
        mfcc_std   = mfcc_stds.mean(axis=0)

    # Load model — try final model first, fall back to fold 1
    final_path = os.path.join(model_dir, "final_model.pt")
    fold1_path = os.path.join(model_dir, "best_fold1.pt")

    if os.path.exists(final_path):
        model_path = final_path
        print(f"  Using: final_model.pt")
    elif os.path.exists(fold1_path):
        model_path = fold1_path
        print(f"  ⚠️  No final model — using best_fold1.pt")
        print(f"  Note: Run retrain.py after all CV folds done")
    else:
        print(f"  ❌ No model file found")
        return None

    model = model_class(dropout=0.0).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Get test files
    test_files = get_test_files(test_split)
    print(f"  Test files: {len(test_files)}")

    # F0 is normalized per-speaker using the TEST set's own speakers --
    # they were never seen during training, so any train-derived F0
    # mean/std would be normalizing against the wrong population's pitch
    # range entirely. This only reads raw F0 values, never labels, so
    # there's no leakage in computing it here.
    print("  Computing per-speaker F0 stats for test set...")
    test_f0_speaker_stats, test_f0_fallback = compute_per_speaker_f0_stats(test_files)

    # Normalize tensors
    mfcc_mean_t = torch.FloatTensor(mfcc_mean).to(device)
    mfcc_std_t  = torch.FloatTensor(mfcc_std).to(device)

    # Soft voting — clip level
    clip_probs  = {}
    clip_labels = {}
    clip_wins   = {}   # per-window probs for display

    with torch.no_grad():
        for path in tqdm(test_files, desc="Evaluating", leave=False):
            try:
                f0_mean, f0_std = lookup_f0_stats(
                    path, test_f0_speaker_stats, test_f0_fallback
                )
                f0_mean_t = torch.FloatTensor(f0_mean).to(device)
                f0_std_t  = torch.FloatTensor(f0_std).to(device)

                d  = np.load(path, allow_pickle=True)
                n  = int(d['n_windows'])
                lb = int(d['label'])

                lm = torch.FloatTensor(
                    d['logmel']).unsqueeze(1).to(device)
                mc = (torch.FloatTensor(
                    d['mfcc']).to(device) - mfcc_mean_t
                    ) / mfcc_std_t
                f0 = (torch.FloatTensor(
                    d['f0']).to(device) - f0_mean_t
                    ) / f0_std_t
                d.close()

                logits = model(lm, mc, f0)
                probs  = torch.sigmoid(logits).cpu().numpy().flatten()

                clip_probs[path]  = probs.tolist()
                clip_labels[path] = lb
                clip_wins[path]   = probs.tolist()

            except Exception as e:
                print(f"⚠️  Error processing {os.path.basename(path)}: {e}")

    # Aggregate
    final_preds   = []
    final_labels  = []
    final_probs   = []   # mean prob for AUC

    for path in clip_probs:
        mean_prob = np.mean(clip_probs[path])
        final_preds.append(1 if mean_prob >= 0.5 else 0)
        final_labels.append(clip_labels[path])
        final_probs.append(mean_prob)

    preds  = np.array(final_preds)
    labels = np.array(final_labels)
    probs  = np.array(final_probs)

    # Metrics
    acc   = np.mean(preds == labels)
    f1    = f1_score(labels, preds, zero_division=0)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    auc   = roc_auc_score(labels, probs)
    cm    = confusion_matrix(labels, preds)

    # EER
    from scipy.interpolate import interp1d
    thresholds = np.linspace(0, 1, 1000)
    fars = []
    frrs = []
    for t in thresholds:
        p   = (probs >= t).astype(int)
        tp  = np.sum((p == 1) & (labels == 1))
        fp  = np.sum((p == 1) & (labels == 0))
        fn  = np.sum((p == 0) & (labels == 1))
        tn  = np.sum((p == 0) & (labels == 0))
        far = fp / (fp + tn + 1e-8)
        frr = fn / (fn + tp + 1e-8)
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)
    diff = np.abs(fars - frrs)
    eer  = (fars[np.argmin(diff)] +
            frrs[np.argmin(diff)]) / 2

    print(f"\n  📊 {model_name} — {test_split} Results:")
    print(f"     Total clips : {len(clip_probs)}")
    print(f"     Accuracy    : {acc:.4f}")
    print(f"     F1-score    : {f1:.4f}")
    print(f"     Precision   : {prec:.4f}")
    print(f"     Recall      : {rec:.4f}")
    print(f"     ROC-AUC     : {auc:.4f}")
    print(f"     EER         : {eer:.4f}")
    print(f"\n     Confusion Matrix:")
    print(f"       TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"       FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

    # Sample per-window display (first 3 clips)
    print(f"\n     Sample per-window probabilities:")
    shown = 0
    for path, win_probs in clip_wins.items():
        if shown >= 3:
            break
        mean_p   = np.mean(win_probs)
        decision = "AI-Gen" if mean_p >= 0.5 else "Real  "
        true_lb  = "AI-Gen" if clip_labels[path] == 1 else "Real  "
        correct  = "✅" if (mean_p >= 0.5) == (
            clip_labels[path] == 1) else "❌"
        win_str  = " | ".join([f"{p:.2f}" for p in win_probs])
        print(f"       {os.path.basename(path)[:30]}")
        print(f"         Windows : [{win_str}]")
        print(f"         Mean    : {mean_p:.4f} → "
              f"{decision} (True: {true_lb}) {correct}")
        shown += 1

    result = {
        'model'    : model_name,
        'split'    : test_split,
        'accuracy' : acc,
        'f1'       : f1,
        'precision': prec,
        'recall'   : rec,
        'auc'      : auc,
        'eer'      : eer,
        'cm'       : cm
    }

    return result


if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    set_seed(SEED)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["CNN-only", "CNN-GRU", "CNN-GRU-F", "all"],
        default="all"
    )
    args = parser.parse_args()

    all_results = []

    if args.model == "all":
        model_list = list(MODELS.items())
    else:
        model_list = [(args.model, MODELS[args.model])]

    for model_name, model_class in model_list:
        # Clean test set
        r_clean = evaluate_on_test(
            model_name, model_class, "test")
        if r_clean:
            all_results.append(r_clean)

        # Degraded test set
        r_deg = evaluate_on_test(
            model_name, model_class, "test_degraded")
        if r_deg:
            all_results.append(r_deg)

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Model':<12} {'Split':<16} {'Acc':>7} "
              f"{'F1':>7} {'AUC':>7} {'EER':>7}")
        print(f"  {'-'*60}")
        for r in all_results:
            print(f"  {r['model']:<12} {r['split']:<16} "
                  f"{r['accuracy']:>7.4f} {r['f1']:>7.4f} "
                  f"{r['auc']:>7.4f} {r['eer']:>7.4f}")

    # Save
    np.save(
        os.path.join(RESULTS_DIR, "test_results.npy"),
        all_results
    )
    print(f"\n✅ Results saved to {RESULTS_DIR}")