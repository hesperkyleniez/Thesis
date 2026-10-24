# evaluate.py — Test set evaluation with soft voting
import os
import sys
import glob
import csv
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

# The running script directory is already on sys.path; do not inject historical code paths.
from improvedCQCC_config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, SEED
from improvedCQCC_train import (
    CNNOnly, CNNGRUOnly, CNNGRUFusion, set_seed,
    compute_per_speaker_f0_stats, lookup_f0_stats,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "CNN-only"  : CNNOnly,
    "CNN-GRU"   : CNNGRUOnly,
    "CNN-GRU-F" : CNNGRUFusion,
}

# Names are the condition labels saved inside each feature .npz file.
# Read that saved label; never infer degradation type from a filename.
DEGRADED_CONDITIONS = (
    "telephone", "mobile_gaussian", "room_gaussian",
    "room_urban", "mobile_urban",
)


def condition_metrics(labels, probs, threshold=0.5):
    """Recording-level metrics, using the SAME rules as the original evaluation.

    EER deliberately retains the historical 1,000-threshold calculation so
    condition rows are comparable with the previously reported overall EERs.
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if labels.size == 0:
        raise ValueError("Cannot compute metrics for an empty condition.")
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan")
    if len(np.unique(labels)) == 2:
        thresholds = np.linspace(0, 1, 1000)
        fars, frrs = [], []
        for t in thresholds:
            p = (probs >= t).astype(int)
            tp_t = np.sum((p == 1) & (labels == 1))
            fp_t = np.sum((p == 1) & (labels == 0))
            fn_t = np.sum((p == 0) & (labels == 1))
            tn_t = np.sum((p == 0) & (labels == 0))
            fars.append(fp_t / (fp_t + tn_t + 1e-8))
            frrs.append(fn_t / (fn_t + tp_t + 1e-8))
        fars, frrs = np.asarray(fars), np.asarray(frrs)
        ix = int(np.argmin(np.abs(fars - frrs)))
        eer = float((fars[ix] + frrs[ix]) / 2)
    else:
        eer = float("nan")
    return dict(
        n=int(len(labels)),
        accuracy=float(np.mean(preds == labels)),
        f1=float(f1_score(labels, preds, zero_division=0)),
        precision=float(precision_score(labels, preds, zero_division=0)),
        recall=float(recall_score(labels, preds, zero_division=0)),
        auc=auc, eer=eer,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        threshold=float(threshold),
    )


def write_csv(path, records, columns):
    """Write a portable research table (Excel/Google Sheets-compatible CSV)."""
    with open(path, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def get_test_files(split="test"):
    """Get all NPZ files from test or test_degraded split."""
    return sorted(glob.glob(
        os.path.join(FEATURES_DIR, split, "**", "*.npz"),
        recursive=True
    ))

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
    uses_aux = bool(getattr(model_class, "uses_aux_features", False))

    # Load fold results to get norm stats
    # Use fold 1 norm stats as approximation for final model
    # (ideally final model would have its own norm stats)
    results_path = os.path.join(model_dir, "fold_results.npy")
    if not os.path.exists(results_path):
        print(f"  ❌ No results found for {model_name}")
        return None

    results = list(np.load(results_path, allow_pickle=True))

    # Auxiliary normalization is required only by CNN-GRU-F.
    if uses_aux:
        norm_path = os.path.join(model_dir, "final_norm_stats.npy")
        if os.path.exists(norm_path):
            norm = np.load(norm_path, allow_pickle=True).item()
            mfcc_mean = norm['mfcc_mean']
            mfcc_std = norm['mfcc_std']
            print("  Using: final_norm_stats.npy")
        else:
            print("  No final norm stats — averaging CV fold MFCC stats")
            mfcc_means = np.stack([r['norm']['mfcc_mean'] for r in results])
            mfcc_stds = np.stack([r['norm']['mfcc_std'] for r in results])
            mfcc_mean = mfcc_means.mean(axis=0)
            mfcc_std = mfcc_stds.mean(axis=0)
    else:
        mfcc_mean = mfcc_std = None
        print("  Log-Mel-only model: MFCC/F0 normalization skipped")

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

    if uses_aux:
        print("  Computing per-speaker F0 stats for test set...")
        test_f0_speaker_stats, test_f0_fallback = compute_per_speaker_f0_stats(test_files)
        mfcc_mean_t = torch.FloatTensor(mfcc_mean).to(device)
        mfcc_std_t = torch.FloatTensor(mfcc_std).to(device)
    else:
        test_f0_speaker_stats = {}
        test_f0_fallback = None

    # Soft voting — clip level
    clip_probs  = {}
    clip_labels = {}
    clip_conditions = {}
    clip_speakers = {}
    clip_wins   = {}   # per-window probs for display
    failed = []

    with torch.no_grad():
        for path in tqdm(test_files, desc="Evaluating", leave=False):
            try:
                with np.load(path, allow_pickle=True) as d:
                    lb = int(d['label'])
                    # Saved metadata from the frozen test/degraded feature cache.
                    # Do not infer the condition from a filename or assign it anew.
                    if 'condition' not in d.files:
                        raise KeyError("Missing 'condition' in feature cache")
                    condition = str(d['condition'].item())
                    allowed = ('clean',) if test_split == 'test' else DEGRADED_CONDITIONS
                    if condition not in allowed:
                        raise ValueError(f"Unexpected condition {condition!r} for {test_split}")
                    speaker = str(d['speaker'].item())
                    lm = torch.FloatTensor(d['logmel']).unsqueeze(1).to(device)
                    if uses_aux:
                        f0_mean, f0_std = lookup_f0_stats(path, test_f0_speaker_stats, test_f0_fallback)
                        f0_mean_t = torch.FloatTensor(f0_mean).to(device)
                        f0_std_t = torch.FloatTensor(f0_std).to(device)
                        mc = (torch.FloatTensor(d['mfcc']).to(device) - mfcc_mean_t) / mfcc_std_t
                        f0 = (torch.FloatTensor(d['f0']).to(device) - f0_mean_t) / f0_std_t

                logits = model(lm, mc, f0) if uses_aux else model(lm)
                probs  = torch.sigmoid(logits).cpu().numpy().flatten()

                clip_probs[path]  = probs.tolist()
                clip_labels[path] = lb
                clip_conditions[path] = condition
                clip_speakers[path] = speaker
                clip_wins[path]   = probs.tolist()

            except Exception as e:
                failed.append((path, str(e)))
                print(f"⚠️  Error processing {os.path.basename(path)}: {e}")

    # Never silently produce incomplete publication metrics.
    if failed or len(clip_probs) != len(test_files):
        raise RuntimeError(
            f"Incomplete evaluation of {model_name} / {test_split}: "
            f"processed {len(clip_probs)}/{len(test_files)}. "
            f"First error: {failed[0] if failed else 'unknown'}. "
            "No partial condition-wise metrics will be saved."
        )
    if not clip_probs:
        raise RuntimeError(f"No recordings found in split {test_split!r}")

    # Aggregate
    final_preds   = []
    final_labels  = []
    final_probs   = []   # mean prob for AUC

    clip_paths = list(clip_probs)
    for path in clip_paths:
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

    # Conditions are evaluated FROM THE SAME predictions as overall degraded.
    # No inference is repeated and the overall score is never a mean of scores.
    condition_results = []
    if test_split == 'test_degraded':
        found_counts = Counter(clip_conditions.values())
        unexpected = set(found_counts) - set(DEGRADED_CONDITIONS)
        missing = set(DEGRADED_CONDITIONS) - set(found_counts)
        if unexpected or missing:
            raise RuntimeError(
                f"Unexpected/missing degraded groups: unexpected={unexpected}, missing={missing}; "
                f"distribution={dict(found_counts)}"
            )
        for condition in DEGRADED_CONDITIONS:
            mask = np.asarray([clip_conditions[p] == condition for p in clip_paths])
            row = condition_metrics(labels[mask], probs[mask])
            row.update(model=model_name, split=test_split, condition=condition)
            condition_results.append(row)
        assert sum(row['n'] for row in condition_results) == len(clip_paths)
        print(f"\n  CONDITION-WISE DEGRADED RESULTS: {model_name}")
        print(f"  {'Condition':<18} {'n':>6} {'Acc':>8} {'Prec':>8} "
              f"{'Rec':>8} {'F1':>8} {'AUC':>8} {'EER':>8}")
        print('  ' + '-'*80)
        for row in condition_results:
            print(f"  {row['condition']:<18} {row['n']:>6,d} "
                  f"{row['accuracy']:>8.4f} {row['precision']:>8.4f} "
                  f"{row['recall']:>8.4f} {row['f1']:>8.4f} "
                  f"{row['auc']:>8.4f} {row['eer']:>8.4f}")
        print('  ' + '-'*80)
        print(f"  {'OVERALL DEGRADED':<18} {len(clip_paths):>6,d} "
              f"{acc:>8.4f} {prec:>8.4f} {rec:>8.4f} "
              f"{f1:>8.4f} {auc:>8.4f} {eer:>8.4f}")
        print('  Note: overall metrics use all degraded recordings; NOT the mean of condition scores.')
        print('  Condition confusion matrices (TN, FP, FN, TP):')
        for row in condition_results:
            print(f"    {row['condition']:<18} ({row['tn']}, {row['fp']}, "
                  f"{row['fn']}, {row['tp']})")

    # A flat, reusable per-recording file also permits later paired model
    # comparisons without re-running network inference.
    clip_rows = [dict(
        model=model_name, split=test_split,
        condition=clip_conditions[p], speaker=clip_speakers[p],
        clip_id=os.path.relpath(p, os.path.join(FEATURES_DIR, test_split)).replace('\\', '/'),
        true_label=int(clip_labels[p]),
        predicted_label=int(final_preds[i]),
        probability=float(final_probs[i]),
        threshold=0.5, n_windows=len(clip_probs[p]),
    ) for i, p in enumerate(clip_paths)]

    result = {
        'model'    : model_name,
        'split'    : test_split,
        'accuracy' : acc,
        'f1'       : f1,
        'precision': prec,
        'recall'   : rec,
        'auc'      : auc,
        'eer'      : eer,
        'cm'       : cm,
        'condition_results': condition_results,
        '_clip_rows': clip_rows,
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
    all_condition_rows = []
    all_recording_rows = []

    if args.model == "all":
        model_list = list(MODELS.items())
    else:
        model_list = [(args.model, MODELS[args.model])]

    for model_name, model_class in model_list:
        # Clean test set
        r_clean = evaluate_on_test(
            model_name, model_class, "test")
        if r_clean:
            all_recording_rows.extend(r_clean.pop('_clip_rows'))
            all_results.append(r_clean)

        # Degraded test set
        r_deg = evaluate_on_test(
            model_name, model_class, "test_degraded")
        if r_deg:
            all_recording_rows.extend(r_deg.pop('_clip_rows'))
            all_condition_rows.extend(r_deg['condition_results'])
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

    # New publication-ready CSV outputs; no changes to checkpoints or caches.
    # CSV files only contain the selected model(s) from this run.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    overall_csv = os.path.join(RESULTS_DIR, 'baseline_overall_results.csv')
    condition_csv = os.path.join(RESULTS_DIR, 'baseline_degraded_by_condition.csv')
    recording_csv = os.path.join(RESULTS_DIR, 'baseline_recording_predictions.csv')
    if all_results:
        overall_rows = []
        for r in all_results:
            tn, fp, fn, tp = np.asarray(r['cm']).ravel().tolist()
            overall_rows.append(dict(
                model=r['model'], split=r['split'], n=int(np.asarray(r['cm']).sum()),
                accuracy=r['accuracy'], precision=r['precision'], recall=r['recall'],
                f1=r['f1'], auc=r['auc'], eer=r['eer'],
                tn=tn, fp=fp, fn=fn, tp=tp, threshold=0.5,
            ))
        write_csv(overall_csv, overall_rows, [
            'model', 'split', 'n', 'accuracy', 'precision', 'recall',
            'f1', 'auc', 'eer', 'tn', 'fp', 'fn', 'tp', 'threshold',
        ])
        print(f"  Overall metrics CSV: {overall_csv}")
    if all_condition_rows:
        write_csv(condition_csv, all_condition_rows, [
            'model', 'split', 'condition', 'n',
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'eer',
            'tn', 'fp', 'fn', 'tp', 'threshold',
        ])
        print(f"  Condition-wise CSV: {condition_csv}")
    if all_recording_rows:
        write_csv(recording_csv, all_recording_rows, [
            'model', 'split', 'condition', 'speaker', 'clip_id',
            'true_label', 'predicted_label', 'probability',
            'threshold', 'n_windows',
        ])
        print(f"  Per-recording CSV: {recording_csv}")

    # Preserve existing overall test-results artifact and metric definitions.
    np.save(
        os.path.join(RESULTS_DIR, "test_results.npy"),
        all_results
    )
    print(f"\n✅ Results saved to {RESULTS_DIR}")