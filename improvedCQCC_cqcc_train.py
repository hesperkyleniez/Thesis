"""Development-only MLP retraining for paper-inspired Akıncı–Özbay CQCC.

Reuses the exact v25 auxiliary-expert architecture, per-window training,
clip-level AUC checkpoint selection, speaker-grouped 4-fold validation,
and train-only normalization. Reads ONLY the separately generated CQCC
sidecars. Writes ONLY models/CQCC-Akinci-Ozbay-DEV.

    python train_akinci_cqcc_expert.py --stage check
    python train_akinci_cqcc_expert.py --stage cv
    python train_akinci_cqcc_expert.py --stage compare

If four-fold CV proves worthwhile, optionally:
    python train_akinci_cqcc_expert.py --stage final

Not fusion training. Never evaluates held-out test.
"""

import argparse
import glob
import hashlib
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, log_loss, roc_auc_score, roc_curve,
                             precision_score, recall_score, confusion_matrix)

from improvedCQCC_config import (FEATURES_DIR as ORIGINAL_FEATURES_DIR, BASE_DIR, MODELS_DIR, FOLD_ASSIGNMENTS,
                    LEARNING_RATE, WEIGHT_DECAY, SEED, LR_FACTOR,
                    MAX_TRAIN_WINDOWS_PER_FILE, RETRAIN_EPOCH_POLICY,
                    CQCC_HIDDEN, CQCC_DROPOUT, CQCC_MAX_EPOCHS,
                    F0_HIDDEN, F0_DROPOUT, F0_MAX_EPOCHS,
                    CQCC_VECTOR_DIM, F0_VECTOR_DIM)


# Explicit separate development-only sidecar (NOT E:\Thesis\features\train).
FEATURES_DIR = os.path.join(BASE_DIR, 'features_experiments', 'akinci_ozbay_cqcc')
ORIGINAL_TRAIN = os.path.join(ORIGINAL_FEATURES_DIR, 'train')
EXPECTED_CACHE_VERSION = 'akinci_ozbay_2026_cqcc_dev_sidecar_v1'

VERSION = 'akinci_ozbay_2026_cqcc_mlp_cv_v1'
MIN_EPOCHS = 15                 # At least this many epochs before early-stop.
PATIENCE = 10                   # Patience measured against clip-level ROC-AUC.
AUC_MIN_DELTA = 0.0001         # Tiny AUC changes count as meaningful.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CQCCMLP(nn.Module):
    """Exact CQCC MLP architecture from the existing CQCC-v25-AUC expert."""
    def __init__(self, input_dim, hidden=CQCC_HIDDEN, dropout=CQCC_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden//2,16)),
            nn.GELU(),
            nn.Dropout(dropout*0.6),
            nn.Linear(max(hidden//2,16),1),
        )
    def forward(self,x):
        return self.net(x)


class CQCCDataset(Dataset):
    """Exact previous CQCC window selection/standardization; strict input checks."""
    def __init__(self, files, feature_name, mean, std, max_windows=None, seed=SEED):
        self.x=[];self.y=[];self.paths=[]
        total_seen=total_kept=0
        mean=np.asarray(mean,np.float32);std=np.asarray(std,np.float32)
        for path in tqdm(files,desc=f'Loading {feature_name}',leave=False):
            with np.load(path,allow_pickle=False) as d:
                arr=np.asarray(d[feature_name],np.float32)
                label=int(d['label']);n=len(arr)
                if arr.ndim!=2 or arr.shape[1]!=40 or not np.isfinite(arr).all():
                    raise RuntimeError(f'Bad CQCC training features: {path}')
                if max_windows is not None and n>max_windows:
                    idx=np.linspace(0,n-1,max_windows,dtype=np.int64)
                else:idx=np.arange(n)
                arr=(arr[idx]-mean)/std
                self.x.append(arr);self.y.extend([label]*len(arr))
                self.paths.extend([path]*len(arr))
                total_seen+=n;total_kept+=len(arr)
        if not self.x:raise RuntimeError('No CQCC features loaded')
        self.x=torch.as_tensor(np.concatenate(self.x,axis=0),dtype=torch.float32)
        self.y=torch.as_tensor(self.y,dtype=torch.float32).unsqueeze(1)
        print(f'  {feature_name}: kept {total_kept:,}/{total_seen:,} windows ({100*total_kept/max(total_seen,1):.1f}%)')
        print(f'  {feature_name}: {len(self.y):,} windows loaded into RAM')
    def __len__(self):return len(self.y)
    def __getitem__(self,idx):return self.x[idx],self.y[idx],self.paths[idx]


def cqcc_stats(files,feature_name):
    """Fold train-only statistics; consistent with the original v25 implementation."""
    arrays=[]
    for path in tqdm(files,desc=f'Stats {feature_name}',leave=False):
        with np.load(path,allow_pickle=False) as d:
            arrays.append(np.asarray(d[feature_name],np.float32))
    if not arrays:raise RuntimeError('No features for normalization')
    cat=np.concatenate(arrays,axis=0)
    return cat.mean(axis=0),cat.std(axis=0)+1e-8


EXPERTS = {
    'cqcc': dict(name='CQCC', folder='CQCC-Akinci-Ozbay-DEV', dim=CQCC_VECTOR_DIM,
                 hidden=CQCC_HIDDEN, dropout=CQCC_DROPOUT,
                 epochs=CQCC_MAX_EPOCHS, batch_size=128,
                 model=CQCCMLP, dataset=CQCCDataset),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def all_files():
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, 'train', '**', '*.npz'), recursive=True))
    if not files:
        raise RuntimeError(f'No development cache files in {FEATURES_DIR}\\train')
    speakers = Counter(os.path.basename(os.path.dirname(p)) for p in files)
    unknown = sorted(set(speakers) - set(FOLD_ASSIGNMENTS))
    if unknown:
        raise RuntimeError(f'Unassigned speakers found: {unknown}. Stopping to avoid dropping samples.')
    print(f'Development cache: {len(files):,} clips, {len(speakers)} speakers')
    return files


def require_complete_development_sidecar(files):
    """Don't silently train on an incomplete subset or any held-out recording."""
    originals = sorted(glob.glob(os.path.join(ORIGINAL_TRAIN, '**', '*.npz'), recursive=True))
    if not originals:
        raise RuntimeError(f'Original development cache not found: {ORIGINAL_TRAIN}')
    orig_rel = {os.path.relpath(f, ORIGINAL_TRAIN) for f in originals}
    side_rel = {os.path.relpath(f, os.path.join(FEATURES_DIR, 'train')) for f in files}
    if orig_rel != side_rel:
        missing = sorted(orig_rel - side_rel)
        extra = sorted(side_rel - orig_rel)
        raise RuntimeError(f'Incomplete/mismatched experimental sidecar: missing={len(missing)}, extra={len(extra)}; '
                           'run: python prepare_akinci_cqcc_dev.py --stage build; then --stage verify')
    for f in files:
        sp = os.path.basename(os.path.dirname(f))
        if sp not in FOLD_ASSIGNMENTS:
            raise RuntimeError(f'Unknown speaker in experimental feature cache: {sp}')
    # Check cache provenance/metadata for every file without reading full feature arrays.
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            if str(d.get('cache_version', '')) != EXPECTED_CACHE_VERSION:
                raise RuntimeError(f'Unexpected sidecar version: {f}')
            if str(d['speaker']) != os.path.basename(os.path.dirname(f)):
                raise RuntimeError(f'Speaker mismatch: {f}')
    print(f'PASS: complete development-only paper-inspired CQCC cache: {len(files):,} clips')


def compare_with_existing_v25():
    new_path = os.path.join(MODELS_DIR, EXPERTS['cqcc']['folder'], 'fold_results.npy')
    old_path = os.path.join(MODELS_DIR, 'CQCC-v25-AUC', 'fold_results.npy')
    if not os.path.isfile(new_path):
        raise FileNotFoundError(f'Run new four-fold CV first: {new_path}')
    if not os.path.isfile(old_path):
        print(f'Original v25 fold results not present: {old_path}. New fold results are still saved.')
        return
    new = list(np.load(new_path, allow_pickle=True))
    old = list(np.load(old_path, allow_pickle=True))
    if len(new) != 4 or len(old) != 4:
        raise RuntimeError('Expected exactly four CV results per expert')
    by_fold = {int(r['fold']): r for r in old}
    import csv
    folder=os.path.join(MODELS_DIR, EXPERTS['cqcc']['folder'])
    output=os.path.join(folder,'dev_fold_comparison_with_current_v25.csv')
    rows=[]
    print('\nDEVELOPMENT CV COMPARISON — no held-out test')
    print(' Fold   Current AUC   Paper CQCC AUC   Delta AUC    Current F1    Paper CQCC F1')
    for n in new:
        f=int(n['fold'])
        if f not in by_fold: raise RuntimeError('Fold mismatch')
        o=by_fold[f]
        # Current v25 select by clip AUC using identical speaker CV, provided versions match.
        a=float(o['best_stats']['auc']); b=float(n['best_stats']['auc'])
        fa=float(o['best_stats']['f1_at_0_5']); fb=float(n['best_stats']['f1_at_0_5'])
        rows.append({'fold':f,'current_auc':a,'paper_auc':b,'delta_auc':b-a,
                     'current_f1':fa,'paper_f1':fb,'delta_f1':fb-fa,
                     'current_n_val':int(o.get('val_clips',0)), 'paper_n_val':int(n.get('val_clips',0))})
        if rows[-1]['current_n_val'] != rows[-1]['paper_n_val']:
            raise RuntimeError('Validation counts disagree: comparisons may not be paired')
        print(f'   {f}      {a:7.4f}          {b:7.4f}       {b-a:+.4f}         {fa:.4f}         {fb:.4f}')
    print(f' Mean delta AUC = {np.mean([r["delta_auc"] for r in rows]):+.4f}')
    print(f' Mean delta F1  = {np.mean([r["delta_f1"] for r in rows]):+.4f}')
    print('NOTE: Descriptive fold results only; four folds do not establish significance.')
    if os.path.isfile(output):
        raise RuntimeError(f'Existing comparison report would be overwritten: {output}')
    with open(output,'x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Saved NEW fold comparison CSV:',output)


def preflight(files, feature, expected_dim):
    """No cache rewrite, only check a few files for shape/finiteness."""
    rng = np.random.default_rng(SEED)
    indices = sorted(set([0, len(files)-1] + rng.choice(len(files), min(100, len(files)), replace=False).tolist()))
    versions = Counter()
    for index in indices:
        with np.load(files[index], allow_pickle=False) as d:
            if feature not in d.files:
                raise RuntimeError(f'{files[index]} missing {feature}')
            arr = d[feature]
            if arr.ndim != 2 or arr.shape[1] != expected_dim or len(arr) == 0:
                raise RuntimeError(f'{files[index]} has {feature} shape {arr.shape}; expected (N,{expected_dim})')
            if not np.isfinite(arr).all():
                raise RuntimeError(f'{files[index]} contains NaN/Inf {feature} values')
            if 'cache_version' in d.files:
                versions[str(d['cache_version'])] += 1
    print(f'  Sampled {len(indices)} {feature} caches: dimension={expected_dim}, finite=True')
    print(f'  Cache versions in sample: {dict(versions)}')
    if len(versions) > 1:
        raise RuntimeError('Mixed cache versions detected. Freeze preprocessing before training.')


def prepare_loader(files, spec, mean, std, training, seed):
    ds = spec['dataset'](files, spec_key(spec), mean, std,
                         max_windows=MAX_TRAIN_WINDOWS_PER_FILE if training else None,
                         seed=seed)
    if not torch.isfinite(ds.x).all():
        raise RuntimeError('Nonfinite normalized auxiliary features detected.')
    from torch.utils.data import DataLoader
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=spec['batch_size'], shuffle=training,
                        num_workers=0, pin_memory=DEVICE.type == 'cuda',
                        generator=generator if training else None)
    return ds, loader


def spec_key(spec):
    return spec['name'].lower()


def make_model(spec):
    return spec['model'](spec['dim'], spec['hidden'], spec['dropout']).to(DEVICE)


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    weighted_loss = 0.0
    total = 0
    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        weighted_loss += float(loss.detach()) * len(x)
        total += len(x)
    return weighted_loss / max(total, 1)


def summarize_scores(y, scores):
    """Clip metrics, 0.5 decision threshold; no threshold fitting on validation."""
    y=np.asarray(y,int);scores=np.asarray(scores,float)
    pred=(scores>=0.5).astype(int)
    if len(np.unique(y))<2:
        return dict(n=len(y),auc=None,eer=None,accuracy_at_0_5=float(accuracy_score(y,pred)),
                    precision_at_0_5=float(precision_score(y,pred,zero_division=0)),
                    recall_at_0_5=float(recall_score(y,pred,zero_division=0)),
                    f1_at_0_5=float(f1_score(y,pred,zero_division=0)))
    fpr,tpr,_=roc_curve(y,scores)
    fnr=1-tpr
    k=int(np.argmin(np.abs(fpr-fnr)))
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),auc=float(roc_auc_score(y,scores)),
                eer=float((fpr[k]+fnr[k])/2),
                accuracy_at_0_5=float(accuracy_score(y,pred)),
                precision_at_0_5=float(precision_score(y,pred,zero_division=0)),
                recall_at_0_5=float(recall_score(y,pred,zero_division=0)),
                f1_at_0_5=float(f1_score(y,pred,zero_division=0)),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def validate(model, loader, cond_by_file):
    """Mean window probabilities per clip, matching original v25 expert validation."""
    model.eval()
    score_lists={};labels={}
    with torch.inference_mode():
        for x,y,paths in loader:
            logits=model(x.to(DEVICE,non_blocking=True))
            probabilities=torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
            targets=y.numpy().astype(int).reshape(-1)
            for path,prob,label in zip(paths,probabilities,targets):
                score_lists.setdefault(path,[]).append(float(prob))
                labels[path]=int(label)
    paths=sorted(score_lists)
    target=np.asarray([labels[path] for path in paths],dtype=int)
    scores=np.asarray([np.mean(score_lists[path]) for path in paths],dtype=float)
    if len(np.unique(target))<2:
        raise RuntimeError('Validation is missing one class')
    preds=(scores>=0.5).astype(int)
    condition_metrics={}
    for condition in sorted(set(cond_by_file[p] for p in paths)):
        mask=np.asarray([cond_by_file[path]==condition for path in paths])
        condition_metrics[condition]=summarize_scores(target[mask],scores[mask])
        condition_metrics[condition]['count']=int(mask.sum())
        condition_metrics[condition]['ai_prediction_rate']=float(np.mean(preds[mask]))
    summary=summarize_scores(target,scores)
    return {
        'auc':summary['auc'], 'eer':summary['eer'],
        'clip_log_loss':float(log_loss(target,np.clip(scores,1e-7,1-1e-7),labels=[0,1])),
        'f1_at_0_5':summary['f1_at_0_5'],
        'accuracy_at_0_5':summary['accuracy_at_0_5'],
        'precision_at_0_5':summary['precision_at_0_5'],
        'recall_at_0_5':summary['recall_at_0_5'],
        'confusion_matrix':{k:summary[k] for k in ('tn','fp','fn','tp')},
        'balanced_acc_at_0_5':float(balanced_accuracy_score(target,preds)),
        'ai_prediction_rate':float(np.mean(preds)),
        'ai_ground_truth_rate':float(np.mean(target)),
        'p05':float(np.quantile(scores,.05)),
        'p50':float(np.quantile(scores,.50)),
        'p95':float(np.quantile(scores,.95)),
        'conditions':condition_metrics,
    }



def read_conditions(files):
    result = {}
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            result[path] = str(d['condition'].item()) if 'condition' in d.files else 'clean'
    return result


def fold_paths(files, fold):
    tr = [p for p in files if FOLD_ASSIGNMENTS[os.path.basename(os.path.dirname(p))] != fold]
    va = [p for p in files if FOLD_ASSIGNMENTS[os.path.basename(os.path.dirname(p))] == fold]
    train_speakers = {os.path.basename(os.path.dirname(p)) for p in tr}
    val_speakers = {os.path.basename(os.path.dirname(p)) for p in va}
    if not tr or not va or (train_speakers & val_speakers):
        raise RuntimeError(f'Fold {fold} is invalid or overlaps speakers')
    return tr, va


def run_cv(expert, files, force=False):
    spec = EXPERTS[expert]
    directory = os.path.join(MODELS_DIR, spec['folder'])
    os.makedirs(directory, exist_ok=True)
    result_path = os.path.join(directory, 'fold_results.npy')
    results = []
    if os.path.exists(result_path) and not force:
        old = list(np.load(result_path, allow_pickle=True))
        if old and all(r.get('training_version') == VERSION and r.get('expert') == expert for r in old):
            results = old
    finished = {int(r['fold']) for r in results
                if os.path.isfile(os.path.join(directory, f"best_fold{int(r['fold'])}.pt"))}
    for fold in range(1, 5):
        if fold in finished:
            print(f'{spec["name"]} fold {fold}: existing experimental checkpoint; skipping')
            continue
        seed_everything(SEED + fold)
        tr, va = fold_paths(files, fold)
        print(f'\n{spec["name"]} paper-inspired development fold {fold}: train={len(tr):,} val={len(va):,}')
        mean, std = cqcc_stats(tr, expert)
        if len(mean) != spec['dim'] or not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise RuntimeError(f'{expert} fold {fold}: invalid normalizer')
        train_ds, train_loader = prepare_loader(tr, spec, mean, std, True, SEED + fold)
        val_ds, val_loader = prepare_loader(va, spec, mean, std, False, SEED + fold)
        cond_by_file = read_conditions(va)
        model = make_model(spec)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=3, min_lr=1e-7)
        scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
        criterion = torch.nn.BCEWithLogitsLoss()
        best_auc = -1.0
        best_epoch = 0
        best_stats = None
        history = []
        checkpoint = os.path.join(directory, f'best_fold{fold}.pt')
        for epoch in range(1, spec['epochs']+1):
            start = time.perf_counter()
            loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            diag = validate(model, val_loader, cond_by_file)
            scheduler.step(diag['clip_log_loss'])
            improved = diag['auc'] > best_auc + AUC_MIN_DELTA
            if improved:
                best_auc, best_epoch, best_stats = diag['auc'], epoch, diag
                torch.save(model.state_dict(), checkpoint)
            history.append({'epoch': epoch, 'train_window_bce': loss,
                            'val': diag, 'lr': optimizer.param_groups[0]['lr']})
            print(f'  Epoch {epoch:02d}/{spec["epochs"]} TrainBCE={loss:.4f} '
                  f'ClipAUC={diag["auc"]:.4f} ClipLoss={diag["clip_log_loss"]:.4f} '
                  f'F1@.5={diag["f1_at_0_5"]:.4f} AI-pred={diag["ai_prediction_rate"]:.1%} '
                  f'best={best_epoch} time={time.perf_counter()-start:.1f}s')
            if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
                print(f'  AUC-based early stop: epoch={epoch}, best={best_epoch}')
                break
        result = {
            'training_version': VERSION, 'expert': expert, 'fold': fold,
            'best_epoch': best_epoch, 'best_auc': best_auc,
            'best_stats': best_stats, 'mean': np.asarray(mean, dtype=np.float32),
            'std': np.asarray(std, dtype=np.float32), 'history': history,
            'selection_metric': 'clip_auc', 'objective': 'window_bce',
            'train_clips': len(tr), 'val_clips': len(va),
        }
        results = [r for r in results if int(r['fold']) != fold] + [result]
        results.sort(key=lambda r: int(r['fold']))
        np.save(result_path, np.asarray(results, dtype=object), allow_pickle=True)
        print(f'  Saved {checkpoint} (AUC={best_auc:.4f}, best epoch={best_epoch})')
        del train_ds, val_ds, train_loader, val_loader, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'{spec["name"]}: CV best epochs: {[r["best_epoch"] for r in results]}')
    print(f'{spec["name"]}: CV selected AUCs: {[round(r["best_auc"], 4) for r in results]}')


def run_final(expert, files, force=False):
    spec = EXPERTS[expert]
    directory = os.path.join(MODELS_DIR, spec['folder'])
    result_path = os.path.join(directory, 'fold_results.npy')
    if not os.path.isfile(result_path):
        raise FileNotFoundError(f'Run CV first: {result_path}')
    results = list(np.load(result_path, allow_pickle=True))
    if len(results) != 4 or {int(r['fold']) for r in results} != {1,2,3,4} or any(r.get('training_version') != VERSION for r in results):
        raise RuntimeError('Incomplete or incompatible v25 CV results.')
    epochs = [int(r['best_epoch']) for r in results]
    if RETRAIN_EPOCH_POLICY == 'median':
        target = max(1, int(np.median(epochs)))
    elif RETRAIN_EPOCH_POLICY == 'mean':
        target = max(1, int(round(np.mean(epochs))))
    else:
        target = max(epochs)
    print(f'{spec["name"]}: CV best epochs={epochs}, final epoch budget={target} ({RETRAIN_EPOCH_POLICY})')
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, 'final_model.pt')
    stats_path = os.path.join(directory, 'final_norm_stats.npy')
    if os.path.exists(model_path) and os.path.exists(stats_path) and not force:
        old = np.load(stats_path, allow_pickle=True).item()
        if old.get('training_version') == VERSION:
            if os.path.getmtime(model_path) < os.path.getmtime(result_path):
                raise RuntimeError('CV results are newer than the saved final checkpoint. Refit final using --force.')
            print(f'{spec["name"]}: experimental final model exists; use --force to refit.')
            return
        raise RuntimeError('Existing final file belongs to another version; refusing overwrite without --force.')
    seed_everything(SEED)
    mean, std = cqcc_stats(files, expert)
    ds, loader = prepare_loader(files, spec, mean, std, True, SEED)
    model = make_model(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=3, min_lr=1e-7)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, target+1):
        start = time.perf_counter()
        loss = train_epoch(model, loader, optimizer, criterion, scaler)
        scheduler.step(loss)
        history.append({'epoch': epoch, 'window_bce': loss, 'lr': optimizer.param_groups[0]['lr']})
        print(f'{spec["name"]} final {epoch:02d}/{target} BCE={loss:.4f}, {time.perf_counter()-start:.1f}s')
    torch.save(model.state_dict(), model_path)
    np.save(stats_path, {'mean': np.asarray(mean, dtype=np.float32),
                         'std': np.asarray(std, dtype=np.float32),
                         'training_version': VERSION, 'feature': expert,
                         'objective': 'window_bce', 'checkpoint_selection': 'clip_auc',
                         'final_epochs': target}, allow_pickle=True)
    np.save(os.path.join(directory, 'final_training_history.npy'), history, allow_pickle=True)
    print(f'Saved experimental final model: {model_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert', choices=['cqcc'], default='cqcc')
    parser.add_argument('--stage', choices=['check','cv','compare','final'], default='check')
    parser.add_argument('--force', action='store_true', help='Overwrite experimental CQCC-Akinci-Ozbay-DEV checkpoints ONLY.')
    args = parser.parse_args()
    print(f'Paper-inspired CQCC expert, DEVELOPMENT only | device={DEVICE} | metric=clip AUC | minimum={MIN_EPOCHS}, patience={PATIENCE}')
    files = all_files()
    require_complete_development_sidecar(files)
    if args.stage == 'check':
        preflight(files, 'cqcc', CQCC_VECTOR_DIM)
        print('READY. Next: python train_akinci_cqcc_expert.py --stage cv')
    elif args.stage == 'cv':
        preflight(files, 'cqcc', CQCC_VECTOR_DIM)
        run_cv('cqcc', files, force=args.force)
    elif args.stage == 'compare':
        compare_with_existing_v25()
    elif args.stage == 'final':
        run_final('cqcc', files, force=args.force)



if __name__ == '__main__':
    main()
