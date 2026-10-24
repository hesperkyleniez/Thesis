"""Create a safe isolated training copy of the 16-file improvedCQCC project.

Intended for reproducing previously trained thesis models WITHOUT overwriting
original checkpoints or feature caches. Only the model/results output ROOT is
changed; the source WAVs, feature sidecars, and condition maps stay read-only
at the thesis base path. Does NOT run training itself.

Usage (from E:\\Thesis\\code_improved):
  python prepare_improvedCQCC_repro_trial.py setup --root E:\\Thesis --name trial01
  cd /d E:\\Thesis\\reproducibility\\trial01\\code
  python improvedCQCC_train.py --model CNN-only
  python E:\\Thesis\\code_improved\\prepare_improvedCQCC_repro_trial.py compare --root E:\\Thesis --name trial01 --model CNN-only
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

EXPECTED_FILES = [
    'improvedCQCC_augmentation.py', 'improvedCQCC_config.py',
    'improvedCQCC_cqcc_f0_fusion.py', 'improvedCQCC_cqcc_retrain.py',
    'improvedCQCC_cqcc_train.py', 'improvedCQCC_dataset.py',
    'improvedCQCC_evaluate.py', 'improvedCQCC_f0_retrain.py',
    'improvedCQCC_f0_train.py', 'improvedCQCC_features.py',
    'improvedCQCC_latency_test.py', 'improvedCQCC_prepare_dataset.py',
    'improvedCQCC_retrain.py', 'improvedCQCC_robustness.py',
    'improvedCQCC_run_degraded_test.py', 'improvedCQCC_train.py',
]
METRICS = ('accuracy','f1','precision','recall','auc','eer','best_epoch','best_condition_macro_f1')
MODELS=('CNN-only','CNN-GRU','CNN-GRU-F')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def verify_original(root):
    for base in ('features/train', 'models', 'results'):
        if not (root / base).exists():
            raise SystemExit(f'ERROR: Thesis root lacks {base}: {root}')
    if not list((root/'features/train').glob('**/*.npz')):
        raise SystemExit(f'ERROR: No cached training features under {root / "features/train"}')


def setup(root,name,source):
    verify_original(root)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',name):
        raise SystemExit('ERROR: trial name must consist of letters, digits, _ or - only')
    destination=root/'reproducibility'/name
    if destination.exists():
        raise SystemExit(f'ERROR: Trial already exists: {destination}\nUse a new name; existing runs are never overwritten.')
    src_files=list(source.glob('improvedCQCC_*.py'))
    names=sorted(x.name for x in src_files)
    if names!=sorted(EXPECTED_FILES):
        raise SystemExit(f'ERROR: Expected exactly the 16 improvedCQCC source files in {source}. Found {len(src_files)}: {names}')
    # Check output paths refer to config, not the original root embedded elsewhere.
    for name_check in ['improvedCQCC_train.py','improvedCQCC_retrain.py','improvedCQCC_cqcc_train.py','improvedCQCC_f0_train.py','improvedCQCC_cqcc_f0_fusion.py']:
        s=(source/name_check).read_text(encoding='utf8')
        if name_check=='improvedCQCC_train.py':
            if 'from improvedCQCC_config import (' not in s or 'MODELS_DIR' not in s:
                raise SystemExit('ERROR: Unexpected baseline trainer; refusing unsafe patch')
    cfg=source/'improvedCQCC_config.py'
    old=cfg.read_text(encoding='utf8')
    if 'BASE_DIR      = os.environ.get("THESIS_BASE_DIR", r"E:\\Thesis")' not in old:
        raise SystemExit('ERROR: Source config uses an unexpected dataset-root declaration')
    patterns={
        'MODELS_DIR': 'MODELS_DIR    = os.path.join(BASE_DIR, "reproducibility", '+repr(name)+', "models")',
        'RESULTS_DIR': 'RESULTS_DIR   = os.path.join(BASE_DIR, "reproducibility", '+repr(name)+', "results")',
    }
    for key, replacement in patterns.items():
        old,n=re.subn(r'(?m)^'+key+r'\s*=\s*os\.path\.join\(BASE_DIR, "(?:models|results)"\)\s*$',lambda m:replacement,old,count=1)
        if n!=1:raise SystemExit(f'ERROR: Could not safely isolate {key} in config')
    # We do not copy, rebase, or modify the original dataset and 2 condition maps.
    code=destination/'code'
    try:
        code.mkdir(parents=True, exist_ok=False)
        for src in src_files:
            output=code/src.name
            if src==cfg: output.write_text(old,encoding='utf8')
            else: shutil.copy2(src,output)
        notes={
            'purpose':'Isolated 4-fold retraining: original features read-only; outputs only in this trial',
            'thesis_root':str(root), 'original_source':str(source),
            'original_source_sha256':{p.name:sha(p) for p in src_files},
            'trial_source_sha256':{p.name:sha(code/p.name) for p in src_files},
            'models_output':str(destination/'models'),
            'results_output':str(destination/'results'),
            'feature_input':str(root/'features'),
            'paper_feature_input':str(root/'features_experiments'/'akinci_ozbay_cqcc'),
            'split_assignments':'unchanged; read from original configuration',
        }
        (destination/'trial_manifest.json').write_text(json.dumps(notes,indent=2),encoding='utf8')
        (destination/'README_FIRST.txt').write_text(
            f'RUN CNN-ONLY FRESH VALIDATION IN ISOLATION:\n'
            f'cd /d "{code}"\n'
            f'python improvedCQCC_train.py --model CNN-only\n\n'
            f'AFTER TRAINING, COMPARE WITH ORIGINAL:\n'
            f'python "{source / "prepare_improvedCQCC_repro_trial.py"}" compare --root "{root}" --name {name} --model CNN-only\n',encoding='utf8')
    except Exception:
        # Avoid accidentally deleting new artifacts on failed work. At this stage no training occurs.
        raise
    print('PASS: isolated training trial prepared. NO training launched.')
    print('Read-only original features:',root/'features')
    print('All NEW model checkpoints saved only under:',destination/'models')
    print('All NEW evaluation outputs saved only under:',destination/'results')
    print('\nNEXT COMMANDS:')
    print(f'cd /d "{code}"')
    print('python improvedCQCC_train.py --model CNN-only')
    print('\nThis runs all four CNN-only folds, preserving the original fold order and initial seed.')


def compare(root,name,model):
    if model not in MODELS: raise SystemExit('ERROR: Unknown baseline model')
    orig=root/'models'/model; new=root/'reproducibility'/name/'models'/model
    o=orig/'fold_results.npy';n=new/'fold_results.npy'
    if not o.is_file() or not n.is_file():
        raise SystemExit(f'ERROR: Need BOTH original and fresh fold results.\nOriginal: {o}\nFresh: {n}')
    od=list(np.load(o,allow_pickle=True)); nd=list(np.load(n,allow_pickle=True))
    oo={int(r['fold']):r for r in od};nn={int(r['fold']):r for r in nd}
    print(f'ORIGINAL vs ISOLATED FRESH 4-FOLD TRAINING: {model}')
    print(f'Original: {o}')
    print(f'Fresh:    {n}')
    missing=sorted(set(oo)-set(nn));extras=sorted(set(nn)-set(oo))
    if missing:print('MISSING fresh fold(s):',missing)
    if extras:print('EXTRA fresh fold(s):',extras)
    exact=0; compared=0
    for k in sorted(set(oo)&set(nn)):
        a=oo[k];b=nn[k]
        print(f'\nFold {k}:')
        print(' training_version:',a.get('training_version'),'(original) versus',b.get('training_version'),'(fresh)')
        if a.get('training_version')!=b.get('training_version'):
            print('  WARNING: training-version mismatch')
        for metric in METRICS:
            if metric not in a or metric not in b:
                continue
            av=float(a[metric]);bv=float(b[metric])
            print(f'  {metric:<27} original={av:.7f} fresh={bv:.7f} delta={bv-av:+.7f}')
        op=orig/f'best_fold{k}.pt';npth=new/f'best_fold{k}.pt'
        if op.is_file() and npth.is_file():
            same=sha(op)==sha(npth)
            print('  checkpoint SHA256 exact: ',same)
            exact+=int(same);compared+=1
        else: print('  WARNING: Missing checkpoint(s) for fold',k)
    print(f'\nChecked exact checkpoint hashes: {exact}/{compared}.')
    print('NOTE: Different hashes do not by themselves imply incorrect model performance.')
    print('Reproducing all original metrics numerically is stronger evidence than rounded displays alone.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['setup','compare'])
    p.add_argument('--root',default=r'E:\Thesis',help='Existing original thesis data root')
    p.add_argument('--name',default='trial01',help='Name for NEW isolated experiment')
    p.add_argument('--model',default='CNN-only',choices=MODELS)
    args=p.parse_args()
    root=Path(args.root).resolve()
    source=Path(__file__).resolve().parent
    if args.action=='setup':setup(root,args.name,source)
    else:compare(root,args.name,args.model)
if __name__=='__main__': main()
