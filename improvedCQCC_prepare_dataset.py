"""All-model feature preparation. Original base + new 40-D paper CQCC sidecars.

Preserves the original Log-Mel/MFCC/F0/legacy-CQCC base caches, as well as
exact independently verified Akıncı–Özbay CQCC sidecars in their separate paths.
--stage all / base / paper / verify; --train-only; --workers 1-4.
"""
from __future__ import annotations
import argparse
import sys

def _compile_internal(name, source):
    """Install one readable internal legacy implementation (no extra files)."""
    import types, sys
    cached=sys.modules.get(name)
    if cached is not None:
        return cached
    mod=types.ModuleType(name)
    mod.__file__=f'<embedded source in {__file__}: {name}>'
    sys.modules[name]=mod
    try:
        exec(compile(source,mod.__file__,'exec'),mod.__dict__)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return mod

_SOURCE_base_prepare = r'''
"""Prepare training/validation and clean held-out feature caches.

The fixed augmentation assignment proportions are preserved exactly.  This file
uses augmentation.py as the single degradation implementation so training and
robustness evaluation cannot silently drift apart.
"""

import argparse
import hashlib
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import numpy as np
from tqdm import tqdm

# The running script directory is already on sys.path; do not inject historical code paths.
from improvedCQCC_config import (
    TRAIN_DIR,
    TEST_DIR,
    FEATURES_DIR,
    SAMPLE_RATE,
    RESOURCES_DIR,
    OPENSLR_DIR,
    AIR_DIR,
    URBANSOUND_DIR,
    AUGMENTATION_SPLITS,
    LABEL_REAL,
    LABEL_AI,
    SEED,
    PREP_WORKERS,
    COMPRESS_FEATURE_CACHE,
)
from improvedCQCC_dataset import load_all_files
from improvedCQCC_augmentation import collect_rir_files, apply_augmentation, peak_normalize
from improvedCQCC_features import extract_clip_features

CACHE_VERSION = "v15_robust_noise_diversity_cqcc_f0"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_if_needed(zip_path, out_dir):
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        return
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing resource archive: {zip_path}")
    print(f"Extracting {zip_path}...")
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def ensure_resources():
    extract_if_needed(os.path.join(RESOURCES_DIR, "rirs_noises.zip"), OPENSLR_DIR)
    extract_if_needed(os.path.join(RESOURCES_DIR, "archive.zip"), URBANSOUND_DIR)
    extract_if_needed(os.path.join(RESOURCES_DIR, "AIR_wav_files.zip"), AIR_DIR)

    room_rirs, mobile_rirs, urban_files = collect_rir_files()
    print(f"Room RIRs   : {len(room_rirs)}")
    print(f"Mobile RIRs : {len(mobile_rirs)}")
    print(f"Urban noise : {len(urban_files)}")

    if not room_rirs:
        raise RuntimeError("No room RIRs found")
    if not mobile_rirs:
        raise RuntimeError("No mobile/device RIRs found")
    if not urban_files:
        raise RuntimeError("No UrbanSound8K WAV files found")
    return room_rirs, mobile_rirs, urban_files


def pair_key(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    return name[:-3] if name.endswith("_AI") else name


def pair_rng(filepath):
    """Stable per-pair RNG so paired real/AI clips get matched degradation draws."""
    digest = hashlib.sha256(f"{SEED}|{pair_key(filepath)}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def assign_fixed_conditions(samples):
    """Load the existing assignment map, or create it once with the same split."""
    conditions_path = os.path.join(FEATURES_DIR, "conditions.npy")
    if os.path.exists(conditions_path):
        print("Loading existing condition assignments (split preserved)...")
        return dict(np.load(conditions_path, allow_pickle=True).item())

    print("Assigning paired augmentation conditions...")
    names = list(AUGMENTATION_SPLITS.keys())
    probs = list(AUGMENTATION_SPLITS.values())

    pair_conditions = {}
    assignments = {}
    for filepath, _, _ in samples:
        key = pair_key(filepath)
        if key not in pair_conditions:
            pair_conditions[key] = np.random.choice(names, p=probs)
        assignments[filepath] = pair_conditions[key]

    np.save(conditions_path, assignments)

    from collections import Counter
    counts = Counter(assignments.values())
    print("Paired condition distribution:")
    for condition, count in sorted(counts.items()):
        print(f"  {condition:<20}: {count:>6} ({100*count/len(assignments):.1f}%)")
    print(f"Saved: {conditions_path}")
    return assignments


def _output_path(split_name, filepath, label, speaker_id):
    speaker_dir = os.path.join(FEATURES_DIR, split_name, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(filepath))[0]
    if label == LABEL_AI and not name.endswith("_AI"):
        name += "_AI"
    elif label == LABEL_REAL and not name.endswith("_Real"):
        name += "_Real"
    return os.path.join(speaker_dir, name + ".npz")


def _cache_is_current(out_path):
    if not os.path.exists(out_path):
        return False
    try:
        with np.load(out_path, allow_pickle=True) as cached:
            return (
                all(k in cached.files for k in ("logmel", "mfcc", "cqcc", "f0", "ffv"))
                and "cache_version" in cached.files
                and str(cached["cache_version"]) == CACHE_VERSION
            )
    except Exception:
        return False


def _save_cache(out_path, payload):
    # Atomic replace means Ctrl+C cannot leave a half-written cache that looks valid.
    tmp_path = out_path + ".tmp.npz"
    saver = np.savez_compressed if COMPRESS_FEATURE_CACHE else np.savez
    saver(tmp_path, **payload)
    os.replace(tmp_path, out_path)


def _process_one(sample, split_name, room_rirs, mobile_rirs, urban_files, conditions):
    filepath, label, speaker_id = sample
    out_path = _output_path(split_name, filepath, label, speaker_id)
    if _cache_is_current(out_path):
        return "skipped", None

    try:
        audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)

        condition = conditions.get(filepath, "clean") if conditions else "clean"
        if condition == "clean":
            audio = peak_normalize(audio)
        else:
            audio = apply_augmentation(
                audio, condition, room_rirs, mobile_rirs, urban_files,
                rng=pair_rng(filepath), training=(split_name == "train"),
            )

        feat = extract_clip_features(audio)
        n_windows = len(feat["cqcc"])
        payload = dict(
            **feat,
            label=np.array(label),
            speaker=np.array(speaker_id),
            condition=np.array(condition),
            n_windows=np.array(n_windows),
            cache_version=np.array(CACHE_VERSION),
        )
        _save_cache(out_path, payload)
        return "saved", None
    except Exception as exc:
        return "error", (os.path.basename(filepath), str(exc))


def process_split(samples, split_name, room_rirs, mobile_rirs, urban_files, conditions=None):
    success = skipped = 0
    errors = []
    workers = max(1, int(PREP_WORKERS))
    print(f"\nProcessing {split_name}: {len(samples):,} files with {workers} worker(s)")

    if workers == 1:
        iterator = (
            _process_one(s, split_name, room_rirs, mobile_rirs, urban_files, conditions)
            for s in tqdm(samples)
        )
        for status, err in iterator:
            success += status == "saved"
            skipped += status == "skipped"
            if err is not None:
                errors.append(err)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _process_one, s, split_name, room_rirs, mobile_rirs,
                    urban_files, conditions,
                )
                for s in samples
            ]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                status, err = fut.result()
                success += status == "saved"
                skipped += status == "skipped"
                if err is not None:
                    errors.append(err)

    print(f"  Saved   : {success}")
    print(f"  Skipped : {skipped}")
    print(f"  Errors  : {len(errors)}")
    for name, reason in errors[:5]:
        print(f"    {name}: {reason}")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-only", action="store_true",
        help="Rebuild only development features. Recommended for the robustness rerun so held-out caches stay untouched.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    room_rirs, mobile_rirs, urban_files = ensure_resources()

    train_samples = load_all_files(TRAIN_DIR)
    print(f"Training/development files: {len(train_samples):,}")
    conditions = assign_fixed_conditions(train_samples)

    t0 = time.time()
    process_split(
        train_samples, "train", room_rirs, mobile_rirs, urban_files, conditions
    )
    print(f"Training cache time: {(time.time()-t0)/60:.1f} min")

    if not args.train_only:
        test_samples = load_all_files(TEST_DIR)
        print(f"Held-out test files        : {len(test_samples):,}")
        t0 = time.time()
        process_split(
            test_samples, "test", room_rirs, mobile_rirs, urban_files, conditions=None
        )
        print(f"Clean test cache time: {(time.time()-t0)/60:.1f} min")
    else:
        print("Held-out clean/degraded caches left untouched (--train-only).")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
'''

_SOURCE_paper_dev = r'''
"""Build a SEPARATE development-only CQCC sidecar for the Akıncı–Özbay pilot.

Uses the proven pilot extractor and reproduces frozen development degradations.
Never reads the held-out recordings, never changes original caches/checkpoints.

Commands:
    python prepare_akinci_cqcc_dev.py --stage check
    python prepare_akinci_cqcc_dev.py --stage build --limit 12
    python prepare_akinci_cqcc_dev.py --stage build --workers 2
    python prepare_akinci_cqcc_dev.py --stage verify
"""
from __future__ import annotations
import argparse
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

import improvedCQCC_config as cfg
from improvedCQCC_dataset import load_all_files
from improvedCQCC_features import (paper_cqcc_frames, pool_by_centers,
                                    source_cache_path, get_degraded_audio, CONDITIONS)

VERSION = 'akinci_ozbay_2026_cqcc_dev_sidecar_v1'
OUTPUT_ROOT = Path(cfg.BASE_DIR) / 'features_experiments' / 'akinci_ozbay_cqcc'
OUTPUT_TRAIN = OUTPUT_ROOT / 'train'


def candidates():
    """Strict original-dev-indexed list; all known speakers and conditions required."""
    original = load_all_files(cfg.TRAIN_DIR)
    if not original:
        raise RuntimeError('No development WAV files found; check TRAIN_DIR')
    mapping_path = Path(cfg.FEATURES_DIR) / 'conditions.npy'
    if not mapping_path.exists():
        raise RuntimeError(f'Missing frozen original dev condition mapping: {mapping_path}')
    mapping = np.load(mapping_path, allow_pickle=True).item()
    records = []
    for wav, label, speaker in original:
        if speaker in cfg.TEST_SPEAKERS or speaker not in cfg.FOLD_ASSIGNMENTS:
            raise RuntimeError(f'Test/unassigned speaker in development: {speaker}')
        if wav not in mapping:
            raise RuntimeError(f'Development condition missing for {wav}')
        cond = str(mapping[wav])
        if cond not in CONDITIONS:
            raise RuntimeError(f'Unexpected condition {cond} for {wav}')
        orig = Path(source_cache_path(cfg, wav, label, speaker))
        if not orig.is_file():
            raise RuntimeError(f'Missing original feature cache: {orig}')
        out = OUTPUT_TRAIN / speaker / orig.name
        records.append((wav, int(label), speaker, cond, orig, out))
    if len({str(r[-1]) for r in records}) != len(records):
        raise RuntimeError('Output paths are not unique; refusing overwrite')
    return records


def check_entry(rec, verify_original=True):
    wav, label, speaker, cond, orig, out = rec
    with np.load(orig, allow_pickle=False) as d:
        if 'cqcc' not in d.files or d['cqcc'].ndim != 2 or d['cqcc'].shape[1] != 40:
            raise RuntimeError(f'Original cqcc invalid: {orig}')
        if int(d['label']) != label or str(d['condition']) != cond or str(d['speaker']) != speaker:
            raise RuntimeError(f'Original label/speaker/condition mismatch: {orig}')
        old_n = len(d['cqcc'])
    if out.is_file():
        with np.load(out, allow_pickle=False) as d:
            if (str(d.get('cache_version', '')) != VERSION or 'cqcc' not in d.files
                or d['cqcc'].shape != (old_n, 40)
                or int(d['label']) != label or str(d['speaker']) != speaker
                or str(d['condition']) != cond
                or not np.isfinite(d['cqcc']).all()):
                raise RuntimeError(f'Existing experimental cache invalid/incompatible: {out}. Investigate; do not overwrite.')
        return True, old_n
    return False, old_n


def build_one(rec, resources):
    wav, label, speaker, cond, orig, out = rec
    exists, old_n = check_entry(rec)
    if exists:
        return 'skip', 0.0
    import improvedCQCC_features as features
    audio = get_degraded_audio(wav, cond, resources, cfg)
    new = pool_by_centers(audio, paper_cqcc_frames,
                          cfg.WIN_SAMPLES, cfg.HOP_SAMPLES, 512)
    if new.shape != (old_n, 40) or not np.isfinite(new).all():
        raise RuntimeError(f'Window count/feature mismatch {out}: got {new.shape}, expected {(old_n,40)}')
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + '.tmp.npz')
    if out.exists():
        raise RuntimeError(f'Refusing overwrite: {out}')
    # Atomic replace applies ONLY to the experimental directory, never original features.
    np.savez(temp, cqcc=new, label=np.array(label), speaker=np.array(speaker),
             condition=np.array(cond), n_windows=np.array(old_n),
             cache_version=np.array(VERSION))
    os.replace(temp, out)
    return 'saved', 0.0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check','build','verify'), default='check')
    p.add_argument('--limit', type=int, default=0, help='Optional number of clips for extraction smoke test; full training requires all clips.')
    p.add_argument('--workers', type=int, default=6, help='Parallel extraction workers (1-8)')
    args=p.parse_args()
    print('AKINCI-OZBAY CQCC DEVELOPMENT SIDECAR — no test recordings accessed')
    print('Original features: ', Path(cfg.FEATURES_DIR) / 'train')
    print('Experimental features:', OUTPUT_TRAIN)
    records=candidates()
    counts=Counter((r[3],r[1]) for r in records)
    print('Expected original development clips:', len(records))
    print('Conditions x labels:', dict(sorted(counts.items())))
    if args.stage=='check':
        done=sum(int(r[-1].exists()) for r in records)
        print(f'Already created: {done}/{len(records)}. Run --stage build to generate.')
        for rec in [records[0], records[-1]]:
            check_entry(rec)
        return
    if args.stage=='verify':
        incomplete=[]
        for rec in tqdm(records, desc='Verifying experimental development cache'):
            if not check_entry(rec)[0]: incomplete.append(str(rec[-1]))
        if incomplete:
            raise RuntimeError(f'Incomplete experimental cache: {len(incomplete)} missing; first: {incomplete[0]}')
        print(f'PASS: full development cache has {len(records)} clip files, each with valid 40-D CQCC and original condition metadata.')
        return
    if args.limit<0 or args.workers<1 or args.workers>8:
        p.error('--limit must be nonnegative and --workers from 1 to 8')
    if args.limit: records=records[:args.limit]
    # Load resources only when needed, avoiding slow resource discovery on check/verify.
    import improvedCQCC_augmentation as augmentation
    resources=augmentation.collect_rir_files()
    if not resources[0] or not resources[1] or not resources[2]:
        raise RuntimeError('Missing room, mobile RIR or urban noise resources. Cannot replay frozen conditions.')
    print('  Room/mobile/urban resource counts:', tuple(map(len,resources)))
    # Verify deterministic replay of a few ORIGINAL cache entries before creating any sidecars.
    import improvedCQCC_features as features
    seen=Counter()
    errors=[]
    for rec in records:
        cond=rec[3]
        if seen[cond]>=1: continue
        seen[cond]+=1
        wav,label,speaker,cond,orig,_=rec
        audio=get_degraded_audio(wav,cond,resources,cfg)
        replay=pool_by_centers(audio,features.extract_cqcc_frames,
                               cfg.WIN_SAMPLES,cfg.HOP_SAMPLES,cfg.CQCC_HOP_LENGTH)
        with np.load(orig,allow_pickle=False) as d:
            old=np.asarray(d['cqcc'], np.float32)
        err=float(np.linalg.norm(replay-old)/(np.linalg.norm(old)+1e-8))
        print(f'  Frozen-cache reproduction, {cond}: relative error={err:.6f}')
        errors.append(err)
        if err>0.10:
            raise RuntimeError('Frozen original cache does not reproduce; STOP. Check exact project files and augmentation versions.')
    if len(errors) >= 3 and np.median(errors)>0.02:
        raise RuntimeError('Median replay drift too large; stop extraction.')
    t0=time.perf_counter()
    totals=Counter()
    if args.workers==1:
        for rec in tqdm(records, desc='Paper-inspired CQCC'):
            status,_=build_one(rec,resources)
            totals[status]+=1
    else:
        # Bounded map iterator: processing is thread-based, low worker count; no GPU required.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for status,_ in tqdm(pool.map(lambda rec:build_one(rec,resources), records),
                                 total=len(records),desc='Paper-inspired CQCC'):
                totals[status]+=1
    print(f'DONE: {dict(totals)} in {(time.perf_counter()-t0)/60:.1f} minutes')
    if args.limit:
        print('PARTIAL CACHE ONLY. Do not train yet. Rerun --stage build without --limit, then --stage verify.')
    else:
        print('NEXT: python prepare_akinci_cqcc_dev.py --stage verify')

if __name__=='__main__':main()
'''

_SOURCE_paper_heldout = r'''
"""Build ONLY new Akıncı–Özbay-inspired CQCC sidecars for the frozen held-out recordings.

This is a post-development exploratory comparison: the held-out set has already
been inspected in prior work. Existing features/test and features/test_degraded
are READ ONLY. No condition map, checkpoint or audio is rewritten.

Commands:
    python prepare_akinci_cqcc_heldout.py --stage check --split both
    python prepare_akinci_cqcc_heldout.py --stage build --split both --workers 2
    python prepare_akinci_cqcc_heldout.py --stage verify --split both
"""
from __future__ import annotations
import argparse
import hashlib
import random
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import librosa
import numpy as np
from tqdm import tqdm

import improvedCQCC_config as cfg
from improvedCQCC_dataset import load_all_files
from improvedCQCC_features import paper_cqcc_frames, pool_by_centers
from improvedCQCC_prepare_paper_dev import OUTPUT_ROOT
from improvedCQCC_augmentation import collect_rir_files, peak_normalize, apply_augmentation

VERSION = 'akinci_ozbay_2026_cqcc_heldout_sidecar_v1'


def degraded_pair_rng(filepath):
    # Exact seeded realization of the frozen run_degraded_test.py pair_rng.
    stem=Path(filepath).stem
    key=stem[:-3] if stem.endswith('_AI') else stem
    digest=hashlib.sha256(f'degraded-test|{cfg.SEED}|{key}'.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'little'))
ALLOWED = {'test', 'test_degraded'}


def cache_path(split, wav, label, speaker):
    stem = Path(wav).stem
    if label == cfg.LABEL_AI and not stem.endswith('_AI'):
        stem += '_AI'
    elif label == cfg.LABEL_REAL and not stem.endswith('_Real'):
        stem += '_Real'
    return Path(cfg.FEATURES_DIR) / split / speaker / (stem + '.npz')


def records_for_split(split):
    if split not in ALLOWED:
        raise ValueError(split)
    files = load_all_files(cfg.TEST_DIR)
    expected = 7031
    if len(files) != expected:
        raise RuntimeError(f'Unexpected held-out file count: {len(files)}, expected {expected}')
    speakers = {sp for _, _, sp in files}
    if not speakers.issubset(cfg.TEST_SPEAKERS) or speakers & set(cfg.FOLD_ASSIGNMENTS.keys()):
        raise RuntimeError(f'Held-out speaker mismatch: {sorted(speakers)}')
    conditions = None
    if split == 'test_degraded':
        assignment = Path(cfg.FEATURES_DIR) / 'test_degraded' / 'test_degraded_conditions.npy'
        if not assignment.is_file():
            raise FileNotFoundError(f'Frozen held-out condition map missing: {assignment}; STOP')
        conditions = np.load(assignment, allow_pickle=True).item()
    records = []
    for wav, label, speaker in files:
        orig = cache_path(split, wav, label, speaker)
        if not orig.is_file():
            raise FileNotFoundError(f'Original {split} cache missing: {orig}')
        cond = 'clean' if split == 'test' else str(conditions[wav])
        if split == 'test_degraded' and cond not in ('telephone','mobile_gaussian','mobile_urban','room_gaussian','room_urban'):
            raise RuntimeError(f'Unexpected degraded condition {cond} in frozen assignment')
        out = OUTPUT_ROOT / split / speaker / orig.name
        records.append((wav, label, speaker, cond, orig, out))
    if len({str(r[-1]) for r in records}) != len(records):
        raise RuntimeError('Output path collision')
    return records


def inspect(r):
    wav, label, speaker, cond, orig, out = r
    with np.load(orig, allow_pickle=False) as d:
        if 'cqcc' not in d.files or d['cqcc'].ndim != 2 or d['cqcc'].shape[1] != 40:
            raise RuntimeError(f'Invalid original CQCC: {orig}')
        if int(d['label']) != label or str(d['speaker']) != speaker or str(d['condition']) != cond:
            raise RuntimeError(f'Metadata mismatch original {orig}')
        count = len(d['cqcc'])
    if not out.exists():
        return False, count
    with np.load(out, allow_pickle=False) as d:
        if (str(d.get('cache_version', '')) != VERSION or
            d['cqcc'].shape != (count, 40) or not np.isfinite(d['cqcc']).all() or
            int(d['label']) != label or str(d['speaker']) != speaker or
            str(d['condition']) != cond):
            raise RuntimeError(f'Invalid saved experimental sidecar {out}, refusing silent rebuild')
    return True, count


def replay_audio(wav, cond, resources):
    audio, _ = librosa.load(wav, sr=cfg.SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, np.float32)
    audio = (audio - audio.mean()) / (audio.std() + 1e-8)
    if cond == 'clean':
        return peak_normalize(audio)
    return apply_augmentation(audio, cond, *resources,
                              rng=degraded_pair_rng(wav), training=False)


def process_one(r, resources):
    wav, label, speaker, cond, orig, out = r
    done, n = inspect(r)
    if done:
        return 'skip'
    wav_arr = replay_audio(wav, cond, resources)
    new = pool_by_centers(wav_arr, paper_cqcc_frames,
                          cfg.WIN_SAMPLES, cfg.HOP_SAMPLES, 512)
    if new.shape != (n, 40) or not np.isfinite(new).all():
        raise RuntimeError(f'Extracted windows differ from frozen original: {out}: {new.shape} vs ({n}, 40)')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise RuntimeError(f'Refusing overwrite: {out}')
    tmp = out.with_name(out.name + '.tmp.npz')
    np.savez(tmp, cqcc=new, label=np.array(label), speaker=np.array(speaker),
             condition=np.array(cond), n_windows=np.array(n), cache_version=np.array(VERSION))
    os.replace(tmp, out)  # only newly created sidecar
    return 'created'


def execute(split, stage, workers):
    records = records_for_split(split)
    print(f'{split}: expected {len(records):,} recordings; experimental folder: {OUTPUT_ROOT / split}')
    if stage == 'check':
        print(f'  existing: {sum(r[-1].exists() for r in records):,}/{len(records):,}')
        for rec in (records[0], records[-1]):
            inspect(rec)
        print('  original condition map, labels, speakers and source paths verified on sampled entries')
        return
    if stage == 'verify':
        missing=[]
        for r in tqdm(records, desc=f'verify {split}'):
            if not inspect(r)[0]: missing.append(r[-1])
        if missing:
            raise RuntimeError(f'Incomplete {split} CQCC sidecar: {len(missing)} missing. First: {missing[0]}')
        print(f'  PASS: {len(records):,} complete matching held-out CQCC sidecars')
        return
    resources=collect_rir_files()
    if split == 'test_degraded' and not all(resources):
        raise RuntimeError('Missing original frozen degradation resources')
    import improvedCQCC_features as features
    seen=set()
    for r in records:
        if r[3] in seen: continue
        seen.add(r[3])
        wav = replay_audio(r[0], r[3], resources)
        replay=pool_by_centers(wav, features.extract_cqcc_frames,
                               cfg.WIN_SAMPLES, cfg.HOP_SAMPLES, cfg.CQCC_HOP_LENGTH)
        with np.load(r[4], allow_pickle=False) as d:
            reference=np.asarray(d['cqcc'], dtype=np.float32)
        err=float(np.linalg.norm(replay-reference)/(np.linalg.norm(reference)+1e-8))
        print(f'  Frozen original cache replay: {r[3]:<18} relative error={err:.7f}')
        if not np.isfinite(err) or err > .10:
            raise RuntimeError('Original cache replay mismatch; STOP. New features would not match existing clean/degraded test inputs.')
    t0=time.perf_counter()
    totals=Counter()
    if workers==1:
        for rec in tqdm(records, desc=f'paper-inspired {split}'):
            totals[process_one(rec,resources)]+=1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for status in tqdm(pool.map(lambda rec:process_one(rec,resources),records),
                               total=len(records), desc=f'paper-inspired {split}'):
                totals[status]+=1
    print(f'  DONE: {dict(totals)} in {(time.perf_counter()-t0)/60:.1f} minutes')
    execute(split,'verify',workers)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('check','build','verify'), default='check')
    p.add_argument('--split', choices=('test','test_degraded','both'), default='both')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.workers<=8:p.error('--workers must be 1..8')
    print('WARNING: post-selection held-out evaluation is EXPLORATORY; original caches remain read-only')
    for split in (('test','test_degraded') if a.split=='both' else (a.split,)):
        execute(split,a.stage,a.workers)

if __name__=='__main__':main()
'''

def _load_preparation():
    base=_compile_internal('improvedCQCC_original_prepare_dataset',_SOURCE_base_prepare)
    dev=_compile_internal('improvedCQCC_prepare_paper_dev',_SOURCE_paper_dev)
    heldout=_compile_internal('improvedCQCC_prepare_paper_heldout',_SOURCE_paper_heldout)
    return base,dev,heldout

def _invoke(module,argv):
    saved=sys.argv
    try:
        sys.argv=[saved[0],*argv]
        return module.main()
    finally:sys.argv=saved

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('all','base','paper','verify','check'),default='check')
    p.add_argument('--train-only',action='store_true')
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if not 1<=a.workers<=8:p.error('use 1 to 4 workers')
    import improvedCQCC_config as cfg
    cfg.PREP_WORKERS=a.workers
    original,dev,heldout=_load_preparation()
    if a.stage in ('all','base'):
        _invoke(original,['--train-only'] if a.train_only else [])
    if a.stage in ('all','paper','verify','check'):
        step={'all':'build','paper':'build','verify':'verify','check':'check'}[a.stage]
        _invoke(dev,['--stage',step,'--workers',str(a.workers)])
        if not a.train_only:
            _invoke(heldout,['--stage',step,'--split','test','--workers',str(a.workers)])

if __name__=='__main__':main()
