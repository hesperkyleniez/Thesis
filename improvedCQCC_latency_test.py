"""Four-model latency benchmark; proposed model includes Akıncı CQT frontend.
One-second input is ALREADY available; acquisition, resampling, and I/O excluded.
Updated live-window CQCC may differ from full-clip pooled features; not a direct
validation of saved held-out accuracy. Use on own PC/GPU to measure new times.
"""
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

_SOURCE_latency_v28 = r'''
"""Read-only, waveform-to-label latency benchmark for the REVISED CNN-GRU/CQCC/F0 fusion.

Requires the user's ORIGINAL v28 temporal fusion and v25 acoustic experts.
Nothing is trained or modified. Source audio is generated unless --wav is supplied.

Benchmark contract:
* The input is an ALREADY-AVAILABLE 16,000-sample (1-second) 16-kHz mono window.
* Starts timer BEFORE Log-Mel, CQCC, and F0 extraction; ends AFTER final label.
* Uses exactly the existing 3 final expert checkpoints, Platt maps, nine
  aligned-window features, fitted logistic reference, and causal temporal CNN.
* Incorporates --history-windows-1 previously scored windows (cached BEFORE
  the timer) so temporal fusion is exercised for a realistic short clip prefix.
* Requires a warm GPU and synchronizes before/after each timed call.
* Does not measure recording/acquisition, resampling, device I/O, or network.

CAVEAT: The existing cached TEST evaluation pooled CQCC/F0 frame streams over
entire files; per-window live extraction can have edge effects, and a rolling
history is not the same as an arbitrarily long whole-clip aggregate. This test
measures runtime of a feasible per-window streaming path, NOT equivalence of
all accuracy results or latency of the offline full-clip evaluation script.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

from improvedCQCC_config import MODELS_DIR, RESULTS_DIR, SAMPLE_RATE, CQCC_VECTOR_DIM, F0_VECTOR_DIM
from improvedCQCC_features import extract_logmel, extract_improved_cqcc as extract_cqcc, extract_f0
from improvedCQCC_train import CNNGRUOnly
from improvedCQCC_cqcc_train import CQCCMLP
CQCC_DIM=40
from improvedCQCC_f0_train import F0_DIM, F0MLP
import improvedCQCC_cqcc_f0_fusion_v19_calibrated as base
import improvedCQCC_cqcc_f0_fusion_v22_window_level as win
import improvedCQCC_v25_fusion_window as v25
import improvedCQCC_v28_temporal_evidence_fusion as v28

from improvedCQCC_cqcc_f0_fusion import _get_fusion
patch_experiment, preflight_checkpoint, final_info, CQCC_DIR = (lambda m:(m.patch_experiment,m.preflight_checkpoint,m.final_info,m.CQCC_DIR))(_get_fusion()[0])
patch_experiment()
VERSION = 'akinci_ozbay_v28_streaming_latency_v1'
TARGET_MS = 100.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def _time_ms(values):
    a = np.asarray(values, dtype=np.float64)
    return {'n': int(len(a)), 'mean_ms': float(a.mean()),
            'std_ms': float(a.std(ddof=0)), 'median_ms': float(np.median(a)),
            'p95_ms': float(np.percentile(a, 95)),
            'p99_ms': float(np.percentile(a, 99)), 'max_ms': float(a.max()),
            'under_100ms_percent': float(np.mean(a <= TARGET_MS) * 100)}


def make_audio(n, seed=42, signal="legacy_noise"):
    """Pre-generate inputs OUTSIDE timer. legacy_noise mimics original latency_test.py."""
    rng = np.random.default_rng(seed)
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    out = []
    for i in range(n):
        if signal == "legacy_noise":
            out.append(rng.normal(0., 0.1, SAMPLE_RATE).astype(np.float32))
            continue
        f0 = float(rng.uniform(90., 270.))
        phase = 2. * np.pi * (f0 * t + 0.004 * np.sin(2 * np.pi * 3.5 * t))
        sig = sum((1. / k) * np.sin(k * phase) for k in range(1, 9))
        env = (0.4 + 0.6 * np.sin(np.pi * t) ** 2)
        sig = (0.20 * env * sig + rng.normal(0., 0.006, len(t))).astype(np.float32)
        out.append(sig)
    return out


def load_windows_from_wav(path, n):
    """Audio loading/resampling is intentionally OUTSIDE the measured processing time."""
    import librosa
    audio, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        raise RuntimeError('The supplied WAV has zero samples.')
    audio = (audio - float(audio.mean())) / (float(audio.std()) + 1e-8)
    audio /= max(float(np.max(np.abs(audio))), 1e-8)
    hop = SAMPLE_RATE // 2
    starts = np.arange(0, max(1, len(audio)), hop, dtype=np.int64)
    windows = []
    for i in range(n):
        st = int(starts[i % len(starts)])
        w = np.zeros(SAMPLE_RATE, dtype=np.float32)
        src = audio[st:st + SAMPLE_RATE]
        w[:len(src)] = src
        windows.append(w)
    return windows


class Runner:
    def __init__(self):
        # Enforce the same artifacts that v28 evaluation loads.
        self.artifact = v28.load_artifact()  # new Akıncı temporal fusion artifact
        preflight_checkpoint(final=True)  # verify NEW trained CQCC expert and stats
        if CQCC_DIM != CQCC_VECTOR_DIM or F0_DIM != F0_VECTOR_DIM:
            raise RuntimeError('Mismatch between expert and extractor dimensions.')
        if CQCC_DIM != 40 or F0_DIM != 8:
            raise RuntimeError(f'Expected current fixed 40-D CQCC, 8-D F0; got {CQCC_DIM}, {F0_DIM}.')

        self.cnn = CNNGRUOnly(dropout=0.0).to(DEVICE).eval()
        path_cnn = os.path.join(MODELS_DIR, 'CNN-GRU', 'final_model.pt')
        self.cnn.load_state_dict(torch.load(path_cnn, map_location=DEVICE, weights_only=True), strict=True)

        self.cqcc = CQCCMLP(CQCC_DIM).to(DEVICE).eval()
        path_cq = str(CQCC_DIR / 'final_model.pt')
        self.cqcc.load_state_dict(torch.load(path_cq, map_location=DEVICE, weights_only=True), strict=True)
        cq_mu, cq_sigma = final_info()
        cq_stats = dict(mean=cq_mu, std=cq_sigma)

        self.f0 = F0MLP(F0_DIM).to(DEVICE).eval()
        path_f0 = os.path.join(v25.aux_dir('f0'), 'final_model.pt')
        self.f0.load_state_dict(torch.load(path_f0, map_location=DEVICE, weights_only=True), strict=True)
        f0_stats = v25._final_info('f0')

        self.cq_mean = torch.from_numpy(np.asarray(cq_stats['mean'], np.float32)).to(DEVICE)
        self.cq_std = torch.from_numpy(np.asarray(cq_stats['std'], np.float32)).to(DEVICE)
        self.f0_mean = torch.from_numpy(np.asarray(f0_stats['mean'], np.float32)).to(DEVICE)
        self.f0_std = torch.from_numpy(np.asarray(f0_stats['std'], np.float32)).to(DEVICE)
        if (self.cq_std <= 0).any() or (self.f0_std <= 0).any():
            raise RuntimeError('Invalid auxiliary norm std.')

        self.net = v28.TemporalResidual().to(DEVICE).eval()
        ckpt = torch.load(v28.FINAL_STATE_PATH, map_location=DEVICE, weights_only=True)
        if ckpt.get('version') != v28.VERSION or int(ckpt.get('final_epochs', -1)) != int(self.artifact['final_epochs']):
            raise RuntimeError('Temporal net checkpoint version/epoch mismatch.')
        self.net.load_state_dict(ckpt['state_dict'], strict=True)
        self.threshold = float(self.artifact['threshold'])
        self.calibrators = self.artifact['calibrators']
        self.scaler_mean = np.asarray(self.artifact['scaler_mean'], dtype=np.float32)
        self.scaler_scale = np.asarray(self.artifact['scaler_scale'], dtype=np.float32)
        self.ref_coef = np.asarray(self.artifact['reference_coef'], dtype=np.float64)
        self.ref_intercept = float(self.artifact['reference_intercept'])
        if (self.scaler_scale <= 0).any():
            raise RuntimeError('Invalid fusion scaler.')
        if not (len(self.scaler_mean) == len(self.scaler_scale) == len(self.ref_coef) == v28.N_FEATURES):
            raise RuntimeError('Fusion feature dimension mismatch.')
        self.paths = {'cnn_gru': path_cnn, 'cqcc': path_cq, 'f0': path_f0,
                      'fusion_artifact': v28.META_PATH, 'temporal_correction': v28.FINAL_STATE_PATH}

    def expert_probabilities(self, logmel, cqcc, f0):
        """Calculate 3 expert scores, including all device transfers."""
        with torch.inference_mode():
            lm = torch.from_numpy(np.ascontiguousarray(logmel, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            cq = torch.from_numpy(np.ascontiguousarray(cqcc, dtype=np.float32)).to(DEVICE)
            f = torch.from_numpy(np.ascontiguousarray(f0, dtype=np.float32)).to(DEVICE)
            p_cnn = torch.sigmoid(self.cnn(lm, None, None)).reshape(-1).item()
            p_cq = torch.sigmoid(self.cqcc(((cq - self.cq_mean) / self.cq_std)[None])).reshape(-1).item()
            p_f0 = torch.sigmoid(self.f0(((f - self.f0_mean) / self.f0_std)[None])).reshape(-1).item()
        return float(p_cnn), float(p_cq), float(p_f0)

    def feature_row(self, scores):
        a, b, c = scores
        record = dict(path='stream_window', cnn_probs=np.array([a], np.float32),
                      cqcc_probs=np.array([b], np.float32), f0_probs=np.array([c], np.float32))
        x = win.window_features(record, self.calibrators)
        if x.shape != (1, v28.N_FEATURES) or not np.isfinite(x).all():
            raise RuntimeError('Bad fused feature row')
        return x[0].astype(np.float32)

    def fusion(self, rows):
        z = np.asarray(rows, dtype=np.float32).reshape(-1, v28.N_FEATURES)
        if len(z) == 0:
            raise RuntimeError('No fusion history.')
        x = np.ascontiguousarray((z - self.scaler_mean) / self.scaler_scale, dtype=np.float32)
        ref = float(x.mean(axis=0).astype(np.float64) @ self.ref_coef + self.ref_intercept)
        with torch.inference_mode():
            xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            lengths = torch.tensor([len(x)], dtype=torch.long, device=DEVICE)
            baseline = torch.tensor([ref], dtype=torch.float32, device=DEVICE)
            logits, _ = self.net(xt, lengths, baseline)
            p = torch.sigmoid(logits).reshape(-1).item()
        return float(p), 'AI-Generated' if p >= self.threshold else 'Real'

    def process_window(self, audio, previous_rows=(), profile=False):
        if len(audio) != SAMPLE_RATE or not np.isfinite(audio).all():
            raise ValueError('Provide precisely 16,000 finite samples (one second @ 16 kHz).')
        stage = {}
        def mark(name, fn):
            if profile:
                sync()
                start = time.perf_counter()
                result = fn()
                sync()
                stage[name] = (time.perf_counter() - start) * 1000
                return result
            return fn()

        lm = mark('logmel_ms', lambda: extract_logmel(audio))
        cq = mark('cqcc_ms', lambda: extract_cqcc(audio))
        f = mark('f0_ms', lambda: extract_f0(audio))
        if lm.shape != (128, 128) or cq.shape != (CQCC_DIM,) or f.shape != (F0_DIM,):
            raise RuntimeError(f'Feature shape mismatch: {lm.shape}, {cq.shape}, {f.shape}')
        probs = mark('three_experts_ms', lambda: self.expert_probabilities(lm, cq, f))
        row = mark('platt_features_ms', lambda: self.feature_row(probs))
        fused_p, label = mark('temporal_fusion_ms', lambda: self.fusion([*previous_rows, row]))
        if not np.isfinite(fused_p):
            raise RuntimeError('Nonfinite fused probability')
        return fused_p, label, row, stage

    def compare_with_v28_reference(self, rows):
        # For numeric parity, reconstruct expert scores from scalar Platt feature rows?
        # Compare the *fusion math* by creating a dummy already-calibrated sequence
        # directly with the exact official net and inference algebra.
        z = np.asarray(rows, np.float32).reshape(-1, v28.N_FEATURES)
        x = np.ascontiguousarray((z - self.scaler_mean) / self.scaler_scale, np.float32)
        ref = float(x.mean(axis=0).astype(np.float64) @ self.ref_coef + self.ref_intercept)
        with torch.inference_mode():
            xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            baseline = torch.tensor([ref], dtype=torch.float32, device=DEVICE)
            out, delta = self.net(xt, torch.tensor([len(x)], device=DEVICE), baseline)
            baseline_numpy = np.asarray([ref], np.float32)
            # Same exact path as official predict_dataset on a one-item sequence.
            ds = v28.SequenceDataset([x], np.zeros(1), baseline_numpy, np.ones(1))
            dl = v28.make_loader(ds, 1, False, 42)
            p_official, d_official = v28.predict_dataset(self.net, dl, DEVICE)
        p_manual = float(torch.sigmoid(out).item())
        if abs(p_manual - float(p_official[0])) > 1e-6 or abs(float(delta.item()) - float(d_official[0])) > 1e-6:
            raise RuntimeError('Fusion tensor inference failed to match official v28 predict_dataset()')
        return float(p_official[0]), float(d_official[0])


def stage_check(runner, samples):
    print('Loaded and verified ORIGINAL frozen checkpoints and trained fusion artifact.')
    print('  Device:', DEVICE)
    print('  CQCC/F0 dimensions:', CQCC_DIM, F0_DIM)
    print('  Fusion input features:', v28.N_FEATURES, 'threshold:', runner.threshold)
    for label, path in runner.paths.items():
        print('  ', label, '=', path)
    rows = []
    for window in samples[:3]:
        prob, lbl, row, parts = runner.process_window(window, previous_rows=rows)
        rows.append(row)
    p_v28, delta = runner.compare_with_v28_reference(rows)
    if abs(p_v28 - float(prob)) > 1e-6:
        raise RuntimeError('Read-only fusion parity check failed')
    print(f'  3-window prefix: Real/AI output={lbl}, probability={prob:.6f}, correction={delta:.6f}')
    print('PREFLIGHT PASSED. These probabilities are not an accuracy evaluation.')


def stage_fidelity(samples):
    """Compare streaming window features with the historical full-clip cache extractor."""
    from improvedCQCC_features import extract_clip_features
    a = np.asarray(samples[0], np.float32)
    b = np.asarray(samples[1], np.float32)
    full = np.concatenate([a, b], axis=0)
    expected = extract_clip_features(full)
    print("REFERENCE-CACHE vs LIVE-WINDOW feature comparison (first window of a 2-second clip):")
    for k, fn in (("logmel", extract_logmel), ("cqcc", extract_cqcc), ("f0", extract_f0)):
        ref = np.asarray(expected[k][0], np.float64)
        live = np.asarray(fn(a), np.float64)
        if ref.shape != live.shape:
            raise RuntimeError(f"Shape disagreement: {k}: {ref.shape} / {live.shape}")
        rmse = np.sqrt(np.mean((ref - live)**2))
        relative = float(rmse / (np.sqrt(np.mean(ref**2)) + 1e-12))
        print(f"  {k:7} relative-RMSE={relative:.6f} absolute-RMSE={rmse:.5f}")
    print("These are ALTERNATIVE feature-extraction procedures; large differences mean")
    print("the streaming benchmark cannot yet be treated as accuracy-equivalent to the trained cached model.")


def stage_self_test():
    # Does not require original checkpoints; tests data shapes and math only.
    rng = np.random.default_rng(1)
    x = make_audio(1)[0]
    for name, extraction, shape in [('logmel', extract_logmel, (128, 128)),
                                     ('cqcc', extract_cqcc, (CQCC_VECTOR_DIM,)),
                                     ('f0', extract_f0, (F0_VECTOR_DIM,))]:
        arr = extraction(x)
        if arr.shape != shape or not np.isfinite(arr).all():
            raise RuntimeError(f'{name} shape/finiteness check failed: {arr.shape}')
        print(f'  {name:7} {arr.shape}: PASS')
    # The causal temporal network should generate valid outputs for 1 and 8 windows.
    net = v28.TemporalResidual().to(DEVICE).eval()
    with torch.inference_mode():
        for n in (1, 3, 8):
            z = torch.from_numpy(rng.normal(size=(1, n, v28.N_FEATURES)).astype(np.float32)).to(DEVICE)
            out, corr = net(z, torch.tensor([n], device=DEVICE), torch.tensor([0.], device=DEVICE))
            if not (torch.isfinite(out).all() and torch.isfinite(corr).all()):
                raise RuntimeError('Temporal net smoke test failed')
            print(f'  temporal length={n}: PASS')
    print('SELF-TEST PASSED. No model files, audio, or test records modified.')


def stage_benchmark(runner, audio, repetitions, warmup, history_windows, profile):
    if repetitions < 20 or warmup < 1 or history_windows < 1:
        raise ValueError('Require runs>=20, warmup>=1, history-windows>=1')
    # Build historical expert scores without measuring them. These were scored
    # when their previous audio windows arrived; only NEW window work is timed.
    history = []
    for idx in range(history_windows - 1):
        *_others, row, _parts = runner.process_window(audio[idx % len(audio)], history)
        history.append(row)
    sample_audio = audio[history_windows - 1:]
    if len(sample_audio) < repetitions + warmup:
        raise RuntimeError('Insufficient prepared audio sample windows.')

    for j in range(warmup):
        runner.process_window(sample_audio[j], history)
    sync()
    timings = []
    parts_record = []
    last_result = None
    for j in range(repetitions):
        window = sample_audio[warmup + j]
        sync()
        t0 = time.perf_counter()
        prob, label, row, parts = runner.process_window(window, history, profile=profile)
        sync()
        timings.append((time.perf_counter() - t0) * 1000.)
        if profile:
            parts_record.append(parts)
        last_result = {'class': label, 'score': prob}
        if (j + 1) % 25 == 0:
            print(f'  completed {j+1}/{repetitions}')
    summary = _time_ms(timings)
    print(f'\nIMPROVED CNN-GRU + CQCC + F0 -- {history_windows}-window history')
    print(f'  mean={summary["mean_ms"]:.2f} ms std={summary["std_ms"]:.2f} ms '
          f'median={summary["median_ms"]:.2f} ms p95={summary["p95_ms"]:.2f} ms '
          f'p99={summary["p99_ms"]:.2f} ms max={summary["max_ms"]:.2f} ms')
    print(f'  <=100 ms: {summary["under_100ms_percent"]:.1f}% of runs | '
          f'mean target: {"MET" if summary["mean_ms"] <= TARGET_MS else "NOT MET"} | '
          f'p95 target: {"MET" if summary["p95_ms"] <= TARGET_MS else "NOT MET"}')
    if parts_record:
        for name in parts_record[0]:
            stats = _time_ms([r[name] for r in parts_record]); print(f'  {name:<20} mean={stats["mean_ms"]:.2f} ms p95={stats["p95_ms"]:.2f} ms')
    return summary, timings, parts_record, last_result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', choices=('self-test', 'check', 'fidelity', 'benchmark'), default='benchmark')
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--history-windows', type=int, default=8,
                        help='Window count in scored prefix (includes current window); 1=initial cold-history output')
    parser.add_argument('--wav', type=str, default=None, help='Optional real WAV, already outside timed acquisition')
    parser.add_argument('--signal', choices=('legacy_noise', 'voiced'), default='legacy_noise',
                        help='Synthetic input signal; legacy_noise matches the original latency benchmark distribution')
    parser.add_argument('--profile', action='store_true', help='Measure detailed per-stage times (adds synchronization overhead)')
    parser.add_argument('--no-save', action='store_true', help='Do not save JSON report')
    args = parser.parse_args()
    if args.stage == 'self-test':
        stage_self_test()
        return
    n = max(4, args.warmup + args.runs + args.history_windows)
    samples = load_windows_from_wav(args.wav, n) if args.wav else make_audio(n, signal=args.signal)
    print('Benchmark version:', VERSION)
    print('Source:', args.wav if args.wav else f'synthetic {args.signal} (seed=42)')
    if args.stage == 'fidelity':
        stage_fidelity(samples)
        return
    print('Device:', DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else '')
    print('Running checkpoint/source checks outside the timer ...')
    runner = Runner()
    stage_check(runner, samples)
    if args.stage == 'check':
        return
    summary, timings, parts, last = stage_benchmark(runner, samples, args.runs, args.warmup,
                                                   args.history_windows, args.profile)
    print('\nNOTE: 1-second input ALREADY AVAILABLE; acquisition, resampling and I/O excluded.')
    print('NOTE: CQCC/F0 per-window frontends may differ from full-file pooled test caches near boundaries.')
    print('NOTE: Fixed prefix window count; not a demonstration of unbounded real-time streaming.')
    if not args.no_save:
        path = os.path.join(RESULTS_DIR, 'latency_akinci_cqcc_f0')
        os.makedirs(path, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        outfile = os.path.join(path, f'latency_{stamp}.json')
        payload = {'version': VERSION, 'device': str(DEVICE),
                   'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None,
                   'source': str(args.wav) if args.wav else 'synthetic '+args.signal,
                   'runs': args.runs, 'warmup': args.warmup, 'history_windows': args.history_windows,
                   'profile': args.profile, 'samples_per_window': SAMPLE_RATE, 'sample_rate': SAMPLE_RATE,
                   'target_ms': TARGET_MS, 'stats': summary,
                   'stage_ms': [{k: float(v) for k, v in d.items()} for d in parts] if args.profile else None,
                   'measurements_ms': [float(x) for x in timings], 'last_output': last,
                   'checkpoints': runner.paths,
                   'caveat': 'Per-window streaming prefix benchmark. Previously scored history excluded; waveform acquisition, resampling and I/O excluded. Per-window CQCC/F0 may differ from cached full-clip pooling.'}
        with open(outfile, 'x', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print('Saved new report:', outfile)


if __name__ == '__main__':
    main()
'''
_SOURCE_latency_baselines = r'''
"""Per-window end-to-end latency benchmark for retained base and prediction-fusion models."""

import argparse
import os
import sys
import time

import numpy as np
import torch

# The running script directory is already on sys.path; do not inject historical code paths.
from improvedCQCC_config import MODELS_DIR, SAMPLE_RATE, DROPOUT
from improvedCQCC_features import extract_logmel, extract_mfcc, extract_cqcc, extract_f0
from improvedCQCC_train import CNNOnly, CNNGRUOnly, CNNGRUFusion
from improvedCQCC_cqcc_train import CQCCMLP, VERSION as CQCC_VERSION
CQCC_DIM=40
from improvedCQCC_f0_train import F0_DIM, F0MLP, F0_VERSION
from improvedCQCC_cqcc_f0_fusion_v19_calibrated import (
    load_fusion_model as load_f0_fusion_model,
    fusion_predict_proba as f0_fusion_predict_proba,
    summarize_probabilities as summarize_f0_probabilities,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CHOICES = ("CNN-only", "CNN-GRU", "CNN-GRU-F")


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def load_state(model, folder):
    path = os.path.join(MODELS_DIR, folder, "final_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_cqcc_stats():
    path = os.path.join(MODELS_DIR, "CQCC", "final_norm_stats.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    stats = np.load(path, allow_pickle=True).item()
    if stats.get("training_version") != CQCC_VERSION:
        raise RuntimeError(
            "CQCC final model/stats use an older CQCC definition. "
            "Run cqcc_train.py then cqcc_retrain.py first."
        )
    return (
        np.asarray(stats["mean"], dtype=np.float32),
        np.asarray(stats["std"], dtype=np.float32),
    )



def load_f0_stats():
    path = os.path.join(MODELS_DIR, "F0", "final_norm_stats.npy")
    stats = np.load(path, allow_pickle=True).item()
    if stats.get("training_version") != F0_VERSION:
        raise RuntimeError("F0 final model/stats incompatible.")
    return np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)


def build_runner(model_name):
    if model_name == "CNN-only":
        model = load_state(CNNOnly(dropout=0.0).to(DEVICE), "CNN-only")

        def run(audio):
            logmel = extract_logmel(audio)
            lm = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                return torch.sigmoid(model(lm)).item()
        return run

    if model_name == "CNN-GRU":
        model = load_state(CNNGRUOnly(dropout=0.0).to(DEVICE), "CNN-GRU")

        def run(audio):
            logmel = extract_logmel(audio)
            lm = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                return torch.sigmoid(model(lm, None, None)).item()
        return run

    if model_name == "CNN-GRU-F":
        model = load_state(CNNGRUFusion(dropout=0.0).to(DEVICE), "CNN-GRU-F")

        # Use final normalization stats exactly as evaluation does for MFCC.
        norm_path = os.path.join(MODELS_DIR, "CNN-GRU-F", "final_norm_stats.npy")
        stats = np.load(norm_path, allow_pickle=True).item()
        mfcc_mean = np.asarray(stats["mfcc_mean"], dtype=np.float32)
        mfcc_std = np.asarray(stats["mfcc_std"], dtype=np.float32)
        # A single latency window has no speaker-wide F0 context; benchmark the
        # feature extraction/model compute with the raw pooled F0 vector.

        def run(audio):
            logmel = extract_logmel(audio)
            mfcc = (extract_mfcc(audio) - mfcc_mean) / mfcc_std
            f0 = extract_f0(audio)
            lm = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            mc = torch.from_numpy(mfcc.astype(np.float32)).unsqueeze(0).to(DEVICE)
            f0t = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                return torch.sigmoid(model(lm, mc, f0t)).item()
        return run



    if model_name == "CNN-GRU-CQCC-F0-F":
        cnn = load_state(CNNGRUOnly(dropout=0.0).to(DEVICE), "CNN-GRU")
        cqcc = load_state(CQCCMLP(CQCC_DIM).to(DEVICE), "CQCC")
        cqcc_mean, cqcc_std = load_cqcc_stats()
        f0 = load_state(F0MLP(F0_DIM).to(DEVICE), "F0")
        f0_mean, f0_std = load_f0_stats()
        fusion = load_f0_fusion_model()

        def run(audio):
            logmel = extract_logmel(audio)
            cq = (extract_cqcc(audio) - cqcc_mean) / cqcc_std
            f0v = (extract_f0(audio) - f0_mean) / f0_std
            lm = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            cqt = torch.from_numpy(cq.astype(np.float32)).unsqueeze(0).to(DEVICE)
            f0t = torch.from_numpy(f0v.astype(np.float32)).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                p_cnn = torch.sigmoid(cnn(lm, None, None)).item()
                p_cqcc = torch.sigmoid(cqcc(cqt)).item()
                p_f0 = torch.sigmoid(f0(f0t)).item()
            x = summarize_f0_probabilities([p_cnn], [p_cqcc], [p_f0]).reshape(1, -1)
            return float(f0_fusion_predict_proba(fusion, x)[0])
        return run

    raise ValueError(model_name)


def benchmark_one(model_name, n_runs):
    runner = build_runner(model_name)
    rng = np.random.default_rng(42)

    for _ in range(10):
        runner(rng.normal(0, 0.1, SAMPLE_RATE).astype(np.float32))
    sync()

    times = []
    for _ in range(n_runs):
        audio = rng.normal(0, 0.1, SAMPLE_RATE).astype(np.float32)
        sync()
        t0 = time.perf_counter()
        runner(audio)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    values = np.asarray(times)
    return {
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CHOICES) + ["all"], default="all")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    names = MODEL_CHOICES if args.model == "all" else (args.model,)
    print(f"Device: {DEVICE}")
    print(f"Runs per model: {args.runs}")
    print("100 ms per-window target\n")

    for name in names:
        result = benchmark_one(name, args.runs)
        print(
            f"{name:<18} mean={result['mean_ms']:.2f} ms  "
            f"std={result['std_ms']:.2f} ms  p95={result['p95_ms']:.2f} ms  "
            f"{'MET' if result['p95_ms'] <= 100.0 else 'NOT MET'}"
        )


if __name__ == "__main__":
    main()
'''

def _get_latency_module(name):
    from improvedCQCC_cqcc_f0_fusion import _get_fusion
    _get_fusion()
    if name=='improvedCQCC_latency_v28':
        return _compile_internal(name,_SOURCE_latency_v28)
    return _compile_internal(name,_SOURCE_latency_baselines)
"""Benchmark all four comparison models.
For the updated fusion, measure new one-second CQT + all 3 experts + temporal correction.
This benchmark does not include acquisition/I/O/resampling and is not a
full-clip feature-pooling equivalence test. See README for caveats.
"""
import sys,argparse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['CNN-only','CNN-GRU','CNN-GRU-F','CNN-GRU-CQCC-F0-F'])
    p.add_argument('--stage',default='benchmark',choices=['benchmark','check','fidelity','self-test'])
    p.add_argument('--runs',type=int,default=100)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--history-windows',type=int,default=8)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--wav')
    a=p.parse_args()
    saved=sys.argv
    try:
        if a.model=='CNN-GRU-CQCC-F0-F':
            module = _get_latency_module('improvedCQCC_latency_v28')
            sys.argv=[saved[0],'--stage',a.stage,'--runs',str(a.runs),'--warmup',str(a.warmup),
                      '--history-windows',str(a.history_windows)]
            if a.profile:sys.argv.append('--profile')
            if a.wav:sys.argv+=['--wav',a.wav]
            module.main()
        else:
            if a.stage!='benchmark':p.error('baseline CNN models support --stage benchmark only')
            module = _get_latency_module('improvedCQCC_latency_baselines')
            # Legacy baseline script's exact parser, forwarded below after option discovery.
            sys.argv=[saved[0],'--model',a.model,'--runs',str(a.runs)]
            module.main()
    finally:sys.argv=saved
if __name__=='__main__':main()
