# Fresh-training reproducibility protocol (repro-safe 16-file source package)

The original frozen checkpoints have already reproduced clean/degraded metrics,
and the 26,209 dev OOF scores passed provenance verification. Neither means
that training from audio will generate the exact same float arrays again.

**Do not run `run_all.bat` in your original thesis root as a reproduction test.**
The original training code resumes completed folds, and base feature preparation
may rewrite cache entries. The temporal fusion also can reuse old source OOF.

## Stage A — Freeze and audit the successful run

1. Record Python/CUDA/GPU versions and `python -m pip freeze` (already done).
2. Use the separately supplied `verify_retraining_readiness.py` to inventory
   dataset counts, speaker labels, feature-version hashes, normalization files,
   model/checkpoint SHA256, original condition assignments, and saved metrics.
3. Store the manifest and backups safely. Never overwrite the successful run.

## Stage B — Reproduce preprocessing (fresh ISOLATED root only)

- Create a separate project workspace with its own `features/`, `models/`, and
  `results/` directories. Only INPUT audio and resource archives may be shared
  read-only; don't use hardlinks for mutable caches/checkpoints.
- New code uses THESIS_BASE_DIR, but the frozen `conditions.npy` and
  `test_degraded_conditions.npy` use full paths as dictionary keys. Their
  recording paths must be rebased into the isolated workspace before the new
  feature builders can read them. Check that the **normalized condition labels
  per recording** remain equal to the reference; don't randomly reassign them.
- Extract shared base features and paper-style CQCC sidecars. Compare sample
  feature shapes/values, versions, names, 32/8 speaker split and clip counts,
  before starting training. Full numeric feature hashing may be worthwhile.

## Stage C — Reproduce training without changing methodology

Preserve dataset hashes, four original speaker folds, normalization source,
model architecture, optimizer, batch size, learning rate, dropout, max-window
cap, checkpoint selection and epoch budget. For baseline CNN models, the original
code seeds once for the full run; run **CNN-only, then CNN–GRU, then CNN–GRU–F
via one `--model all` process** if comparing with the historical training order.
Running a single model or resuming a skipped fold may have different RNG state.

With fresh output folders only:

1. `python improvedCQCC_train.py --model all`
2. `python improvedCQCC_retrain.py --model all`
3. `python improvedCQCC_cqcc_train.py --stage cv`
4. `python improvedCQCC_f0_train.py --stage cv`
5. `python improvedCQCC_cqcc_f0_fusion.py --stage baseline-oof`
6. `python improvedCQCC_cqcc_f0_fusion.py --stage oof`
7. `python improvedCQCC_cqcc_f0_fusion.py --stage fit`
8. `python improvedCQCC_cqcc_retrain.py`
9. `python improvedCQCC_f0_retrain.py`

The fallback `--stage baseline-oof` is a new convenience pathway and **has not
been proven numerically equivalent to the original historical v25 OOF**.
The original successful v28-based fusion depended on that v25 OOF, which used
trained original CNN/F0 experts. This is the critical remaining comparison for
strict from-scratch parity, not an automatic guarantee from the new source.

## Stage D — Compare outcomes

Compare folds and input arrays first; then compare final metrics with the
original recorded clean/degraded metrics. AUC/EER/F1 differences may arise from
library versions, CUDA/AMP nondeterminism, and changed checkpoint selection.
Predefine acceptable stability using repeat-run variation, not after viewing
held-out results. Do not use the previously inspected held-out test data to
select or tune a new model. For independent confirmation, reserve another
untouched Filipino-speech sample/corpus.

**Never claim exact fresh-training replication unless the new run has actually
passed stage-by-stage numeric checks.**
