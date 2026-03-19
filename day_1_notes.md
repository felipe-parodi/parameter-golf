# Day 1 Notes - 2026-03-19

## Objective

Set up a local experimentation loop for `parameter-golf`, add measurement tooling around export and evaluation, and get the first real 1-GPU baseline plus diagnostics.

## Code Changes

Updated `train_gpt.py` to add a measurement harness:

- Added `EVAL_ONLY`, `LOAD_PATH`, `EVAL_SEQ_LEN`, `EVAL_SEQ_LEN_SWEEP`, and `QUANT_REPORT_TOPK`.
- Decoupled validation sequence length from training sequence length.
- Added eval-only checkpoint analysis for both `.pt` and `.int8.ptz`.
- Added quantization diagnostics and JSON artifacts:
  - `quant_report_<RUN_ID>.json`
  - `eval_sweep_<RUN_ID>.json`
- Added float-vs-quant gap logging and per-tensor quantization stats.
- Switched eval paths to use the uncompiled base model so multi-length sweeps do not trip `torch.compile` recompile limits.
- Added a follow-up warmup fix: after restoring the pre-warmup state, call `torch._dynamo.reset()` and rebuild the compiled training wrapper.

Status:

- `train_gpt.py` compiles.
- File length is still under the repo cap: `1449` lines.

## Local Setup

- Created `.venv` and installed project dependencies.
- Confirmed GPU access works outside the Codex sandbox in WSL.
- Confirmed local CUDA-visible GPUs exist.
- Downloaded a minimal local dataset for fast iteration:
  - SentencePiece tokenizer `sp1024`
  - 1 training shard
  - full validation split

## Runs And Results

### 1. Smoke training run

Run:

- `RUN_ID=smoke_local`
- `ITERATIONS=5`
- `TRAIN_BATCH_TOKENS=32768`

Key results:

- `final_float` at `eval_seq_len=1024`: `val_loss=6.8727`, `val_bpb=4.0704`
- `final_int8_zlib_roundtrip_exact`: `val_loss=6.88173903`, `val_bpb=4.07575348`
- quant gap: `+0.009019 val_loss`, `+0.005342 bpb`
- compressed artifact: `4,961,397` bytes

Takeaway:

- End-to-end local train/eval/export works.
- Even a junk checkpoint shows measurable export loss.

### 2. Eval-only smoke sweep

Run:

- `RUN_ID=smoke_sweep`
- `EVAL_ONLY=1`
- `LOAD_PATH=final_model.pt`
- `EVAL_SEQ_LEN_SWEEP=512,1024,2048`

Key result:

- All three sequence lengths were effectively identical on the 5-step checkpoint.

Takeaway:

- The harness worked, but the checkpoint was too undertrained to reveal useful eval-length behavior.

### 3. 20-step debug run

Run:

- `RUN_ID=debug20`
- `ITERATIONS=20`
- `TRAIN_BATCH_TOKENS=131072`

Key results:

- `final_float` at `eval_seq_len=1024`: `val_loss=5.8006`, `val_bpb=3.4354`
- `final_int8_zlib_roundtrip_exact`: `val_loss=5.84419708`, `val_bpb=3.46126270`
- quant gap: `+0.043620 val_loss`, `+0.025834 bpb`
- compressed artifact: `5,028,505` bytes

Takeaway:

- Export gap got much larger as the model became less terrible.
- That strengthened the case that export quality is a real optimization surface, not noise.

### 4. 700-step local baseline

Run:

- `RUN_ID=local_medium_700b`
- `ITERATIONS=700`
- `TRAIN_BATCH_TOKENS=131072`
- `VAL_BATCH_SIZE=131072`

Key results:

- step `350/700`: `val_loss=2.7090`, `val_bpb=1.6044`
- step `700/700`: `val_loss=2.4996`, `val_bpb=1.4804`
- `final_float`: `val_loss=2.4996`, `val_bpb=1.4804`
- `final_int8_zlib_roundtrip_exact`: `val_loss=2.50419512`, `val_bpb=1.48312540`
- quant gap: `+0.004606 val_loss`, `+0.002728 bpb`
- compressed artifact: `11,046,334` bytes
- total submission size with code: `11,110,441` bytes

Takeaway:

- The local baseline improved a lot and still has substantial size headroom under the 16MB cap.
- By this point the export gap had become small again, at least according to the training-time report.

## Important Bugs And Investigations

### 1. Sweep recompile bug

The first real eval-only sweep on the 700-step checkpoint failed with:

- `torch._dynamo.exc.FailOnRecompileLimitHit`

Cause:

- the sweep changed input shapes enough to burn through `torch.compile` recompilation limits.

Fix applied:

- eval paths now use the uncompiled base model instead of the compiled training wrapper.

### 2. Possible warmup / compiled-state mismatch

After fixing the sweep crash, a worse issue appeared:

- the reloaded `final_model.pt` did not reproduce the `final_float` score reported at the end of training.

Observed mismatch:

- `local_medium_700b` training log reported `final_float val_loss=2.4996 / val_bpb=1.4804`
- reloading `final_model.pt` in eval-only mode produced much worse losses

Direct A/B check on the saved checkpoint, same first validation batch at `seq_len=1024`:

- eager base model loss: `3.1590211391448975`
- compiled model loss: `3.158039093017578`
- absolute difference: `0.000982046127319336`

Interpretation:

- the saved checkpoint itself appears consistent between eager and compiled inference
- the inconsistency is between training-time reporting and what actually got serialized
- the most likely culprit is the warmup-then-restore flow interacting badly with the compiled wrapper

Fix applied:

- after restoring the initial model/optimizer state at the end of warmup, rebuild the compiled wrapper with:
  - `torch._dynamo.reset()`
  - fresh `torch.compile(base_model, ...)`

Verification status:

- not finished
- a short validation run (`debug20b`) was started but interrupted before completion

## Artifacts Created

- `plan.md`
- `quant_report_smoke_local.json`
- `quant_report_smoke_sweep.json`
- `quant_report_debug20.json`
- `quant_report_local_medium_700b.json`
- `eval_sweep_smoke_sweep.json`
- logs under `logs/`

Current caution:

- `final_model.pt` and `final_model.int8.ptz` from the last local medium run should not be treated as trusted baselines until the warmup/save mismatch is re-verified after the recompilation fix.

## Best Current Takeaways

- The local training loop is usable.
- The measurement harness is useful and already found one real bug.
- Export loss is definitely worth tracking, but the more urgent issue is checkpoint fidelity.
- The next session should start by proving that training-time `final_float` matches reloaded-checkpoint eval.

## Next Steps

1. Re-run a short post-fix training job and verify that `final_model.pt` reproduces the logged `final_float`.
2. If that passes, rerun a clean `EVAL_SEQ_LEN` sweep on the same checkpoint.
3. Only after checkpoint fidelity is proven, trust sweep data for planning the next optimization lane.
