# Experiment Log

## Run 1: Baseline (FA2) — 8xH100 SXM
- **Date**: 2026-03-21
- **Config**: PR #315 base, no TTT, no memory tokens
- **GPUs**: 8xH100 SXM (FA2 fallback, ~116ms/step)
- **Steps**: 5162/9000 (wallclock capped)
- **Results**:
  - Pre-quant val_bpb: 1.1534
  - Post-int6 roundtrip val_bpb: 1.1600
  - Sliding window (s64) val_bpb: **1.1366**
  - Artifact: 15.55 MB
- **Notes**: FA2 costs ~30ms/step overhead → 27% fewer steps.

## Run 2: TTT (FA2) — 8xH100 SXM [INTERRUPTED]
- **Date**: 2026-03-21
- **Config**: PR #315 base + TTT
- **Results**: Pod killed (low balance). No results.

## Run 3: TTT (FA2) — 6xH100 SXM [ABORTED]
- **Date**: 2026-03-21
- **Config**: PR #315 base + TTT + USE_COMPILE=0
- **Notes**: Wrong template (Python 3.11), torch.compile broken, ~440ms/step. Aborted.

## Run 4: TTT + FA3 — 8xH100 SXM ⭐
- **Date**: 2026-03-21
- **Config**: PR #315 base + TTT (lr=0.002, epochs=3, freeze=2) + FA3
- **GPUs**: 8xH100 SXM (FA3, ~82.7ms/step)
- **Steps**: 7256, Seed: 1337
- **Results**:
  - Pre-quant val_bpb: 1.1417
  - Post-int6 roundtrip val_bpb: 1.1475 (quant gap: +0.0058)
  - TTT time: 47.8s
  - **Sliding window (s64) val_bpb: 1.1242** ⭐
  - Artifact: 15.80 MB
- **Notes**: Beats PR #315 (1.1248) by 0.0006. Not enough for record (need ≥0.005 nats).

## Run 5: Memory Tokens + TTT + FA3 — 8xH100 SXM
- **Date**: 2026-03-21
- **Config**: Run 4 + NUM_MEMORY_TOKENS=64
- **GPUs**: 8xH100 SXM (FA3, ~83.5ms/step)
- **Steps**: 7183, Seed: 1337
- **Results**:
  - Pre-quant val_bpb: 1.1421
  - Post-int6 roundtrip val_bpb: 1.1476 (quant gap: +0.0055)
  - TTT time: 47.9s (loss barely moved: 1.9398 → 1.9392)
  - **Sliding window (s64) val_bpb: 1.1244**
  - Artifact: 15.82 MB
- **Notes**: Memory tokens helped during training (1.1806 vs 1.1819 at 6k) but washed out after quant+sliding. Essentially same as Run 4. TTT showing diminishing returns (loss flat across epochs).

## Key Observations
- **Quant gap is the bottleneck**: ~0.006 BPB lost to int6 quantization. Reducing this is key.
- **TTT saturating**: Only ~0.002 BPB gain on strong baseline. Loss barely moves across epochs.
- **Memory tokens**: Help pre-quant but don't survive quantization. Not worth the complexity.
- **Need ≤1.1198 BPB** to beat SOTA by 0.005 nats for a record submission.
- **Current gap to record**: 0.0044 BPB.

## New Ideas from PR Review (2026-03-21 evening)

### High Priority
1. **Aggressive warmdown (WARMDOWN_ITERS=20000)** — PR #365: entire schedule is cosine decay. Quant gap drops from 0.014 to 0.005. Single hyperparameter change.
2. **Smaller batch (TRAIN_BATCH_TOKENS=524288)** — PR #364: more optimizer steps in 10 min. 3-seed verified at 1.1497.
3. **Backout connection** — PR #366: subtract lambda*h_mid from final output. 1 scalar param, zero compute.

### Important Negative Results
- **Depth recurrence is dead** — PR #363: quant error amplifies ~900x over 3 cycles.
- **QAT costs 8% of steps** — PR #360: net negative under 10-min budget.
- **Naive EMA with .cpu().clone() loses 32% throughput** — PR #360.

## Pending Experiments
- [ ] Aggressive warmdown: WARMDOWN_ITERS=20000 + TTT
- [ ] Smaller batch: TRAIN_BATCH_TOKENS=524288 WARMDOWN_ITERS=4000 + TTT
- [ ] Backout connection + best warmdown config
- [ ] Aggressive TTT: TTT_LR=0.005 TTT_EPOCHS=5
- [ ] Causal TTT variant
- [ ] Grad quant (GRAD_QUANT=1)
- [ ] Z-loss (ZLOSS_WEIGHT=1e-4)
- [ ] Combine best innovations for 3-seed submission
