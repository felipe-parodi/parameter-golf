# Experiment Log

## Run 1: Baseline (FA2) — 8xH100 SXM
- **Date**: 2026-03-21
- **Config**: PR #315 base, no TTT, no memory tokens
- **GPUs**: 8xH100 SXM (FA2 fallback, ~116ms/step)
- **Steps**: 5162/9000 (wallclock capped, fewer steps due to FA2)
- **Results**:
  - Pre-quant val_bpb: 1.1534
  - Post-int6 roundtrip val_bpb: 1.1600
  - Sliding window (s64) val_bpb: **1.1366**
  - Artifact: 15.55 MB
- **Notes**: FA3 wasn't installed. FA2 costs ~30ms/step overhead → 27% fewer steps than PR #315's 7050. Quant gap (+0.0066) matches PR #315.

## Run 2: TTT (FA2) — 8xH100 SXM [INTERRUPTED]
- **Date**: 2026-03-21
- **Config**: PR #315 base + TTT_ENABLED=1 (lr=0.002, epochs=3, freeze_blocks=2)
- **GPUs**: 8xH100 SXM (FA2 fallback)
- **Steps**: ~mid-training
- **Results**: Pod killed (low balance). No results.

## Run 3: TTT (FA2) — 6xH100 SXM [IN PROGRESS]
- **Date**: 2026-03-21
- **Config**: PR #315 base + TTT_ENABLED=1 (lr=0.002, epochs=3, freeze_blocks=2)
- **GPUs**: 6xH100 SXM (FA2 fallback)
- **Steps**: TBD
- **Results**: TBD
- **Notes**: 6 GPUs → grad_accum_steps changes. Expect slower but should validate TTT works.

## Pending Experiments
- [ ] Memory tokens (NUM_MEMORY_TOKENS=64) + TTT
- [ ] Gradient-guided quant (GRAD_QUANT=1) + TTT
- [ ] Z-loss (ZLOSS_WEIGHT=1e-4) + TTT
- [ ] Full stack: memory tokens + TTT + grad quant + z-loss
- [ ] FA3 build + full stack (need 8xH100 SXM)
- [ ] 3-seed submission runs (SEED=1337,42,2025)
