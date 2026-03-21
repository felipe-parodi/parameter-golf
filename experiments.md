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

## Run 3: TTT (FA2) — 6xH100 SXM [ABORTED]
- **Date**: 2026-03-21
- **Config**: PR #315 base + TTT + USE_COMPILE=0
- **GPUs**: 6xH100 SXM (wrong template, Python 3.11, old PyTorch)
- **Results**: ~440ms/step, too slow. Aborted.
- **Notes**: Non-template pod, torch.compile broken, flash_attn dtype issues.

## Run 4: TTT + FA3 — 8xH100 SXM ⭐
- **Date**: 2026-03-21
- **Config**: PR #315 base + TTT_ENABLED=1 (lr=0.002, epochs=3, freeze_blocks=2) + FA3
- **GPUs**: 8xH100 SXM (FA3, ~82.7ms/step)
- **Steps**: 7256/9000 (wallclock capped)
- **Seed**: 1337
- **Results**:
  - Pre-quant val_bpb: 1.1417
  - Post-int6 roundtrip val_bpb: 1.1475
  - TTT time: 47.8s
  - **Sliding window (s64) val_bpb: 1.1242** ⭐
  - Artifact: 15.80 MB (15,801,787 bytes)
  - Total eval time: ~142s (TTT 48s + roundtrip 2s + sliding 92s)
- **Notes**: Beats PR #315 (1.1248) by 0.0006. Matches PR #338 territory. FA3 gives 82.7ms/step → 7256 steps vs 5162 on FA2.

## Comparison to SOTA
| Submission | BPB | Steps | TTT |
|---|---|---|---|
| PR #315 (SOTA) | 1.1248 | 7051 | No |
| PR #338 (TTT) | 1.1254 | 7068 | Yes (same recipe) |
| **Ours (Run 4)** | **1.1242** | 7256 | Yes |

## Next Steps
- [ ] Seed 42: same config, verify consistency
- [ ] Seed 2025: same config, need 3 seeds for p<0.01
- [ ] Memory tokens (NUM_MEMORY_TOKENS=64) + TTT — potential -0.01 BPB
- [ ] Gradient-guided quant (GRAD_QUANT=1) — save artifact bytes
- [ ] Z-loss (ZLOSS_WEIGHT=1e-4) — may reduce quant gap
