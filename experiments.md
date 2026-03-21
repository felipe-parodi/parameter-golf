# Experiment Log

## Run 1: Baseline (FA2) — 8xH100 SXM
- **Date**: 2026-03-21
- **Config**: PR #315 base, no TTT
- **Steps**: 5162, Seed: 1337, ~116ms/step (FA2)
- **Results**: Sliding **1.1366** | Roundtrip 1.1600 | Pre-quant 1.1534 | Artifact 15.55 MB
- **Notes**: FA2 penalty → 27% fewer steps.

## Run 2: TTT (FA2) — 8xH100 SXM [INTERRUPTED]
- Pod killed (low balance).

## Run 3: TTT — 6xH100 SXM [ABORTED]
- Wrong template (Python 3.11), torch.compile broken, ~440ms/step.

## Run 4: TTT + FA3 — 8xH100 SXM ⭐ BEST
- **Date**: 2026-03-21
- **Config**: PR #315 base + TTT (lr=0.002, ep=3, freeze=2) + FA3
- **Steps**: 7256, Seed: 1337, ~82.7ms/step
- **Results**: Sliding **1.1242** ⭐ | Roundtrip 1.1475 | Pre-quant 1.1417 | Quant gap +0.0058 | Artifact 15.80 MB
- **Notes**: Beats PR #315 (1.1248) by 0.0006. Not enough for record (need ≥0.005 nats = ≤1.1198).

## Run 5: Memory Tokens + TTT + FA3 — 8xH100 SXM
- **Config**: Run 4 + NUM_MEMORY_TOKENS=64
- **Steps**: 7183, Seed: 1337, ~83.5ms/step
- **Results**: Sliding **1.1244** | Roundtrip 1.1476 | Pre-quant 1.1421 | Quant gap +0.0055 | Artifact 15.82 MB
- **Notes**: Memory tokens helped mid-training (0.002 better at 6k) but washed out after quant. Not worth it.

## Run 6: Aggressive Warmdown + TTT + FA3 — 8xH100 SXM ❌
- **Config**: Run 4 but WARMDOWN_ITERS=20000 (entire schedule is cosine decay)
- **Steps**: 7248, Seed: 1337, ~82.8ms/step
- **Results**: Sliding TBD | **Roundtrip 1.2902** ❌ | Pre-quant 1.1503 | **Quant gap +0.1399** | Artifact 12.93 MB
- **Notes**: CATASTROPHIC. Weights over-smoothed → compressed to only 12.9 MB but int6 quantization destroyed the model. Quant gap 24x worse than Run 4. Aggressive warmdown is incompatible with int6.

## Summary Table
| Run | Config | Pre-quant | Quant gap | Sliding BPB | Artifact |
|-----|--------|-----------|-----------|-------------|----------|
| 4 ⭐ | TTT + FA3 | 1.1417 | +0.0058 | **1.1242** | 15.80 MB |
| 5 | + Mem tokens | 1.1421 | +0.0055 | 1.1244 | 15.82 MB |
| 6 ❌ | + WD=20000 | 1.1503 | +0.1399 | ~1.28 | 12.93 MB |

## Key Learnings
1. **Quant gap is the bottleneck** — 0.006 BPB lost. Reducing this without destroying pre-quant quality is the challenge.
2. **TTT saturating** — ~0.002 BPB gain, loss flat across epochs (1.9398→1.9392).
3. **Memory tokens** — don't survive quantization.
4. **Aggressive warmdown** — destroys quantization. The smoothed weights lose too much information at int6.
5. **What we need**: a way to get better pre-quant BPB WITHOUT worsening the quant gap.

## Confirmed Dead Ends
- ❌ Memory tokens (Run 5: no improvement after quant)
- ❌ Aggressive warmdown=20000 (Run 6: quant gap 24x worse)
- ❌ Depth recurrence (PR #363: 900x quant error amplification)
- ❌ Naive EMA with .cpu().clone() (PR #360: 32% throughput loss)

## Next Experiments (prioritized)
- [ ] **Smaller batch 524K + WD=4000** — more optimizer steps, proven in PR #364
- [ ] **Grad quant (GRAD_QUANT=1)** — smarter bit allocation to reduce quant gap
- [ ] **Aggressive TTT (LR=0.005, epochs=5)** — push TTT harder
- [ ] **Causal TTT** — our novel variant, different adaptation dynamics
- [ ] **Z-loss (ZLOSS_WEIGHT=1e-4)** — may reduce quant gap via logit regularization
- [ ] **Backout connection** — 1 param, zero compute (needs code change)
- [ ] **No Late QAT** — PR #360 suggests it's net negative. Test without.
