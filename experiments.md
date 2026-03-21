# Experiment Log

## Run 1: Baseline (FA2) — 8xH100 SXM
- **Config**: PR #315 base, no TTT, FA2 ~116ms/step
- **Steps**: 5162, Seed 1337
- Sliding **1.1366** | Quant gap +0.0066 | Artifact 15.55 MB

## Run 2: TTT (FA2) [INTERRUPTED] — pod killed

## Run 3: TTT — 6xH100 SXM [ABORTED] — wrong template, 440ms/step

## Run 4: EMA + TTT + FA3 — 8xH100 SXM ⭐ BEST
- **Config**: PR #315 base + EMA(0.997) + TTT(lr=0.002,ep=3,freeze=2) + FA3
- **Steps**: 7256, Seed 1337, ~82.7ms/step
- Sliding **1.1242** ⭐ | Roundtrip 1.1475 | Pre-quant 1.1417 | Quant gap +0.0058 | Artifact 15.80 MB

## Run 5: Memory Tokens + EMA + TTT + FA3
- **Config**: Run 4 + NUM_MEMORY_TOKENS=64
- **Steps**: 7183, Seed 1337
- Sliding **1.1244** | Quant gap +0.0055 | Artifact 15.82 MB
- **Verdict**: No improvement after quant. ❌

## Run 6: Aggressive Warmdown + TTT + FA3 ❌
- **Config**: Run 4 but WARMDOWN_ITERS=20000
- **Steps**: 7248, Seed 1337
- Sliding ~1.28 | **Quant gap +0.1399** | Artifact 12.93 MB
- **Verdict**: Catastrophic. Over-smoothed weights destroy int6 quantization. ❌

## Run 7: Smaller Batch 524K + TTT + FA3 ❌
- **Config**: Run 4 but TRAIN_BATCH_TOKENS=524288 WARMDOWN_ITERS=4000
- **Steps**: ~7k, Seed 1337
- 6k val_bpb: 1.2414 (Run 4 was 1.1819 at 6k)
- **Verdict**: Killed early. Way behind Run 4 — fewer tokens/step not compensated. ❌

## Run 8: Tight SWA + TTT + FA3
- **Config**: Run 4 but EMA_ENABLED=0, SWA_ENABLED=1, SWA_THRESHOLD=0.2, SWA_EVERY=50
- **Steps**: 7316, Seed 1337, ~82.0ms/step
- Sliding **1.1249** | Roundtrip 1.1483 | Pre-quant 1.1412 | Quant gap +0.0071 | Artifact 15.52 MB
- **Verdict**: Better pre-quant but worse quant gap than EMA. Net worse. ❌

## Summary Table
| Run | Config | Pre-quant | Quant gap | Sliding BPB | Verdict |
|-----|--------|-----------|-----------|-------------|---------|
| 4 ⭐ | EMA + TTT | 1.1417 | +0.0058 | **1.1242** | BEST |
| 5 | + Mem tokens | 1.1421 | +0.0055 | 1.1244 | ≈same |
| 6 | WD=20000 | 1.1503 | +0.1399 | ~1.28 | ❌ |
| 7 | Batch 524K | — | — | killed | ❌ |
| 8 | Tight SWA | 1.1412 | +0.0071 | 1.1249 | slightly worse |

## Key Learnings
1. **EMA > Tight SWA** for quant robustness (0.0058 vs 0.0071 gap)
2. **Quant gap (+0.006) is the bottleneck** — all innovations that don't reduce this are wasted
3. **TTT gives ~0.002** on strong baseline, diminishing returns
4. **Memory tokens, aggressive warmdown, smaller batch** — all failed
5. **We're at a ceiling of ~1.124** with the current meta + TTT

## Confirmed Dead Ends
- ❌ Memory tokens — don't survive quant
- ❌ Warmdown=20000 — destroys quant (24x worse gap)
- ❌ Batch 524K — fewer tokens/step not compensated
- ❌ Tight SWA — worse quant gap than EMA
- ❌ Depth recurrence — 900x quant error amplification (PR #363)

## Remaining Ideas
- [ ] **Grad quant (GRAD_QUANT=1)** — directly targets the quant gap bottleneck
- [ ] **Z-loss (ZLOSS_WEIGHT=1e-4)** — regularize logits, may help quant
- [ ] **Aggressive TTT (LR=0.005, epochs=5)** — push TTT harder
- [ ] **Causal TTT** — our novel variant
- [ ] **No Late QAT** — test if it's actually helping or hurting
- [ ] **3-seed Run 4 config** — submit as non-record with TTT writeup
