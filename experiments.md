# Experiment Log

## Summary Table
| Run | Config | Pre-quant | Quant gap | Sliding BPB | Artifact | Verdict |
|-----|--------|-----------|-----------|-------------|----------|---------|
| 1 | Baseline FA2 | 1.1534 | +0.0066 | 1.1366 | 15.55 MB | FA2 penalty |
| 4 ⭐ | EMA + TTT(3ep,lr=0.002) + FA3 | 1.1417 | +0.0058 | **1.1242** | 15.80 MB | Best so far |
| 5 | + Memory tokens | 1.1421 | +0.0055 | 1.1244 | 15.82 MB | ≈same |
| 6 | + WD=20000 | 1.1503 | +0.1399 | ~1.28 | 12.93 MB | ❌ catastrophic |
| 7 | + Batch 524K | — | — | killed | — | ❌ way behind |
| 8 | Tight SWA + TTT | 1.1412 | +0.0071 | 1.1249 | 15.52 MB | slightly worse |
| 9 | Causal TTT only | 1.1416 | +0.0077 | 1.1262 | 15.56 MB | faster but worse |
| 10 | Two-phase TTT | 1.1413 | +0.0080 | 1.1262 | 16.15 MB ❌ | over cap |
| 11 | + Grad quant | 1.1427 | +0.0055 | 1.1250 | 16.06 MB ❌ | over cap |
| 12 | + Z-loss + no Late QAT | 1.1443 | +0.0063 | 1.1274 | 15.98 MB | ❌ worse |
| 13 | PPM + TTT (crashed x2) | 1.1415 | +0.0056 | 1.1237 | 15.91 MB | PPM NCCL timeout |

## Current Competition Landscape (2026-03-22)

| PR | BPB | Author | Key Innovation |
|---|---|---|---|
| **#388** | **1.1231** | ElliotSlusky | Tight SWA + VE128 + **TTT(25ep,lr=0.008)** |
| #374 | 1.1246 | unnir | Tight SWA + VE128 (no TTT) |
| #379 | 1.1260 | dannywillowliu | GPTQ-lite + Self-Distillation TTT |
| #315 | 1.1248 | jfprincz | Partial RoPE + LN Scale + EMA + XSA4 |
| **Ours** | **1.1237-1.1242** | — | EMA + TTT(3ep,lr=0.002) + FA3 |
| #383 | 1.1320 | joelnishanth | Tight SWA + Late QAT |

**Key finding: PR #388 uses 25-epoch TTT at lr=0.008 — we use 3 epochs at lr=0.002. We're massively under-tuning TTT.**

## Confirmed Dead Ends (our runs + PR #375's $500 study)
- ❌ Memory tokens — don't survive quant (Run 5, PR #375)
- ❌ Aggressive warmdown=20000 — destroys quant (Run 6)
- ❌ Batch 524K — fewer tokens/step not compensated (Run 7, PR #375)
- ❌ Tight SWA — worse quant gap than EMA (Run 8); though PR #388 uses it
- ❌ Two-phase TTT — phase 2 adds nothing (Run 10)
- ❌ Grad quant — overhead + over artifact cap (Run 11, PR #375)
- ❌ Z-loss — hurts pre-quant quality (Run 12)
- ❌ Causal TTT — marginally worse than standard TTT (Run 9, PR #375)
- ❌ Depth recurrence — 900x quant error amplification (PR #363, #386)
- ❌ Late QAT — may be dead code under torch.compile (PR #315 note)
- ❌ Tokenizer changes — longer tokens harder to predict (PR #384)
- ❌ Multi-token prediction — throughput penalty kills it (PR #375)
- ❌ Self-distillation TTT — slightly negative (PR #379)

## What Actually Works (competition-validated)
- ✅ EMA (0.997) — better than SWA by 0.003 (PR #375 3-seed)
- ✅ TTT with aggressive hyperparams — 25 epochs, lr=0.008 (PR #388)
- ✅ FA3 Hopper — 15-20% more steps (PR #375)
- ✅ 786K batch > 524K (PR #375)
- ✅ Eval-time caching/mixing — -0.003 to -0.005 BPB (PR #384, #387)
- ✅ Int5 uniform + 10% pruning — saves ~1.5MB (PR #389)
- ✅ XSA on last 4-5 layers
- ✅ Partial RoPE (16/64 dims)
- ✅ 1ms overhead = 0.006 BPB cost (PR #375 meta-insight)

## Next Experiment: Aggressive TTT + PPM (Run 14)
- **Config**: Run 4 base + TTT_EPOCHS=25 TTT_LR=0.008 + PPM_ENABLED=1 PPM_ALPHA=0.95 PPM_MAX_ORDER=2
- **Hypothesis**: Aggressive TTT matches PR #388 (1.1231). PPM adds 0.003-0.005 on top → ~1.118-1.120.
- **What's novel**: PPM-C blending is our unique contribution. Nobody else combines classical compression with aggressive TTT.
- **Timing budget**: Training 600s + TTT 25ep ~400s? + PPM precompute ~90s + sliding window 74s. TTT timing is the unknown — 25 epochs is 8x more than our 3. May need to check if it fits in 600s eval.

## Remaining Ideas (if Run 14 works)
- [ ] PPM alpha sweep (0.90, 0.95, 0.99) — tune the blend ratio
- [ ] PPM max_order sweep (1, 2, 3, 4)
- [ ] 3-seed submission runs
- [ ] GPTQ-lite clip percentile search (PR #379) — zero-cost quant refinement
- [ ] Int5 uniform + 10% pruning (PR #389) — save artifact bytes
