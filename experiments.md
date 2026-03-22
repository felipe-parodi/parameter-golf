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
| 9 | Causal TTT | 1.1416 | +0.0077 | 1.1262 | faster but worse |
| 10 | Two-phase TTT | 1.1413 | +0.0080 | 1.1262 | over cap, no gain |
| 11 | Grad quant | 1.1427 | +0.0055 | 1.1250 | over cap, overhead |

## Key Learnings
1. **EMA > Tight SWA** for quant robustness (0.0058 vs 0.0071 gap)
2. **Quant gap (+0.006) is the bottleneck** — all innovations that don't reduce this are wasted
3. **TTT gives ~0.002** on strong baseline, diminishing returns
4. **Memory tokens, aggressive warmdown, smaller batch** — all failed
5. **We're at a ceiling of ~1.124** with the current meta + TTT

## Confirmed Dead Ends
- ❌ Memory tokens — don't survive quant (Run 5)
- ❌ Warmdown=20000 — destroys quant, 24x worse gap (Run 6)
- ❌ Batch 524K — fewer tokens/step not compensated (Run 7)
- ❌ Tight SWA — worse quant gap than EMA (Run 8)
- ❌ Two-phase TTT — phase 2 adds nothing after standard TTT (Run 10)
- ❌ Grad quant — reduces quant gap but adds overhead + over artifact cap (Run 11)
- ❌ Depth recurrence — 900x quant error amplification (PR #363)

## Run 9: Causal TTT (NOVEL) + EMA + FA3 — 8xH100 SXM
- **Config**: Run 4 base + TTT_CAUSAL=1, TTT_CHUNK_TOKENS=16384, TTT_LR=0.003
- **Steps**: 7242, Seed 1337, ~82.9ms/step
- **Results**: Sliding **1.1262** | Roundtrip 1.1493 | Pre-quant 1.1416 | Quant gap +0.0077 | Artifact 15.56 MB | TTT 32.2s
- **What's novel**: Single-pass online TTT. Score each chunk BEFORE updating.
- **Verdict**: Slightly worse than standard TTT (1.1262 vs 1.1242) but 33% faster (32s vs 48s). The gap is mostly training variance, not TTT quality. ≈same

## Run 10: Two-Phase TTT (NOVEL) + EMA + FA3 — 8xH100 SXM ❌
- **Config**: Run 4 base + TTT_CAUSAL=1 (both phases), TTT_CHUNK_TOKENS=16384, TTT_LR=0.003
- **Steps**: 7247, Seed 1337, ~82.8ms/step
- **Results**: Sliding **1.1262** | Roundtrip 1.1493 | Pre-quant 1.1413 | Quant gap +0.0080 | **Artifact 16.15 MB ❌ OVER CAP**
- TTT phase 1: 47.8s | Phase 2 (causal): 32.0s | Total: 79.9s
- **Verdict**: Phase 2 added nothing — same BPB as causal-only (Run 9). Model saturated after standard TTT. Also over 16 MB artifact limit. ❌

## Run 11: Grad Quant + TTT + FA3 — 8xH100 SXM
- **Config**: Run 4 + GRAD_QUANT=1 (adaptive int5/int6/int7 per tensor)
- **Steps**: 7139, Seed 1337, ~84.0ms/step (slower due to grad accumulation)
- **Results**: Sliding **1.1250** | Roundtrip 1.1482 | Pre-quant 1.1427 | Quant gap +0.0055 | **Artifact 16.06 MB ❌ OVER CAP**
- **Quant distribution**: {int5: 13, int6: 47, int7: 6, int8: 2}
- **Verdict**: Quant gap improved (0.0055 vs 0.0058) but overhead cost ~117 steps AND artifact over 16 MB. Net worse. ❌

## Summary After 11 Runs
- **Best**: Run 4 (EMA + TTT) at **1.1242 BPB**
- **Nothing has beaten Run 4** — 7 variations tried, all equal or worse
- **Quant gap reduced** by grad quant (0.0055 vs 0.0058) but offset by overhead
- **Ceiling**: ~1.124 with current 11L/512d meta

## Run 11: Grad Quant + TTT + FA3 — 8xH100 SXM
- **Config**: Run 4 + GRAD_QUANT=1 (adaptive int5/int6/int7 per tensor)
- **Steps**: 7139, Seed 1337, ~84.0ms/step (slower due to grad accumulation overhead)
- **Results**: Sliding **1.1250** | Roundtrip 1.1482 | Pre-quant 1.1427 | Quant gap +0.0055 | **Artifact 16.06 MB ❌ OVER CAP**
- **Quant distribution**: {int5: 13, int6: 47, int7: 6, int8: 2}
- **Verdict**: Quant gap improved (0.0055 vs 0.0058 ✓) but overhead cost ~117 steps AND artifact over 16 MB. The int7 tensors cost more bytes than int5 saves. ❌

## Key Insight
The remaining BPB gains likely come from **eval-time techniques** (not training):
- PPM-C classical probability mixing (PR #283): -0.003 to -0.008 BPB, zero artifact
- Neural Cache cross-window KV (PR #318): extend context to full documents
- These compose with TTT and cost zero training budget
- **PPM-C is now implemented** — ready to test
