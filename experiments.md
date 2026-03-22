# Experiment Log

## Summary Table
| Run | Config | Pre-quant | Quant gap | Sliding BPB | Artifact | Verdict |
|-----|--------|-----------|-----------|-------------|----------|---------|
| 1 | Baseline FA2 | 1.1534 | +0.0066 | 1.1366 | 15.55 MB | FA2 penalty |
| 4 | EMA + TTT(3ep,0.002,frz=2) + FA3 | 1.1417 | +0.0058 | 1.1242 | 15.80 MB | prev best |
| 5 | + Memory tokens | 1.1421 | +0.0055 | 1.1244 | 15.82 MB | ≈same |
| 6 | + WD=20000 | 1.1503 | +0.1399 | ~1.28 | 12.93 MB | ❌ catastrophic |
| 7 | + Batch 524K | — | — | killed | — | ❌ way behind |
| 8 | Tight SWA + TTT | 1.1412 | +0.0071 | 1.1249 | 15.52 MB | slightly worse |
| 9 | Causal TTT only | 1.1416 | +0.0077 | 1.1262 | 15.56 MB | faster but worse |
| 10 | Two-phase TTT | 1.1413 | +0.0080 | 1.1262 | 16.15 MB ❌ | over cap |
| 11 | + Grad quant | 1.1427 | +0.0055 | 1.1250 | 16.06 MB ❌ | over cap |
| 12 | Z-loss + no Late QAT | 1.1443 | +0.0063 | 1.1274 | 15.98 MB | worse |
| 14 | TTT(20ep,0.008,frz=2)+PPM | 1.1413 | +0.0301 | 1.1488 | — | ❌ catastrophic |
| 15 ⭐ | TTT(20ep,0.008,**frz=0**)+noQAT+noXSA | 1.1418 | +0.0028 | **1.1213** | 15.53 MB | **SUBMITTED** |

## Final 3-Seed Results (Run 15 config — PR #398)
| Seed | Steps | Step avg | Pre-quant | Roundtrip | Sliding BPB | Artifact |
|------|-------|----------|-----------|-----------|-------------|----------|
| 1337 | 7386 | 81.2ms | 1.1418 | 1.1446 | **1.1213** | 15.53 MB |
| 42 | 7411 | 81.0ms | 1.1426 | 1.1454 | 1.1221 | 15.51 MB |
| 2025 | 7386 | 81.2ms | 1.1418 | 1.1461 | 1.1228 | 15.53 MB |
| **Mean** | | | | | **1.1221** | |
| **Std** | | | | | **0.0008** | |

## Winning Config
```bash
SEED=1337 NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=0 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=0 \
TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=20 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Findings (ranked by importance)

### 1. FREEZE_BLOCKS=0 is the single biggest lever
With aggressive TTT (20ep, lr=0.008), freezing early blocks is catastrophic:
- freeze=2: quant gap +0.030, sliding 1.1488 ❌
- freeze=0: quant gap +0.003, sliding 1.1213 ✅
All layers must adapt coherently during TTT. Partial freezing creates internal inconsistency.

### 2. Late QAT must be disabled with aggressive TTT
PR #388 noted this too. QAT makes weights unfriendly for subsequent TTT adaptation.

### 3. XSA removal saves steps → better BPB
Without XSA: ~81ms/step (7386 steps). With XSA: ~83ms/step (7256 steps).
130 extra steps × better convergence > XSA quality benefit.

### 4. EMA beats Tight SWA for quant robustness
EMA quant gap: +0.006. Tight SWA quant gap: +0.007. Consistent across runs.

### 5. Aggressive TTT (20ep) >> Conservative TTT (3ep)
TTT loss drop: 0.007 (20ep) vs 0.002 (3ep). The model keeps learning through epoch 20.
But only works with freeze=0.

### 6. PPM-C eval-time blending hurts strong models
On weak baseline (smoke, 200 steps): -0.11 BPB improvement.
On strong baseline (1.12): +0.014 BPB WORSE. The neural model already captures bigram patterns.

### 7. Memory tokens, grad quant, z-loss — all dead ends on strong baselines
Confirmed by both our experiments and PR #375's $500 systematic study.

### 8. 1ms/step overhead = 0.006 BPB cost
This heuristic from PR #375 held true in all our runs. Any technique that adds per-step overhead must justify itself against this cost.

## Confirmed Dead Ends
| Technique | Run(s) | Why it failed |
|-----------|--------|---------------|
| Memory tokens | 5 | Don't survive int6 quantization |
| Warmdown=20000 | 6 | Over-smooths weights, 24x worse quant gap |
| Batch 524K | 7 | Fewer tokens/step not compensated by more steps |
| Tight SWA | 8 | Worse quant gap than EMA |
| Two-phase TTT | 10 | Phase 2 adds nothing after standard TTT |
| Grad-guided quant | 11 | Overhead + artifact over 16 MB |
| Z-loss | 12 | Hurts pre-quant quality |
| TTT with freeze=2 at high LR | 14 | Internal inconsistency, catastrophic quant gap |
| PPM-C eval blending | 14,15 | Hurts strong models, adds noise |
| Depth recurrence | PR #363 | 900x quant error amplification |
| Late QAT | PR #360 | Net negative under 10-min budget |
| Tokenizer changes | PR #384 | Longer tokens harder to predict |
| Multi-token prediction | PR #375 | Throughput penalty kills it |
| Self-distillation TTT | PR #379 | Slightly negative |
| Causal TTT | 9, PR #375 | Neutral or worse on strong baselines |

## Operational Lessons

### Pod Setup
- **ALWAYS use the Parameter Golf template** (`runpod/parameter-golf:latest`). Non-template pods have wrong PyTorch → torch.compile breaks, flash_attn dtype errors. Wasted 1 full run cycle learning this.
- **FA3 via pre-built wheel** (instant): `pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291`. Building from source wastes 10+ min.
- **Install numba** for CPU-bound eval loops (50-100x speedup). PPM went from 338s → 1.5s.
- **Use tmux** — terminal disconnects kill running jobs.
- **Smoke test first** (ITERATIONS=200, MAX_WALLCLOCK_SECONDS=60) — we wasted 3 full runs on features that crashed during eval.
- **Use setup_pod.sh** for reproducible setup in one command.

### Budget
- 8xH100 SXM on-demand: ~$21.50/hr
- Each full run (train + eval): ~15 min = ~$5.40
- Data download + FA3 install + setup: ~5 min
- Session 1 total: ~$33 for 15 runs (including wasted runs on wrong pods)
- **Remaining: ~$67**

### Workflow
- **Check new PRs FIRST** before running anything. We could have matched PR #388's config from the start if we'd seen their FREEZE_BLOCKS=0 finding.
- **Log everything** — exact commands, timestamps, results. This notebook saved us from repeating failed experiments.
- **`runpodctl send/receive`** for file transfer between pod and local machine.
- **landscape.md** tracks the full competitive field — update at start of each session.

## CRITICAL UPDATE: TTT "Not in Spirit" (2026-03-22)

From Issue #140 (live AI commentary) and organizer @cocohearts on PR #317:
- **TTT is "not in the spirit of the challenge"**
- **PR #388 (25-epoch TTT) was CLOSED** — pre-eval TTT ruled invalid
- Our PR #398 is flagged with ⚠️ for TTT usage
- 5 TTT variants confirmed dead at frontier by PR #375's $500 study
- **Best non-TTT validated: PR #315 at 1.1250 BPB**

This means: **our TTT-based 1.1221 may not be accepted as a record.** We need a non-TTT path.

### Our Position Without TTT
Run 15 pre-quant: 1.1418 → post-quant roundtrip: 1.1446 (before TTT)
Sliding window without TTT would be approximately: **~1.125** (estimated from roundtrip - sliding gain of ~0.020)
This matches #315's 1.1250. We are NOT ahead without TTT.

### Non-TTT Path: What Could Beat 1.1250?

**Highest priority (from Issue #140's untried combos):**
1. **OptRot pre-quantization** (arXiv:2512.24124) — rotation matrix redistributes weight outliers before int6. Reduces quant gap 30-50%. Zero artifact cost. Drop-in. Est. -0.002 to -0.005 BPB.
2. **FP8 forward pass** (torchao) — H100 does 2x TFLOPS in FP8 E4M3 vs BF16. 20-40% more training steps. Systems-only (significance waived). Est. -0.003 to -0.008 BPB.
3. **2:4 structured sparsity** — relu² is already 84-98% sparse; enforce NVIDIA 2:4 pattern for 2x sparse matmul. ~15-20% more steps. Systems-only. Est. -0.003 to -0.008 BPP.
4. **Liger-Kernel fused ops** — fused RMSNorm (6x), fused CE (1.7x), pip-installable. 20-43% throughput. Systems-only. Est. -0.002 to -0.006 BPB.
5. **DyT (Dynamic Tanh)** (arXiv:2503.10622) — replace RMSNorm with tanh(α·x). Saves 1-2ms/step. 1-line change. Est. -0.001 to -0.004 BPB.
6. **Mousse optimizer** (arXiv:2603.09697) — curvature-aware Muon. 12% more effective at 3% overhead. Drop-in. Est. -0.003 to -0.008 BPB.
7. **VE128 (Shared Value Embeddings)** — from PR #374. Novel architecture component.
8. **GPTQ-lite** (PR #379) — per-layer optimal clip search. Zero training cost.

**Key insight from #140:** "Each 1ms step overhead = 0.006 BPB cost." Systems optimizations (FP8, fused kernels, sparsity) that increase throughput are the highest-EV path at the frontier. Significance test is WAIVED for systems-only changes.

## Ideas for Session 2 (Non-TTT Focus)
- [ ] **OptRot pre-quantization** — directly reduces quant gap, zero artifact cost
- [ ] **Liger-Kernel fused ops** — pip install, 20-43% throughput
- [ ] **DyT (Dynamic Tanh)** — 1-line RMSNorm replacement, saves 1-2ms/step
- [ ] **FP8 forward pass** — 2x TFLOPS, 20-40% more steps
- [ ] **Mousse optimizer** — drop-in Muon replacement
- [ ] **VE128** — shared value embeddings (PR #374)
- [ ] **GPTQ-lite clip search** — zero-cost quant refinement
- [ ] Request additional compute from OpenAI ($1M pool)

## Timeline
- Session 1: 2026-03-21/22 — 15 runs, submitted PR #398 (1.1221, TTT-based)
- **Session 2 priority: non-TTT submission beating 1.1250**
- Competition ends: 2026-04-30 (~5.5 weeks remaining)
- Budget: ~$67 remaining + potential OpenAI compute grant
