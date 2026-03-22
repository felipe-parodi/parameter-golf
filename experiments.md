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

## Ideas for Next Session
- [ ] Check PR #398 review status and feedback
- [ ] Scan for new SOTA (competition moves fast — 20+ PRs/day)
- [ ] Try TTT_EPOCHS=25 (match PR #388, we have eval budget headroom)
- [ ] Implement Shared Value Embeddings (VE128) from PR #374/388
- [ ] Try 12L architecture (NUM_LAYERS=12 MLP_HIDDEN=1408) — more capacity
- [ ] Try cuDNN SDPA instead of FA3 (PR #388 claims 1.18x faster for GQA)
- [ ] GPTQ-lite clip percentile search (PR #379) — zero-cost quant refinement
- [ ] Int5 uniform + 10% pruning (PR #389) — save artifact bytes
- [ ] Consider combining TTT with eval-time bigram cache at very low alpha (0.01-0.05)

## Timeline
- Session 1: 2026-03-21/22 (this session)
- PR #398 submitted: https://github.com/openai/parameter-golf/pull/398
- Competition ends: 2026-04-30 (~5.5 weeks remaining)
- Budget: ~$67 (~12 full runs on 8xH100 SXM)
