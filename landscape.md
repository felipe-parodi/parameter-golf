# Parameter Golf Competitive Landscape

Last updated: 2026-03-22

## Leaderboard (Unmerged Open PRs)

| PR | BPB | Author | Key Stack | Date |
|---|---|---|---|---|
| #388 | 1.1231 | ElliotSlusky | Tight SWA + VE128 + TTT(25ep,lr=0.008,freeze=0) + cuDNN SDPA | 03-21 |
| #374 | 1.1246 | unnir | Tight SWA + VE128 + Partial RoPE + LN Scale + XSA4 (no TTT) | 03-21 |
| #315 | 1.1248 | jfprincz | Partial RoPE + LN Scale + EMA + Late QAT + XSA4 + FA3 | 03-21 |
| #338 | 1.1254 | alertcat | PR #315 base + TTT(3ep,lr=0.002,freeze=2) | 03-21 |
| #379 | 1.1260 | dannywillowliu | GPTQ-lite + Self-Distillation TTT | 03-21 |
| #332 | 1.1320 | saml212 | 12L + Gradient-Guided Quant + Partial RoPE + LN Scale | 03-21 |
| #383 | 1.1320 | joelnishanth | Tight SWA + XSA4 + Late QAT | 03-21 |
| #369 | 1.1328 | signalrush | FA3 + Batch524K + XSA4 + EMA | 03-21 |
| #359 | 1.1345 | tmustier | FA3 + SwiGLU + XSA + EMA + BigramHash | 03-21 |
| #349 | 1.1399 | Mapika | XSA + EMA + Int5-MLP | 03-21 |
| #389 | 1.1466 | trasnake87 | Int5-All + XSA5 + EMA + 10% Pruning | 03-22 |
| #385 | 1.1488 | dentity007 | Int6 QAT + SmearGate + SWA(0.4) | 03-22 |

Official merged leaderboard tops out at 1.1428 (PR #180).

## The Converged Meta (what everyone in the top 10 uses)

### Architecture (11L standard)
- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA)
- MLP 3x (hidden=1536), relu-squared activation
- Tied embeddings, vocab 1024
- U-Net skip connections (encoder-decoder with skip weights)
- SmearGate (learned token mixing)
- BigramHash(2048, dim=128) (hash-based bigram embeddings)
- OrthoInit (orthogonal weight initialization, projection scaled by 1/sqrt(2*num_layers))
- Logit softcap = 30

### Training
- Muon optimizer (matrix params) + AdamW (scalar/embed params)
- matrix_lr=0.025, scalar_lr=0.025, tied_embed_lr=0.035
- Muon momentum warmup: 0.92 → 0.99 over 1500 steps
- Muon WD=0.04, Adam WD=0.04
- Train seq_len=2048, batch_tokens=786432
- Grad clip = 0.3
- Warmdown 3000 steps (cosine decay in final phase)
- 20 warmup steps (compile warmup, then restore weights)
- ~7000-7300 steps in 600s wallclock

### Quantization
- Int6 per-row quantization (±31) for MLP + attention weights
- FP16 passthrough for embeddings and small tensors
- zstd-22 compression
- Artifact target: ~15.5-15.9 MB (cap: 16.0 MB)

### Eval
- Sliding window eval, stride=64, seq_len=2048
- Each token scored with maximum available context

## Technique Catalog

### Proven Winners (validated in top submissions)

**EMA (Exponential Moving Average)** — PR #315, #338, #349
- Decay=0.997, every step, GPU in-place update
- Better than SWA by ~0.003 BPB (PR #375 3-seed verified)
- Must avoid .cpu().clone() (PR #360: 32% throughput loss)

**Tight SWA** — PR #374, #383, #388
- Collect checkpoints only when lr_scale < 0.2 (~last 600 steps)
- Average ~12-16 checkpoints every 50 steps
- Zero quality penalty from averaging (post-SWA BPB = pre-SWA BPB)
- Weights still smoother for quantization
- PR #388 claims works better than EMA with aggressive TTT

**XSA (Exclusive Self Attention)** — PR #315, #374, #349
- Remove self-value bias from attention output via orthogonal projection
- Applied to last 4-5 layers
- Forces attention to carry cross-token information only
- PR #388 DISABLES this (says too slow without FA3)

**Partial RoPE** — PR #315, #374, #388
- Rotary embeddings on only 16/64 head dims
- Remaining 75% of dims use position-free attention
- Improves generalization. Zero parameters.

**LN Scale** — PR #315, #374, #388
- RMSNorm output scaled by 1/sqrt(layer_idx + 1)
- Damps deeper layers' contributions, stabilizes training
- Zero parameters.

**TTT (Test-Time Training)** — PR #338, #388, #371, #317
- Full-weight SGD on validation tokens after quantization
- **Conservative**: 3 epochs, lr=0.002, freeze first 2 blocks (PR #338: +0.002 BPB)
- **Aggressive**: 25 epochs, lr=0.008, freeze 0 blocks (PR #388: +0.015 BPP)
- Must freeze 0 blocks for aggressive TTT (freezing 2 causes internal inconsistency)
- Late QAT must be disabled with aggressive TTT
- ~16s per epoch on 8xH100

**FlashAttention 3 (Hopper)** — PR #315, #369
- ~85ms/step vs ~115ms with FA2 (30% faster)
- More steps in 600s budget = lower BPB
- Install: `pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291`

**Late QAT** — PR #315 (but disputed)
- STE int6 fake-quantization in final ~4% of training
- PR #315 note: may be dead code under torch.compile (constant folds _qat_enabled)
- PR #388 DISABLES it (says catastrophic with SWA/aggressive TTT)
- PR #360: costs 8% of training steps, net negative

### Promising but Unproven

**Shared Value Embeddings (VE128)** — PR #374, #388
- Single learned embedding table (dim=128) shared across layers 9-10
- Per-layer learned scaling factors
- Saves parameters while adding cross-layer information sharing
- Used in both #374 (1.1246) and #388 (1.1231)

**GPTQ-lite** — PR #379
- Per-layer optimal clip percentile search during int6 quantization
- 5 clip ratios per weight matrix, pick minimum reconstruction error
- Zero training cost. Post-training quant refinement.

**Eval-time Caching/Mixing** — PR #384, #387
- Unigram cache + OGD on vocab bias: -0.003 BPB (PR #384)
- Bigram cache with entropy-gated mixing: -0.005 BPB (PR #387, alpha=0.20)
- Zero artifact cost, eval-time only

**PPM-C (Prediction by Partial Matching)** — PR #283
- Classical compression blended with neural predictions
- -0.069 BPB on weak baseline (1.2244), ~0.003-0.008 on strong
- Zero artifact cost, eval-time only

**Int5 Uniform + 10% Pruning** — PR #389
- All weights at int5 (not just MLP), saves ~1.5MB
- 10% magnitude pruning before quant (zeros compress well in zstd)

**Backout Connection** — PR #339, #366
- Learned residual subtraction at U-Net midpoint
- 1 scalar parameter, zero compute

### Confirmed Dead Ends

**Depth Recurrence / Weight Sharing** — PR #363, #386
- Quantization error amplifies ~900x over 3 recurrence cycles
- Int6 gap goes from 0.006 to 1.14 BPB. Completely unviable.

**Memory Tokens** — PR #352, PR #375
- Learnable embedding vectors at start of each sequence
- Don't survive int6 quantization

**Aggressive Warmdown (20000)** — our Run 6
- Over-smooths weights → int6 quantization destroys the model
- Quant gap 24x worse

**Smaller Batch (524K)** — PR #375, our Run 7
- 786K > 524K: total tokens seen > gradient update count

**Multi-Token Prediction** — PR #375
- +0.028 BPB from throughput penalty

**Tokenizer Changes** — PR #384
- Longer merged tokens harder to predict per-token, offsetting compression

**Self-Distillation TTT** — PR #379
- KL-divergence from frozen teacher. Slightly negative (-0.0003).

**Causal TTT** — PR #375, our Run 9
- Score-then-update per chunk. Neutral or slightly worse than standard TTT on strong baselines.

**Label Smoothing, L1 Reg, Cautious WD, Canon Layers, 1M Batch** — PR #375
- All negative on strong baselines.

## Meta-Insights

1. **1ms/step overhead = 0.006 BPB cost** (PR #375). Most techniques fail this throughput test.
2. **Quant gap (~0.006) is a hard bottleneck.** All top submissions lose ~0.006 BPB to int6.
3. **Sliding window stride=64 gives ~0.023 BPB** over non-overlapping eval.
4. **TTT gives 0.002-0.015 BPB** depending on aggressiveness. The biggest single eval-time lever.
5. **The meta is converged.** Top 10 all run the same 11L stack. Differentiation comes from eval-time techniques (TTT, caching) and quant tricks.
6. **EMA vs Tight SWA**: EMA is safer, Tight SWA works better with aggressive TTT.
7. **Late QAT is controversial**: may be dead code under torch.compile, catastrophic with SWA/TTT.

## Architecture Variants (non-standard)

**12L with narrower MLP** — PR #332
- 12 layers, MLP_HIDDEN=1408 (instead of 1536)
- Gradient-guided quant (int5/6/7) saves ~1MB for the extra layer
- Got 1.1320 without Partial RoPE/LN Scale

**Hybrid Depth-Recurrent** — PR #341
- 1 unique entry + 4 shared × 5 loops + 1 unique exit = 22 effective layers
- Unique entry/exit protect against quant amplification
- Preliminary: 1.3323 on 2xH100

**INL BetaMu Attention** — PR #377
- Error-driven (x - mu) replaces QKV, O(n) via causal cumsum
- No attention matrix at all. 1.41-1.46 BPB.

## CRITICAL: TTT Ruling (Issue #402, filed 2026-03-22)

Issue #402 challenges the validity of epoch-based TTT (including our PR #398). Argument: full TTT adaptation on all eval tokens before scoring = training on eval set. Only token-by-token backward-looking TTT (score token t, then adapt on tokens ≤ t) may be valid. **No official ruling yet** but organizer @cocohearts already said TTT is "not in spirit" on PR #317.

**Our PR #398 is explicitly listed as potentially invalid in Issue #402.**

## Updated Leaderboard (as of Issue #140 update, Mar 22 7:05 PM PT)

### Non-TTT (safe for record track)
| PR | BPB | Author | Key Innovation |
|---|---|---|---|
| #414 | **1.1233** (3-seed) | signalrush | **GPTQ-lite** + EMA + Tight SWA + VE128 + XSA4 + warmdown3500 |
| #445 | 1.1236 (2-seed) | newjordan | #414 base + "TTT Burst" (training data replay, claims non-TTT) |
| #401 | 1.1243 (1-seed) | newjordan | EMA + Tight SWA stacking + Late QAT@0.15 + VE128 |
| #374 | 1.1246 (1-seed) | unnir | Tight SWA + VE128 + Partial RoPE + LN Scale |
| #315 | 1.1250 (3-seed) | jfprincz | Partial RoPE + LN Scale + EMA + XSA4 (Late QAT was dead code!) |

### TTT-based (validity disputed, Issue #402)
| PR | BPB | Author | Key Innovation |
|---|---|---|---|
| #490 | **1.0891** (1-seed!) | amaljithkuttamath | Value Residual + Gated Attention + AdamW TTT |
| #481 | 1.0970 (3-seed) | mrdavtan | Cosine TTT + per-layer LR (3x MLP output proj) |
| #442 | 1.1027 (3-seed) | sjp611 | AdamW TTT 10ep (3-line diff from our #398) |
| #486 | 1.1132 (3-seed) | ndokutovich | TrigramHash + Value Residual + GradQuant + TTT |
| #473 | 1.1219 (3-seed) | abaybektursun | Legal score-first TTT + Parallel Muon on #414 |
| #398 | 1.1221 (3-seed) | **us** | EMA + TTT(20ep,freeze=0) — converted to non-record |

### Key intel from Issue #140 (live commentary)
- **Value Residual on #414 base (no TTT) is listed as Tier 2 highest-value untried combo.** Est. -0.005 to -0.010 BPB → ~1.113-1.118.
- **This is exactly what we're doing now** (+ TrigramHash on top).
- Late QAT in PR #315 was **dead code** (torch.compile constant-folds _qat_enabled). Many submissions claiming Late QAT benefit may not actually be running it.
- PR #490 (1.0891) shows Value Residual + Gated Attention is massive with TTT.
- PR #445 "TTT Burst" replays TRAINING data (not val) → may be legal non-TTT technique.

## New Techniques Since Session 1 (PRs #390-444)

### Must-Adopt (proven, low-risk)
- **GPTQ-lite** (PR #414): per-row clip percentile search during int6 quant. 5 candidates per row, pick min MSE. Zero training cost. -0.0006 BPB. NEW NON-TTT SOTA.
- **Value Residual / ResFormer** (PR #413, arXiv:2410.17897): cache V from layer 0, mix into all layers via 18 learnable scalars. -0.015 BPB. Trivial to add.
- **Gated Attention** (PR #413, arXiv:2505.06708): per-head sigmoid gate after SDPA. Eliminates attention sinks. ~37K params. -0.003 BPB.
- **EMA+SWA stacking** (PR #401): run EMA every step, SWA collects from EMA weights. Best of both.

### Promising (novel, higher complexity)
- **Dynamic Eval** (PR #397, Krause 2018): SGD steps DURING sliding window scoring, not before. Model adapts to local text as it evaluates. -0.024 BPB. May survive TTT ruling since it scores-before-adapts per token.
- **Two-Phase TTT** (PR #417): Phase 1 = norm-only recalibration (50ep Adam, 22K params). Phase 2 = selective freeze (10ep SGD, last 3 blocks). More defensible TTT.
- **CANON-AC DeltaGate** (PR #400): learned sigmoid-gated conv on last 5 blocks. 1.1296.
- **TrigramHash + context gating** (PR #418): extends BigramHash with trigrams + hidden-state-modulated gate.
- **Parameter Banking + Parallel Muon** (PR #399): batched Newton-Schulz (4 bmm ops vs 66 sequential). 81.87ms/step.

### Nobody Has Implemented Yet (our differentiation)
- FP8 E4M3 forward pass
- Liger-Kernel fused Triton ops
- 2:4 structured activation sparsity
- OptRot pre-quantization rotation
- Mousse optimizer
- DyT (Dynamic Tanh normalization)

## Key PRs to Watch
- PR #414 (non-TTT SOTA, 1.1233) — GPTQ-lite
- PR #442 (1.1027 claimed) — AdamW TTT, needs verification
- PR #417 (two-phase TTT, 1.1222) — may survive TTT ruling
- PR #397 (dynamic eval, -0.024 BPB) — scores-then-adapts, different from epoch TTT
- PR #413 (value residual + gated attention) — cheap architectural wins
- PR #402 (TTT validity challenge) — directly affects our PR #398
- PR #375 ($500 systematic negatives) — reference for what not to try

## Competition Timeline
- Started: March 18, 2026
- Session 1: March 21-22 (15 runs, PR #398 submitted, ~$33 spent)
- Ends: April 30, 2026 (~5.5 weeks remaining)
- Budget: ~$67 remaining + pending compute grant request
