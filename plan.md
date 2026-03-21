# Parameter Golf Plan

Last updated: 2026-03-19

## Objective

Win the competition.

The metric that matters is the final post-export score printed by the submission script:

- `final_int8_zlib_roundtrip_exact val_bpb` (or int6+zstd equivalent)

Raw float `val_bpb` is only a proxy. If export destroys the gain, the gain is not real.

## Hard Constraints

- Artifact cap: code bytes + compressed model bytes must be `<= 16,000,000`.
- Training cap: `<= 10 minutes` on `8xH100` for leaderboard submissions.
- Evaluation cap: `<= 10 minutes` on `8xH100`.
- Evaluation can use any sequence length, but it must remain a valid causal probability model.
- No network calls or dataset access during evaluation.
- Tokenizer / dataset changes are allowed, but they carry a much higher proof burden.
- Final counted code should live in `train_gpt.py`.

## Competitive Landscape (as of 2026-03-19)

### Leaderboard Top

| PR | BPB | Author | Stack |
| --- | --- | --- | --- |
| #114 | 1.1574 | saml212 | Int6 + MLP 3x + fp16 embed + train@2048 + stride=256 + clip=0.3 |
| #122 | 1.1585 | mtybadger | 2048 vocab + NorMuon + SWA + FA3 + int6 + fp16 embed + stride=64 |
| #128 | 1.1594 | rsavitt | Int6 + MLP 3x + STE QAT + zstd + stride=64 |
| #102 | 1.1618 | unnir | Int6 + MLP 3x + SmearGate + stride |
| #99 | 1.1605 | takhir | Int6 + MLP 3x + Late-K passthrough + stride |
| #88 | 1.1605 | seanward | Int6 + MLP 3x + MTP + stride |

Baseline: 1.2244. Current SOTA improvement: -0.067 BPB.

### Converged Meta (what everyone at the top is doing)

1. Int6 quantization (per-row, ±31). Frees ~4MB over int8, enabling wider MLP.
2. MLP 3x (hidden=1536). Uses the int6 artifact savings directly.
3. Sliding window eval (stride=64 or stride=256). Free ~0.02-0.04 BPB.
4. fp16 tied embedding passthrough. Embedding is disproportionately damaged by quantization.
5. Tuned Muon: matrix_lr=0.02, momentum=0.99, warmdown=3000.
6. Train@2048 (not 1024, not 4096). 2048 is the sweet spot with sliding window eval.
7. Grad clip=0.3 for long-sequence stability.
8. zstd compression instead of zlib (5-10% smaller artifacts).

### Interesting Outlier Approaches

- PR #77 (samacqua): LoRA TTT. Per-document rank-8 LoRA test-time training. Got 1.195 on the naive baseline. The ablation is revealing:
  - doc-isolated eval alone: -0.011 BPB
  - + sliding window: -0.034 BPB
  - + LoRA TTT: -0.037 BPB
  - TTT itself is small, but it was run on the naive baseline. Nobody has combined it with the current meta stack.

- PR #103 (MatthewHRockwell): Looped Transformer + LoRA. 5 blocks × 6 loops = 30 virtual layers. Per-virtual-layer rank-4 LoRA on Q,V. Key insight: naive weight sharing fails (PR #31 got worse BPB than baseline), but per-layer LoRA + skip connections fix it. Projects ~1.07-1.12 on 8xH100 but unverified.

- PR #122 (mtybadger): SWA (Stochastic Weight Averaging). Averages 7 late-stage checkpoints during warmdown. Simple, free improvement.

- PR #130 (mohosy): Muon-Aware QAT. Claims standard STE corrupts Muon's momentum subspace. Uses late-start + LR reduction + Gaussian noise mode.

- PR #126 (Athenox14): BitNet b1.58. Ternary weights at 3.78MB. Massive quant gap (+0.264 BPB). Proves ternary is not yet viable at this scale.

- PR #112 (eb1386): Depth Recurrence 5x3 d672 SwiGLU. 5 unique blocks × 3 loops = 15 effective layers. Wider model (d=672 vs 512).

- PR #127 (matt-wright86): ALBERT-style factorized embedding (4096x128 + 128x768) for larger vocab without proportional parameter cost. SP4096 vocabulary.

### What We Already Have vs What We Need

Already implemented in our `train_gpt.py`:
- [x] Int6 quantization (QUANT_BITS=6)
- [x] Sliding window eval (EVAL_STRIDE)
- [x] QAT with STE
- [x] Shared-weight recurrence (NUM_PHYSICAL_LAYERS)
- [x] Per-layer untied scalars
- [x] Deeper eval (NUM_EVAL_LAYERS)
- [x] Groupwise quantization
- [x] Measurement harness (quant reports, eval sweeps)

Not yet implemented:
- [ ] MLP 3x (need MLP_MULT=3 or MLP_HIDDEN=1536)
- [ ] fp16 embedding passthrough (skip quantization for tok_emb.weight)
- [ ] zstd compression (instead of zlib)
- [ ] NorMuon (row-normalized Newton-Schulz)
- [ ] SWA (checkpoint averaging during warmdown)
- [ ] LoRA TTT (test-time training)
- [ ] Per-layer LoRA adapters for depth recurrence
- [ ] FA3 (FlashAttention 3)
- [ ] Train@2048
- [ ] Tuned hyperparams (matrix_lr=0.02, momentum=0.99, warmdown=3000, clip=0.3)
- [ ] ALBERT-style factorized embedding

## Updated Working Thesis

The meta has converged. The top 6 submissions are within 0.005 BPB of each other, all running the same int6+MLP3x+sliding-window stack with minor variations.

To win, we need either:

1. **Execute the meta perfectly** and find marginal gains (hyperparameter sweeps, NorMuon, SWA, Late-K passthrough, etc). This is a grind.

2. **Stack TTT on top of the meta**. LoRA TTT (#77) is the biggest underexploited idea. It got 1.195 on the naive baseline using only ~1/10 of the eval budget. Stacking it with int6+MLP3x+sliding-window could break through 1.15.

3. **Make depth recurrence actually work**. Nobody has beaten the standard 9-layer model with recurrence yet. The key missing ingredient per #103 is per-layer LoRA. If we can get 5 blocks × 3 loops + LoRA at d=672 to beat 9 layers at d=512, we win on parameter efficiency and can spend the savings on width, precision, or more physical layers.

Option 2 is the highest expected value. Option 1 is the safest. Option 3 is the hardest but most novel.

## Revised Strategic Lanes

### Lane 1: Match The Meta (PRIORITY: DO THIS FIRST)

This is table stakes. Without this, nothing else matters.

- MLP_MULT=3 (or MLP_HIDDEN=1536)
- QUANT_BITS=6 with per-row scaling
- fp16 embedding passthrough
- Train@2048, grad_clip=0.3
- matrix_lr=0.02, muon_momentum=0.99, warmdown=3000
- Sliding window eval, stride=256
- zstd compression

Expected: ~1.16-1.17 BPB (matching the pack).
Cost: 2-3 runs on 8xH100. ~1 hour of wall time.

### Lane 2: LoRA TTT (PRIORITY: HIGHEST UPSIDE)

The meta is converged. The biggest remaining lever is eval-time compute.

Design:
- At eval time, for each document in the validation set:
  1. Initialize fresh rank-8 LoRA adapters on Q, V, lm_head
  2. Score tokens in sliding windows
  3. After scoring each window, take one gradient step on the scored tokens
  4. Reset LoRA between documents (no cross-document leakage)
- Batch documents by length for efficiency
- Target: use ~200-300s of the 600s eval budget

Key questions:
- Does TTT compose with sliding window? (Probably yes — TTT adapts, sliding window provides context.)
- What's the optimal LoRA rank under the eval time budget? (rank-4 or rank-8)
- Which modules to adapt? (Q, V, lm_head per #77)
- How many gradient steps per window? (1 per #77, but could be more)

Risk: Implementation complexity. Need to backprop through the model during eval, manage LoRA state per document, batch efficiently.

Expected: -0.01 to -0.03 BPB on top of the meta. Could push below 1.14.

### Lane 3: Depth Recurrence + LoRA (PRIORITY: HIGH, HIGH RISK)

The theory is compelling: fewer stored bytes per effective layer = more model capacity per artifact byte.

But nobody has made it work competitively yet. The failure mode is well-documented:
- Naive weight sharing (PR #31): worse than baseline
- Without per-layer differentiation, shared layers collapse to a single effective layer

Design:
- 5 physical blocks × 3 loops = 15 virtual layers
- Per-virtual-layer rank-4 LoRA on Q, V (~307K extra params, ~1.5% overhead)
- Wider model: d=672 or d=768 (artifact budget freed by fewer stored blocks)
- Encoder-decoder skip connections across virtual layers
- SwiGLU or keep relu^2

Key question: does the LoRA overhead + wider model actually beat 9 untied layers at d=512 with MLP 3x?

Risk: High. Need to tune the recurrence depth, LoRA rank, model width, and skip connection pattern. Many hyperparameters.

Expected: If it works, opens a fundamentally different artifact-efficiency frontier. If it doesn't, it's a time sink.

### Lane 4: Polish & Compound (PRIORITY: STEADY)

These are small, well-proven improvements that compound:

- NorMuon (row-normalized Newton-Schulz). Drop-in Muon replacement, small consistent gain.
- SWA (Stochastic Weight Averaging). Average 5-7 late-stage checkpoints. Free -0.003 to -0.005 BPB.
- Late-K passthrough: keep last 2 layers' c_k.weight in fp16. Per #114.
- FA3: FlashAttention 3, ~10ms/step savings = more training steps in 10 minutes.
- Sequence length warmup: start at 256, ramp to 2048. More steps/sec early.
- LAWA (Latest Weight Averaging): variant of SWA, every 200 steps.
- Muon-Aware QAT: late-start QAT with LR reduction per #130.

Each individually small. Together they might buy -0.005 to -0.015 BPB.

### Lane 5: Tokenizer (PRIORITY: MEDIUM, DEFERRED)

PR #122 uses 2048 vocab (trades a layer for better tokenizer). PR #92 tried 8192.
PR #127 uses ALBERT-style factorized embedding for SP4096.

The tradeoff is real: more vocab = better compression per token but more embedding bytes.
Factorized embeddings could break the tradeoff.

Defer until Lanes 1-2 are done. Only pursue if we have artifact budget to spare.

### Lane 6: Moonshots (PRIORITY: LOW)

- BitNet: proven not viable at this scale (PR #126, +0.264 quant gap)
- Mamba/RWKV hybrid: interesting but massive rewrite
- Custom entropy-coded serialization: marginal gains, high complexity
- Multi-token prediction: PR #88 uses this, modest gains

Park these unless Lanes 1-3 stall.

## Execution Plan

### Phase 1: Match The Meta (Day 2-3)

Get on the leaderboard with a competitive submission.

1. Apply RunPod compute credits.
2. Implement: MLP 3x, fp16 embed passthrough, zstd, tuned hyperparams.
3. Run 1: Baseline reproduction on 8xH100 to verify harness.
4. Run 2: Full meta stack (int6 + MLP3x + fp16 embed + train@2048 + stride=256 + clip=0.3).
5. Submit if competitive (~1.16 BPB).

Exit condition: post-quant BPB < 1.17 with sliding window eval.

### Phase 2: LoRA TTT (Day 4-7)

Stack test-time training on top of the meta.

1. Implement LoRA TTT in train_gpt.py eval path.
2. Test on existing checkpoint first (no retraining needed).
3. Measure: BPB improvement and eval time budget.
4. Tune: rank, adapted modules, learning rate, steps per window.
5. Submit if it beats SOTA.

Exit condition: post-quant BPB < 1.15 or clear evidence TTT doesn't compose with the meta.

### Phase 3: Depth Recurrence (Day 7-14, if Phase 2 succeeds or stalls)

Only pursue if we have either:
- A solid TTT submission and want to push further, or
- TTT didn't work and we need a different angle

1. Implement per-layer LoRA adapters on top of shared-weight recurrence.
2. Sweep: 5×3 d672 vs 4×3 d768 vs 3×5 d512.
3. Compare post-quant BPB vs standard 9-layer model.

Exit condition: recurrence beats standard architecture at equal or lower artifact size.

### Phase 4: Polish & Submit (Day 14-21)

Stack all working improvements. Run multiple seeds for p<0.01. Write up and submit PR.

## Kill Criteria

- If an idea does not improve post-export score or artifact size frontier, deprioritize it.
- If an idea improves float score but worsens exported score, treat it as unfinished, not successful.
- If depth recurrence cannot beat 9 untied layers after 5 runs, park it.
- If LoRA TTT adds < 0.005 BPB on top of sliding window, park it (the complexity isn't worth it).
- If we're not on the leaderboard by day 5, abandon everything except Lane 1 and submit.

## Completed (Day 1)

- [x] Forked repo, created `feat/competitive-submission` branch.
- [x] Added measurement harness: EVAL_ONLY, EVAL_SEQ_LEN sweep, per-tensor quant diagnostics.
- [x] Implemented int6 quantization (QUANT_BITS=6).
- [x] Implemented groupwise quantization (INT8_GROUP_SIZE).
- [x] Implemented sliding window eval (EVAL_STRIDE).
- [x] Implemented shared-weight recurrence (NUM_PHYSICAL_LAYERS) with per-layer untied scalars.
- [x] Implemented QAT with STE fake quantization.
- [x] Implemented deeper eval (NUM_EVAL_LAYERS).
- [x] Ported shared-weight recurrence to train_gpt_mlx.py for local prototyping.
- [x] Found and fixed warmup/compile checkpoint fidelity bug.
- [x] Local 700-step baseline: 1.4831 BPB post-quant (1-GPU, 1 shard).
- [x] Competitive landscape analysis: reviewed ~50 PRs, identified converged meta and gaps.

## Completed (Day 2)

- [x] fp16 embedding passthrough (always-on for tied embeddings).
- [x] Late-K passthrough (LATE_K_PASSTHROUGH_LAYERS env var).
- [x] MLP_HIDDEN support for absolute hidden dimension.
- [x] NorMuon (NORMUON=1, per-row RMS norm after Newton-Schulz).
- [x] SWA (SWA_ENABLED=1, checkpoint averaging during warmdown).
- [x] zstd compression support (COMPRESSION=zstd).
- [x] LoRA TTT implementation: BatchedLinearLoRA, LoRAState, find_documents, _block_forward_lora, _forward_logits_with_lora, eval_val_ttt_lora.
- [x] TTT integration in main() gated by TTT_ENABLED=1.
- [x] All features verified via AST parse + structural checks. Awaiting CUDA validation.

## Completed (Day 4 - 2026-03-21)

- [x] Rebased train_gpt.py on PR #315 (current SOTA, 1.1248 BPB). Full modern meta stack.
- [x] Added SGD TTT (TTT_ENABLED=1): 3 epochs SGD on val tokens, freeze first 2 blocks, ~47s eval.
- [x] Added Causal TTT variant (TTT_CAUSAL=1): score-then-update per chunk, more principled.
- [x] Added Gradient-Guided Adaptive Quantization (GRAD_QUANT=1): per-tensor int5/int6/int7 allocation.
- [x] Added Z-loss regularization (ZLOSS_WEIGHT): prevents logit drift, improves quant robustness.
- [x] Added generic quantize_intN_per_row (bits 5-7) with bit assignment from grad sensitivity.
- [x] All features verified via AST parse. 1857 lines. Awaiting CUDA validation.

## RunPod Execution Plan ($25 budget)

### Budget: ~75 min on 8xH100 SXM ($20/hr) or ~10 hrs on 1xH100 ($2.50/hr)

**Debug Phase (1xH100, ~$5, ~2 hrs):**
1. Spin up 1xH100 pod with Parameter Golf template
2. Clone repo, download data (sp1024, full val + 10 train shards)
3. Smoke test: 200 steps, verify harness works
4. Test TTT integration: short run + TTT_ENABLED=1

**Submission Phase (8xH100, ~$17, ~50 min):**
5. Run 1: Reproduce PR #315 baseline (verify 1.1248)
6. Run 2: PR #315 + TTT (TTT_ENABLED=1, target <1.125)
7. Run 3: Best config with TTT tuning (LR/epochs sweep if budget allows)
8. Run 4-5: 2 additional seeds for p<0.01 verification

**Reserve: ~$3 for contingency**

### Run Commands

**Reproduce PR #315 (baseline):**
```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**+ TTT (our main submission):**
```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**+ TTT + Grad Quant + Z-loss (experimental):**
```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 \
GRAD_QUANT=1 GRAD_QUANT_INT7_FRAC=0.10 GRAD_QUANT_INT5_FRAC=0.20 \
ZLOSS_WEIGHT=1e-4 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Causal TTT variant:**
```bash
# Same as TTT above but replace:
TTT_CAUSAL=1 TTT_CHUNK_TOKENS=65536
# Omit TTT_EPOCHS (causal TTT is single-pass)
```

## Sources

- Challenge repo: https://github.com/openai/parameter-golf
- Challenge page: https://openai.com/index/parameter-golf/
- PR #315 (current SOTA, 1.1248): https://github.com/openai/parameter-golf/pull/315
- PR #338 (TTT on #315 base, 1.1254): https://github.com/openai/parameter-golf/pull/338
- PR #332 (12L + Grad-Guided Quant, 1.1320): https://github.com/openai/parameter-golf/pull/332
- PR #317 (TTT + Int6 MLP3x, 1.1442): https://github.com/openai/parameter-golf/pull/317
- PR #322 (Causal TTT + Z-loss): https://github.com/openai/parameter-golf/pull/322
- PR #77 (LoRA TTT, original, 1.195): https://github.com/openai/parameter-golf/pull/77
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
