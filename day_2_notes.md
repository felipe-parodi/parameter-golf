# Day 2 Notes - 2026-03-19

## Objective

Match the competitive meta and implement LoRA TTT as our differentiator.

## Competitive Landscape Analysis

Reviewed ~50 PRs on the upstream repo. Key findings:

### Leaderboard (as of 2026-03-19)

| PR | BPB | Author | Stack |
| --- | --- | --- | --- |
| #114 | 1.1574 | saml212 | Int6 + MLP 3x + fp16 embed + train@2048 + stride=256 + clip=0.3 |
| #122 | 1.1585 | mtybadger | 2048 vocab + NorMuon + SWA + FA3 + int6 + fp16 embed |
| #128 | 1.1594 | rsavitt | Int6 + MLP 3x + STE QAT + zstd |
| #102 | 1.1618 | unnir | Int6 + MLP 3x + SmearGate |
| #99 | 1.1605 | takhir | Int6 + MLP 3x + Late-K passthrough |
| #88 | 1.1605 | seanward | Int6 + MLP 3x + MTP |

### Converged Meta

All top submissions run the same stack:
1. Int6 quantization (per-row, ±31)
2. MLP 3x (hidden=1536) — uses int6 artifact savings
3. Sliding window eval (stride=64 or 256)
4. fp16 tied embedding passthrough
5. Tuned Muon: matrix_lr=0.02, momentum=0.99, warmdown=3000
6. Train@2048, grad_clip=0.3

### Biggest Underexploited Idea

PR #77 (samacqua): LoRA TTT (test-time training). Per-document rank-8 LoRA adaptation at eval time. Got 1.195 BPB on the naive baseline using ~1/10 of eval budget. Nobody has combined it with the meta stack yet.

## Code Changes

### MLX Script: Shared-Weight Recurrence Port

Ported shared-weight recurrence from `train_gpt.py` to `train_gpt_mlx.py`:
- Added `NUM_PHYSICAL_LAYERS` and `NUM_EVAL_LAYERS` hyperparameters
- Added q_gain override to `CausalSelfAttention.__call__`
- Added scalar overrides to `Block.__call__`
- Rewrote `GPT` to support both standard (U-Net skips) and shared mode (cycled blocks + per-layer scalars)
- Updated `SplitOptimizers` to route GPT-level scalars to Adam in shared mode
- Added `compiled_eval_loss` for deeper eval in shared mode
- Verified: standard mode (17M params), shared mode 3×9 (6M params), forward pass, optimizer, quantization roundtrip all pass

### CUDA Script: Meta + LoRA TTT

Added to `train_gpt.py` (1759 → 2115 lines):

**Part 1 — Match The Meta:**

1. fp16 embedding passthrough: `tok_emb.weight` skips quantization, stored as fp16. Always on.
2. Late-K passthrough: `LATE_K_PASSTHROUGH_LAYERS=N` keeps last N layers' c_k.weight in fp16.
3. MLP_HIDDEN: `MLP_HIDDEN=1536` for absolute hidden dimension (vs MLP_MULT multiplier).
4. NorMuon: `NORMUON=1` adds per-row RMS normalization after Newton-Schulz iteration. Uses global flag pattern (like `_qat_active`).
5. SWA: `SWA_ENABLED=1` collects model snapshots during warmdown (`scale < 1.0`), averages them before final eval/export. Running sum in float32.
6. zstd: `COMPRESSION=zstd` replaces zlib with zstandard. Helper functions `compress_blob()` / `decompress_blob()`. Conditional import.

**Part 2 — LoRA TTT (~200 lines):**

- `BatchedLinearLoRA`: Per-batch-element rank-r adapter. A (kaiming), B (zero). Forward: `bmm(bmm(x, A^T), B^T)`.
- `LoRAState`: Creates adapters for c_q, c_v (all physical blocks) + lm_head. Adam optimizer (lr=0.01). Reset between document batches.
- `find_documents()`: Scans for BOS_ID=1 boundaries. Returns (start, length) list.
- `_block_forward_lora()`: Inlines one block forward with LoRA on Q, V. Supports shared-mode scalar overrides.
- `_forward_logits_with_lora()`: Full forward returning logits. Both standard and shared mode.
- `eval_val_ttt_lora()`: Main loop. Sort docs by length, shard across GPUs, batch 64 docs, chunk-based score-before-train. ~294s estimated eval time.
- Integration: gated by `TTT_ENABLED=1`, runs after quantized roundtrip eval.

All features backward compatible, togglable via env vars.

## Local Setup

- Created `.venv` with MLX deps + 1 training shard + full validation split
- Applied for quick-start compute credits ($25 / ~8 compute hours)
- MLX smoke tests killed early — full 62M-token validation is too slow on Mac. Use `VAL_BATCH_SIZE>=131072`.

## Verified

- `train_gpt_mlx.py`: Shared mode model init, forward pass (both depths), optimizer, quantization roundtrip. All pass.
- `train_gpt.py`: AST parse clean. All new classes/functions/hyperparameters present. Line count 2115.
- Cannot verify CUDA execution locally — needs RunPod.

## Key Decisions

- fp16 embed passthrough is always-on (no env var gate). Every top submission does this.
- LoRA TTT targets c_q, c_v, lm_head (per PR #77). Not c_k, not MLP.
- TTT adapters are per-physical-block in shared mode (same delta across all reuses of a block).
- Score-before-train: chunk loss is accumulated (detached) before backward/step.

## Risks

1. LoRA TTT memory: 64 docs × rank-8 LoRA across 9 blocks + lm_head + Adam state ≈ ~170MB. Should fit on H100.
2. TTT eval time: estimated ~294s on 8xH100. Tight. May need to reduce `TTT_DOC_BATCH_SIZE` or increase `TTT_CHUNK_SIZE`.
3. TTT + shared mode interaction: LoRA on physical blocks means the delta compounds across recurrent passes. This might help (persistent per-doc knowledge) or hurt (noise amplification). Needs empirical testing.

## Next Steps

1. Get RunPod compute credits.
2. Reproduce baseline on 8xH100: `torchrun --standalone --nproc_per_node=8 train_gpt.py` with default config.
3. Run meta stack: `MLP_HIDDEN=1536 QUANT_BITS=6 TRAIN_SEQ_LEN=2048 EVAL_STRIDE=256 MATRIX_LR=0.02 MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 TRAIN_BATCH_TOKENS=786432 NORMUON=1 SWA_ENABLED=1 LATE_K_PASSTHROUGH_LAYERS=2`. Target: ~1.16-1.17 BPB.
4. Run TTT on top: `TTT_ENABLED=1`. Target: < 1.15 BPB.
5. If TTT works, submit PR.
