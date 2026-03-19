# Parameter Golf Plan

Last updated: 2026-03-18

## Objective

Win the competition.

The metric that matters is the final post-export score printed by the submission script:

- `final_int8_zlib_roundtrip_exact val_bpb`

Raw float `val_bpb` is only a proxy. If export destroys the gain, the gain is not real.

## Hard Constraints

- Artifact cap: code bytes + compressed model bytes must be `<= 16,000,000`.
- Training cap: `<= 10 minutes` on `8xH100` for leaderboard submissions.
- Evaluation cap: `<= 10 minutes` on `8xH100`.
- Evaluation can use any sequence length, but it must remain a valid causal probability model.
- No network calls or dataset access during evaluation.
- Tokenizer / dataset changes are allowed, but they carry a much higher proof burden.
- Final counted code should live in `train_gpt.py`.

## Current Repo Facts

From the included records:

- 10-minute baseline:
  - pre-quant stop metric: `val_bpb 1.2172`
  - final post-quant metric: `val_bpb 1.22436570`
  - total submission size: `15,863,489`
  - slack to cap: `136,511`
  - eval time: about `1.4s`
  - model params: `17,059,912`
- 4-hour non-record baseline:
  - pre-quant stop metric: `val_bpb 1.1749`
  - final post-quant metric: `val_bpb 1.20737944`
  - total submission size: `15,810,161`
  - slack to cap: `189,839`
  - eval time: about `1.36s`
  - same parameter count: `17,059,912`

Implications:

- Export loss is a first-order problem.
- Evaluation-time compute is massively underused.
- Size slack is tiny. We cannot casually add parameters or code.
- A winning run likely needs a better combination of:
  - base model quality
  - export robustness
  - eval-time compute

## Challenge Reading: What Is Explicitly Allowed

The official challenge description explicitly encourages:

- test-time compute
- aggressive parameter tying
- depth recurrence
- low-rank training
- novel tokenizers
- test-time training
- long context
- megakernels

This is the clearest signal from the organizers: weird methods are not a side show here. They are central.

## Public Signal From Other Participants

- PR #10 (`Keep tied embeddings in fp32`) reported a small directional gain with negligible artifact impact. This suggests tiny precision-retention changes can matter if applied to high-leverage tensors.
- PR #5 (`Sparse Attention + Recursive Weight Sharing`) argues for sliding-window attention plus recursive/shared layers as a direct way to increase effective depth under the size cap.
- PR #4 mentions local architecture research on MLX with Muon and muP.

Takeaway:

- Other people are already converging on:
  - precision-sensitive tensor handling
  - weight sharing / recurrence
  - sparse attention

## Public Signal From modded-nanogpt

The speedrun repo is useful because it is a large catalog of "small non-obvious gains compound":

- rotary
- qk norm
- ReLU^2
- Muon
- zero-init projections
- skip variants
- value embeddings
- long-short sliding window attention
- batch / seq schedules
- sparse attention gate
- multi-token prediction
- untie head late in training
- bigram hash embedding
- partitioned hyperconnections

Two notable attempts matter even more for this challenge:

- TokenMonster-based run:
  - reportedly reduced vocab nearly in half while preserving bytes-per-token, saving parameters and FLOPs in embeddings and head
- test-time training / parameter nudging:
  - reportedly improved later-document prediction by doing document-local updates from the earlier prefix

Takeaway:

- Tokenizer co-design is real leverage in parameter-constrained settings.
- Document-local fast adaptation is not science fiction; people already saw signal in adjacent competitions.

## External Compression / Architecture Ideas Worth Stealing

- TokenMonster:
  - strong claim that a smaller vocab can preserve similar compression while freeing embedding/head budget
- QuaRot:
  - rotate hidden states to remove outliers and make low-bit quantization easier
- AQLM:
  - additive quantization / codebooks for much stronger weight compression than naive int8
- BitNet:
  - native low-bit / ternary-style training is possible, though risky for a small repo and short timeline
- Mamba:
  - linear-time recurrent-style blocks are plausible alternatives when eval-time compute or long context matters
- RWKV:
  - proven causal recurrent alternative already explored in the modded-nanogpt ecosystem

Takeaway:

- For this competition, "quantization method" and "architecture" are coupled. A model that is easier to compress may beat a slightly better float model.

## Working Thesis

A plausible winning submission is not just "better training hyperparameters."

It is more likely to be a stack of three things:

1. A more parameter-efficient base model.
2. A much lower export penalty.
3. A stronger evaluator that spends far more of the allowed 10-minute eval budget.

## Ranked Strategic Lanes

### Lane A: Eval-Time Compute

Highest priority.

Why:

- Baseline eval uses about `1.4s` out of `600s`.
- The rules explicitly allow aggressive evaluation methods.
- The metric is document compression, not generation throughput.

Best sub-ideas:

- evaluate at much longer sequence lengths than training
- stateful eval across whole documents
- shared-block model with more recurrent steps at eval than train
- adaptive per-token or per-document extra compute
- per-document fast-weight adaptation on tiny control tensors only

### Lane B: Export / Quantization

Also highest priority.

Why:

- The 4-hour run loses a huge amount when moving from float to exported score.
- That means even a strong base model can fail to convert to leaderboard value.

Best sub-ideas:

- better retention rules for sensitive tensors
- fp32 or fp16 keep rules for tiny high-leverage parameters
- groupwise / codebook / additive quantization
- export-aware fine-tuning or quantization-aware training
- outlier-reduction methods that make the model easier to quantize

### Lane C: Shared-Weight / Recurrent Architecture

Highest priority after instrumentation.

Why:

- This challenge rewards effective capacity per stored byte, not standard transformer orthodoxy.
- Shared physical layers reused over many logical steps are exactly on-target.

Best sub-ideas:

- `NUM_PHYSICAL_LAYERS < NUM_LAYERS`
- untied norms / scalars on top of shared heavy matrices
- train shallow, evaluate deep
- optional adaptive halting / early exit

### Lane D: Sparse / Local Attention

Medium priority.

Why:

- It can cut training compute enough to buy width or more recurrent steps.
- It is especially attractive if paired with shared-depth recurrence.

Best sub-ideas:

- sliding-window attention
- long-short mixed windows
- sparse gating or selective global tokens

### Lane E: Tokenizer-Model Co-Design

Medium priority, high upside, high proof burden.

Why:

- Embeddings and head are expensive in tiny models.
- Smaller vocab may let us reallocate bytes into width, depth, or precision retention.

Best sub-ideas:

- smaller SentencePiece variants first
- TokenMonster-style tokenizer exploration
- byte / hybrid tokenizers only if bpb math is airtight

### Lane F: Distillation

Medium priority.

Why:

- Unlimited offline compute is allowed in spirit if the submission remains fair and reproducible.
- A stronger teacher could improve a tiny student more than manual tuning.

Best sub-ideas:

- teacher logits on held-out training shards
- self-distillation after recurrent-depth or eval-time adaptation ideas stabilize

### Lane G: Moonshots

Low priority for now.

- native ternary / BitNet-style training
- Mamba or RWKV hybrid blocks
- custom entropy-coded serialization formats
- extreme structured sparsity plus recovery

These could win, but they can also absorb the entire schedule.

## First-Principles Questions To Answer Fast

- How much of the score gap is caused by a few tensors vs the whole model?
- Does longer eval sequence length help immediately on the current baseline?
- Can document-local fast adaptation improve score while staying causal and under eval budget?
- Does shared depth beat standard untied depth at fixed artifact size?
- How much vocab can we remove before tokenizer efficiency gives it back?

## Experiment Backlog

| ID | Idea | Why it matters | Cost | Risk | Fast validation |
| --- | --- | --- | --- | --- | --- |
| E0 | Instrument raw-vs-quantized score and per-tensor export stats | Makes all later work measurable | Low | Low | Run baseline + dump tensor size / error report |
| E1 | Add `EVAL_SEQ_LEN` separate from train seq len | Baseline may be underusing context | Low | Low | Sweep eval seq len on existing checkpoint |
| E2 | Stateful / whole-document eval | Uses more of the allowed eval budget | Medium | Medium | Compare bpb with causal state carry |
| E3 | Fast-weight adaptation on control tensors only | Potentially huge eval-time gain at tiny byte cost | Medium | Medium | Update only `skip_weights`, `attn_scale`, `mlp_scale`, `q_gain` on doc prefix |
| E4 | Keep more sensitive tensors at higher precision | Direct export-gap attack | Low | Low | Tensor-by-tensor ablation on current baseline |
| E5 | Groupwise / codebook quantization | Better compression-quality frontier | Medium | Medium | Quantize current checkpoint only, no retraining |
| E6 | Quantization-aware / export-aware fine-tuning | Convert raw gains into final gains | Medium | Medium | Fine-tune from baseline and compare post-export only |
| E7 | Recursive weight tying | More effective depth per byte | Medium | Medium | `NUM_PHYSICAL_LAYERS` sweep under fixed size |
| E8 | Train shallow, evaluate deep | Directly trades eval compute for capacity | Medium | High | Shared-block model with deeper eval unroll |
| E9 | Sliding-window or long-short attention | Could buy width/depth under time cap | Medium | Medium | Measure step time and post-export bpb |
| E10 | Smaller SP vocab sweep | Embedding/head budget lever | Medium | Medium | Re-export 256/512/768/1024 vocab datasets |
| E11 | TokenMonster tokenizer lane | High upside if vocab can shrink hard | High | High | Small proof-of-concept on a subset |
| E12 | Teacher distillation | Better student quality without parameter growth | Medium | Medium | Distill from larger model on a fixed shard subset |
| E13 | RWKV / Mamba hybrid block | Strong parameter efficiency potential | High | High | Tiny non-record prototype only |

## Recommended Execution Order

### Phase 0: Measurement

- Add instrumentation before changing the model.
- We need:
  - pre-quant vs post-quant gap
  - tensor-by-tensor byte counts
  - tensor-by-tensor quantization error
  - eval-time breakdown

Exit condition:

- We can explain where bytes and post-quant loss are going.

### Phase 1: Cheap Eval-Time Compute

- `EVAL_SEQ_LEN` sweep
- stateful eval
- document-local adaptation on control tensors

Exit condition:

- Find out whether eval-time tricks can buy `>= 0.005` to `0.01` bpb on existing checkpoints.

### Phase 2: Export Gap Reduction

- precision retention ablations
- stronger quantization
- export-aware fine-tuning

Exit condition:

- Shrink float-to-export loss significantly on the current baseline family.

### Phase 3: Shared-Depth Architecture

- recursive block reuse
- untied cheap scalars/norms
- eval deeper than train

Exit condition:

- Beat the untied baseline at equal or lower artifact size.

### Phase 4: Tokenizer Lane

- smaller SentencePiece sweep first
- only then consider TokenMonster or more radical tokenizers

Exit condition:

- Show that smaller vocab really improves final post-export `val_bpb`, not just model size.

## Kill Criteria

To avoid burning weeks on pretty ideas:

- If an idea does not improve post-export score or artifact size frontier, deprioritize it.
- If an idea requires a tokenizer change but cannot be justified rigorously, park it.
- If an idea improves float score but worsens exported score, treat it as unfinished, not successful.
- If a moonshot needs a large rewrite before producing a tiny prototype, delay it.

## Immediate Next Actions

- [ ] Add measurement hooks for export diagnostics.
- [ ] Add `EVAL_SEQ_LEN` and a clean eval-only sweep path.
- [ ] Add an experimental eval-time adaptation mode for tiny control tensors.
- [ ] Add a shared-block prototype behind env flags.
- [ ] Keep a rolling spreadsheet or markdown log of every run with:
  - config
  - raw score
  - post-export score
  - artifact size
  - eval time
  - notes

## Sources

- Challenge repo: https://github.com/openai/parameter-golf
- Challenge page: https://openai.com/index/parameter-golf/
- PR #10 (fp32 tied embeddings): https://github.com/openai/parameter-golf/pull/10
- PR #5 (sparse attention + recursive sharing): https://github.com/openai/parameter-golf/pull/5
- PR #4 (MLX local research): https://github.com/openai/parameter-golf/pull/4
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
- TokenMonster: https://github.com/alasdairforsythe/tokenmonster
- QuaRot: https://github.com/spcl/QuaRot
- AQLM: https://github.com/Vahe1994/AQLM
- BitNet: https://github.com/microsoft/BitNet
- Mamba: https://github.com/state-spaces/mamba
- RWKV fork of modded-nanogpt: https://github.com/BlinkDL/modded-nanogpt-rwkv
