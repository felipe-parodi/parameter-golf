"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", os.environ.get("TRAIN_SEQ_LEN", 1024)))
    eval_seq_len_sweep = os.environ.get("EVAL_SEQ_LEN_SWEEP", "")
    eval_only = bool(int(os.environ.get("EVAL_ONLY", "0")))
    load_path = os.environ.get("LOAD_PATH", "")
    quant_report_topk = int(os.environ.get("QUANT_REPORT_TOPK", 50))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 0))  # 0=disabled (non-overlapping), e.g. 64 or 256
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 256))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    num_physical_layers = int(os.environ.get("NUM_PHYSICAL_LAYERS", 0))  # 0=disabled (use num_layers unique blocks)
    num_eval_layers = int(os.environ.get("NUM_EVAL_LAYERS", 0))  # 0=use num_layers at eval too

    # QAT (Quantization-Aware Training).
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.85))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_all_validation_tokens(pattern: str) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    return torch.cat([load_data_shard(file) for file in files]).contiguous()


def prepare_validation_tokens(tokens: Tensor, seq_len: int) -> Tensor:
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for EVAL_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    return prepare_validation_tokens(load_all_validation_tokens(pattern), seq_len)


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    eval_seq_len: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    num_layers: int | None = None,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < eval_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, EVAL_SEQ_LEN={eval_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // eval_seq_len
    total_seqs = (val_tokens.numel() - 1) // eval_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * eval_seq_len
            raw_end = batch_seq_end * eval_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, eval_seq_len)
            y = local[1:].reshape(-1, eval_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = (model(x, y, num_layers=num_layers) if num_layers is not None else model(x, y)).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,q_gains,"
        "skip_weight,skip_weights,layer_attn_scales,layer_mlp_scales,layer_resid_mixes,layer_q_gains",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
INT8_GROUP_SIZE = int(os.environ.get("INT8_GROUP_SIZE", 0))  # 0=disabled, e.g. 128
INT8_GROUP_MIN_COLS = int(os.environ.get("INT8_GROUP_MIN_COLS", 256))  # only groupwise if cols >= this
QUANT_BITS = int(os.environ.get("QUANT_BITS", 8))  # 6 for int6 (±31), 8 for int8 (±127)
QUANT_MAX_VAL = (2 ** (QUANT_BITS - 1)) - 1  # 31 for int6, 127 for int8

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor, dict[str, object] | None]:
    """Quantize a float tensor. Supports int6 (QUANT_BITS=6) and int8 (default)."""
    mv = float(QUANT_MAX_VAL)  # 31 for int6, 127 for int8
    t32 = t.float()
    if t32.ndim == 2:
        rows, cols = t32.shape
        # Groupwise quantization: one scale per group of G elements within each row.
        if INT8_GROUP_SIZE > 0 and cols >= INT8_GROUP_MIN_COLS and cols % INT8_GROUP_SIZE == 0:
            gs = INT8_GROUP_SIZE
            num_groups = cols // gs
            reshaped = t32.reshape(rows, num_groups, gs)
            clip_abs = (
                torch.quantile(reshaped.abs(), INT8_CLIP_Q, dim=2)
                if reshaped.numel()
                else torch.empty((rows, num_groups), dtype=torch.float32)
            )
            scale = (clip_abs / mv).clamp_min(1.0 / mv)
            clipped = torch.clamp(reshaped, -clip_abs[..., None], clip_abs[..., None])
            q = torch.clamp(torch.round(clipped / scale[..., None]), -mv, mv).to(torch.int8)
            q = q.reshape(rows, cols).contiguous()
            scale = scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
            return q, scale, {"scheme": "per_group", "axis": 0, "group_size": gs, "bits": QUANT_BITS}

        # Per-row quantization: one scale per row.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / mv).clamp_min(1.0 / mv)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -mv, mv).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(), {"scheme": "per_row", "axis": 0, "bits": QUANT_BITS}

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / mv if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -mv, mv).to(torch.int8).contiguous()
    return q, scale, None

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s, qm = quantize_float_tensor(t)
        if qm is not None:
            qmeta[name] = qm
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def _dequant_tensor(q: Tensor, s: Tensor, meta: dict[str, object], dtype: torch.dtype) -> Tensor:
    """Dequantize a single int8 tensor given its scales and qmeta."""
    scheme = meta.get("scheme", "")
    if scheme == "per_group":
        gs = int(meta["group_size"])
        rows, cols = q.shape
        num_groups = cols // gs
        s32 = s.to(dtype=torch.float32)  # (rows, num_groups)
        q_grouped = q.float().reshape(rows, num_groups, gs)
        return (q_grouped * s32[..., None]).reshape(rows, cols).to(dtype=dtype).contiguous()
    if scheme == "per_row" or s.ndim > 0:
        s = s.to(dtype=torch.float32)
        return (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
    scale = float(s.item())
    return (q.float() * scale).to(dtype=dtype).contiguous()


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        out[name] = _dequant_tensor(q, s, qmeta.get(name, {}), dtype)
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out
def tensor_error_stats(original: Tensor, restored: Tensor) -> dict[str, float | None]:
    if not original.is_floating_point() or original.numel() == 0:
        return {"mean_abs_error": 0.0, "max_abs_error": 0.0, "rel_l2_error": 0.0}
    orig32 = original.detach().float()
    restored32 = restored.detach().float()
    diff = restored32 - orig32
    diff_abs = diff.abs()
    return {
        "mean_abs_error": float(diff_abs.mean().item()),
        "max_abs_error": float(diff_abs.max().item()),
        "rel_l2_error": float(diff.norm().item()) / max(float(orig32.norm().item()), 1e-12),
    }
def tensor_dtype_name(t: Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")
def dtype_name_nbytes(dtype_name: str) -> int:
    return int(torch.empty((), dtype=getattr(torch, dtype_name)).element_size())
def quant_report_entry(
    *, name: str, tensor: Tensor, stored_bytes: int, scheme: str, stored_dtype: str, error_stats: dict[str, float | None]
) -> dict[str, object]:
    return {
        "name": name, "shape": list(tensor.shape), "numel": int(tensor.numel()), "dtype": tensor_dtype_name(tensor),
        "raw_bytes": tensor_nbytes(tensor), "exported_bytes": int(stored_bytes), "scheme": scheme, "stored_dtype": stored_dtype, **error_stats,
    }
def _scheme_label(meta: dict[str, object], s: Tensor) -> str:
    scheme = meta.get("scheme", "")
    bits = int(meta.get("bits", 8))
    tag = f"int{bits}"
    if scheme == "per_group":
        return f"{tag}_per_group_g{int(meta['group_size'])}"
    if scheme == "per_row" or s.ndim > 0:
        return f"{tag}_per_row"
    return f"{tag}_per_tensor"

def build_quant_report_from_quant_obj(obj: dict[str, object]) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        s = obj["scales"][name]
        orig_dtype = obj["dtypes"][name]
        meta = qmeta.get(name, {})
        report.append({
            "name": name, "shape": list(q.shape), "numel": int(q.numel()), "dtype": orig_dtype,
            "raw_bytes": int(q.numel()) * dtype_name_nbytes(orig_dtype), "exported_bytes": tensor_nbytes(q) + tensor_nbytes(s),
            "scheme": _scheme_label(meta, s),
            "stored_dtype": f"{tensor_dtype_name(q)}+{tensor_dtype_name(s)}_scale",
            "mean_abs_error": None, "max_abs_error": None, "rel_l2_error": None,
        })
    for name, t in obj["passthrough"].items():
        orig_dtype = passthrough_orig_dtypes.get(name, tensor_dtype_name(t))
        report.append({
            "name": name, "shape": list(t.shape), "numel": int(t.numel()), "dtype": orig_dtype,
            "raw_bytes": int(t.numel()) * dtype_name_nbytes(orig_dtype), "exported_bytes": tensor_nbytes(t),
            "scheme": "passthrough_downcast" if t.is_floating_point() and orig_dtype != tensor_dtype_name(t) else "passthrough_exact",
            "stored_dtype": tensor_dtype_name(t), "mean_abs_error": None, "max_abs_error": None, "rel_l2_error": None,
        })
    return sorted(report, key=lambda item: (-int(item["exported_bytes"]), str(item["name"])))
def quantize_state_dict_int8_with_report(state_dict: dict[str, Tensor]):
    obj, stats = quantize_state_dict_int8(state_dict)
    report: list[dict[str, object]] = []
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if name in obj["quantized"]:
            q = obj["quantized"][name]
            s = obj["scales"][name]
            dtype = getattr(torch, obj["dtypes"][name])
            meta = qmeta.get(name, {})
            restored = _dequant_tensor(q, s, meta, dtype)
            report.append(quant_report_entry(
                name=name, tensor=t, stored_bytes=tensor_nbytes(q) + tensor_nbytes(s),
                scheme=_scheme_label(meta, s),
                stored_dtype=f"{tensor_dtype_name(q)}+{tensor_dtype_name(s)}_scale", error_stats=tensor_error_stats(t, restored),
            ))
            continue
        kept = obj["passthrough"][name]
        orig_dtype = passthrough_orig_dtypes.get(name)
        restored = kept.to(dtype=getattr(torch, orig_dtype)).contiguous() if isinstance(orig_dtype, str) else kept
        report.append(quant_report_entry(
            name=name, tensor=t, stored_bytes=tensor_nbytes(kept),
            scheme="passthrough_downcast" if kept.is_floating_point() and isinstance(orig_dtype, str) and orig_dtype != tensor_dtype_name(kept) else "passthrough_exact",
            stored_dtype=tensor_dtype_name(kept), error_stats=tensor_error_stats(t, restored),
        ))
    report.sort(key=lambda item: (-int(item["exported_bytes"]), str(item["name"])))
    return obj, stats, report
def quant_report_payload(
    *, args: Hyperparameters, quant_stats: dict[str, int], quant_raw_bytes: int | None, quant_file_bytes: int | None,
    tensor_report: list[dict[str, object]], code_bytes: int, checkpoint_source: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": args.run_id, "checkpoint_source": checkpoint_source, "train_seq_len": args.train_seq_len,
        "eval_seq_len": args.eval_seq_len, "quant_stats": {key: int(value) for key, value in quant_stats.items()},
        "code_bytes": int(code_bytes), "topk_printed": int(args.quant_report_topk), "tensors": tensor_report,
    }
    if quant_raw_bytes is not None:
        payload["quant_raw_bytes"] = int(quant_raw_bytes)
    if quant_file_bytes is not None:
        payload["quant_file_bytes"] = int(quant_file_bytes)
        payload["total_submission_bytes"] = int(quant_file_bytes + code_bytes)
    return payload
def write_json_artifact(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def summarize_quant_report(report: list[dict[str, object]]) -> dict[str, int]:
    num_float_tensors = sum(1 for item in report if str(item["dtype"]).startswith(("float", "bfloat")))
    return {
        "param_count": sum(int(item["numel"]) for item in report), "num_tensors": len(report),
        "num_float_tensors": num_float_tensors, "num_nonfloat_tensors": len(report) - num_float_tensors,
        "baseline_tensor_bytes": sum(int(item["raw_bytes"]) for item in report),
        "int8_payload_bytes": sum(int(item["exported_bytes"]) for item in report),
    }
def log_quant_report(log_fn, tensor_report: list[dict[str, object]], topk: int) -> None:
    for item in tensor_report[:topk]:
        mae = item["mean_abs_error"]
        max_abs = item["max_abs_error"]
        rel_l2 = item["rel_l2_error"]
        log_fn(
            f"quant_tensor name:{item['name']} shape:{item['shape']} dtype:{item['dtype']} "
            f"scheme:{item['scheme']} stored_dtype:{item['stored_dtype']} "
            f"raw_bytes:{item['raw_bytes']} exported_bytes:{item['exported_bytes']} "
            f"mae:{'n/a' if mae is None else f'{float(mae):.6g}'} "
            f"max_abs:{'n/a' if max_abs is None else f'{float(max_abs):.6g}'} "
            f"rel_l2:{'n/a' if rel_l2 is None else f'{float(rel_l2):.6g}'}"
        )
def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 256,
    num_layers: int | None = None,
) -> tuple[float, float]:
    """Sliding window eval: each token scored with near-maximum context.

    Windows of train_seq_len advance by `stride`. Only the last `stride` tokens
    per window contribute to the score (first window scores all).
    """
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride]
    total_windows = len(window_starts)

    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = (model.forward_logits(x_batch, num_layers=num_layers)
                          if num_layers is not None else model.forward_logits(x_batch))

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else wlen - stride
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    model.train()
    return float(val_loss), float(bits_per_token * tokens_per_byte)


def parse_int_csv(raw: str) -> list[int]:
    return [int(text) for part in raw.split(",") if (text := part.strip())] if raw.strip() else []
def run_eval_pass(
    *, args: Hyperparameters, model: nn.Module, rank: int, world_size: int, device: torch.device, grad_accum_steps: int,
    full_val_tokens: Tensor, eval_seq_len: int, base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    num_layers: int | None = None,
) -> tuple[float, float, float]:
    val_tokens = prepare_validation_tokens(full_val_tokens, eval_seq_len)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps, val_tokens, eval_seq_len,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, num_layers=num_layers,
    )
    torch.cuda.synchronize()
    return val_loss, val_bpb, 1000.0 * (time.perf_counter() - t0)


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.int32, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


_qat_active = False  # global flag toggled by training loop

def _fake_quantize_per_row(w: Tensor) -> Tensor:
    """STE fake-quantize: simulate per-row quant in the forward pass (respects QUANT_BITS)."""
    mv = float(QUANT_MAX_VAL)
    w32 = w.float()
    abs_max = w32.abs().amax(dim=1, keepdim=True).clamp_min(1.0 / mv)
    scale = abs_max / mv
    q = torch.clamp(torch.round(w32 / scale), -mv, mv)
    # STE: detach rounding error so gradients pass through linearly
    return ((q - w32 / scale).detach() + w32 / scale) * scale


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if _qat_active and self.weight.ndim == 2 and self.weight.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            w = _fake_quantize_per_row(w).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, q_gain: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        _qg = q_gain if q_gain is not None else self.q_gain
        q = q * _qg.to(dtype=q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            # Repeat KV heads for GQA (manual expansion for older torch without enable_gqa)
            reps = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        attn_scale: Tensor | None = None,
        mlp_scale: Tensor | None = None,
        resid_mix: Tensor | None = None,
        q_gain: Tensor | None = None,
    ) -> Tensor:
        _mix = (resid_mix if resid_mix is not None else self.resid_mix).to(dtype=x.dtype)
        x = _mix[0][None, None, :] * x + _mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), q_gain=q_gain)
        _as = (attn_scale if attn_scale is not None else self.attn_scale).to(dtype=x.dtype)
        x = x + _as[None, None, :] * attn_out
        _ms = (mlp_scale if mlp_scale is not None else self.mlp_scale).to(dtype=x.dtype)
        x = x + _ms[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        num_physical_layers: int = 0,
        num_eval_layers: int = 0,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # Shared-weight recurrence mode: num_physical_layers < num_layers
        self.shared_mode = num_physical_layers > 0
        self.num_physical_layers = num_physical_layers if self.shared_mode else num_layers
        self.num_eval_layers = num_eval_layers if num_eval_layers > 0 else num_layers

        if self.shared_mode:
            # Shared mode: fewer physical blocks, batched per-layer scalars on GPT.
            # No U-Net skip connections (resid_mix already blends with x0).
            self.skip_weights = None
            self.num_encoder_layers = 0
            self.num_decoder_layers = 0
            self.blocks = nn.ModuleList(
                [
                    Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
                    for _ in range(self.num_physical_layers)
                ]
            )
            # Per-logical-layer untied scalars (cheap, stored as passthrough in export)
            self.layer_attn_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))
            self.layer_mlp_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))
            self.layer_resid_mixes = nn.Parameter(
                torch.stack([torch.stack((torch.ones(model_dim), torch.zeros(model_dim))) for _ in range(num_layers)]).float()
            )
            self.layer_q_gains = nn.Parameter(torch.full((num_layers, num_heads), qk_gain_init, dtype=torch.float32))
        else:
            # Standard mode: unique blocks with U-Net skip connections.
            self.num_encoder_layers = num_layers // 2
            self.num_decoder_layers = num_layers - self.num_encoder_layers
            self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
            self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
            self.layer_attn_scales = None
            self.layer_mlp_scales = None
            self.layer_resid_mixes = None
            self.layer_q_gains = None
            self.blocks = nn.ModuleList(
                [
                    Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
                    for _ in range(num_layers)
                ]
            )

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor, num_layers: int | None = None) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        if self.shared_mode:
            # Shared recurrence: cycle through physical blocks, use per-layer scalars.
            n = num_layers if num_layers is not None else self.num_layers
            for i in range(n):
                phys_idx = i % self.num_physical_layers
                scalar_idx = i % self.num_layers  # cyclic reuse for eval layers beyond training
                x = self.blocks[phys_idx](
                    x, x0,
                    attn_scale=self.layer_attn_scales[scalar_idx],
                    mlp_scale=self.layer_mlp_scales[scalar_idx],
                    resid_mix=self.layer_resid_mixes[scalar_idx],
                    q_gain=self.layer_q_gains[scalar_idx],
                )
        else:
            # Standard U-Net path with skip connections.
            skips: list[Tensor] = []
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor, num_layers: int | None = None) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        if self.shared_mode:
            n = num_layers if num_layers is not None else self.num_layers
            for i in range(n):
                phys_idx = i % self.num_physical_layers
                scalar_idx = i % self.num_layers
                x = self.blocks[phys_idx](
                    x, x0,
                    attn_scale=self.layer_attn_scales[scalar_idx],
                    mlp_scale=self.layer_mlp_scales[scalar_idx],
                    resid_mix=self.layer_resid_mixes[scalar_idx],
                    q_gain=self.layer_q_gains[scalar_idx],
                )
        else:
            skips: list[Tensor] = []
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    eval_seq_lens = parse_int_csv(args.eval_seq_len_sweep) or [args.eval_seq_len]
    if any(seq_len <= 0 for seq_len in eval_seq_lens):
        raise ValueError(f"EVAL_SEQ_LEN/EVAL_SEQ_LEN_SWEEP must be positive, got {eval_seq_lens}")
    if args.eval_seq_len_sweep:
        eval_seq_lens = list(dict.fromkeys(eval_seq_lens))
    args.eval_seq_len = eval_seq_lens[0]
    if sys.platform != "win32":
        try:
            zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
        except RuntimeError:
            pass

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        if sys.platform == "win32":
            # Windows torch builds lack flash/cudnn kernels; must enable math SDP
            enable_math_sdp(True)
            enable_mem_efficient_sdp(True)
        else:
            enable_cudnn_sdp(False)
            enable_flash_sdp(True)
            enable_mem_efficient_sdp(False)
            enable_math_sdp(False)
    except (ImportError, AttributeError):
        pass  # Older torch versions don't have these

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    full_val_tokens = load_all_validation_tokens(args.val_files)
    default_val_tokens = prepare_validation_tokens(full_val_tokens, args.eval_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(
        f"val_loader:shards pattern={args.val_files} tokens:{default_val_tokens.numel() - 1} "
        f"eval_seq_len:{args.eval_seq_len}"
    )
    if len(eval_seq_lens) > 1:
        log0(f"eval_seq_len_sweep:{','.join(str(seq_len) for seq_len in eval_seq_lens)}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        num_physical_layers=args.num_physical_layers,
        num_eval_layers=args.num_eval_layers,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    def build_train_model() -> nn.Module:
        if sys.platform != "win32":
            try:
                compiled = torch.compile(base_model, dynamic=False, fullgraph=True)
            except RuntimeError:
                compiled = base_model
        else:
            compiled = base_model  # Windows: no triton/inductor
        return DDP(compiled, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled

    model: nn.Module = build_train_model()

    n_params = sum(p.numel() for p in base_model.parameters())
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    if base_model.shared_mode:
        log0(
            f"shared_mode:True num_physical_layers:{base_model.num_physical_layers} "
            f"num_train_layers:{args.num_layers} num_eval_layers:{base_model.num_eval_layers}"
        )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"eval_seq_len:{args.eval_seq_len} iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    code_bytes = len(code.encode("utf-8"))

    # Use deeper eval unroll in shared mode when evaluating the final model
    _eval_num_layers = base_model.num_eval_layers if base_model.shared_mode else None

    def evaluate_sequence_lengths(
        label: str,
        eval_model: nn.Module | None = None,
        use_eval_depth: bool = True,
    ) -> list[dict[str, float | int | str]]:
        results: list[dict[str, float | int | str]] = []
        active_model = base_model if eval_model is None else eval_model
        nl = _eval_num_layers if use_eval_depth else None
        for seq_len in eval_seq_lens:
            val_loss, val_bpb, eval_ms = run_eval_pass(
                args=args,
                model=active_model,
                rank=rank,
                world_size=world_size,
                device=device,
                grad_accum_steps=grad_accum_steps,
                full_val_tokens=full_val_tokens,
                eval_seq_len=seq_len,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
                num_layers=nl,
            )
            log0(f"{label} eval_seq_len:{seq_len} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} eval_time:{eval_ms:.0f}ms")
            results.append(
                {
                    "label": label,
                    "eval_seq_len": seq_len,
                    "val_loss": float(val_loss),
                    "val_bpb": float(val_bpb),
                    "eval_time_ms": float(eval_ms),
                }
            )
        return results

    def log_quant_gap(float_results: list[dict[str, float | int | str]], quant_results: list[dict[str, float | int | str]]) -> None:
        if not float_results or not quant_results:
            return
        quant_by_seq = {int(item["eval_seq_len"]): item for item in quant_results}
        for item in float_results:
            seq_len = int(item["eval_seq_len"])
            quant_item = quant_by_seq.get(seq_len)
            if quant_item is None:
                continue
            loss_gap = float(quant_item["val_loss"]) - float(item["val_loss"])
            bpb_gap = float(quant_item["val_bpb"]) - float(item["val_bpb"])
            log0(
                f"quant_gap eval_seq_len:{seq_len} "
                f"delta_val_loss:{loss_gap:+.6f} delta_val_bpb:{bpb_gap:+.6f}"
            )

    def emit_quant_report(
        *,
        checkpoint_source: str | None,
        quant_stats: dict[str, int],
        tensor_report: list[dict[str, object]],
        quant_raw_bytes: int | None,
        quant_file_bytes: int | None,
    ) -> None:
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"quant_report_summary payload:{quant_stats['int8_payload_bytes']} "
            f"baseline_tensor_bytes:{quant_stats['baseline_tensor_bytes']} payload_ratio:{ratio:.2f}x"
        )
        if quant_raw_bytes is not None:
            log0(f"quant_report_raw_bytes:{quant_raw_bytes}")
        if quant_file_bytes is not None:
            log0(f"quant_report_file_bytes:{quant_file_bytes} total_submission_bytes:{quant_file_bytes + code_bytes}")
        log_quant_report(log0, tensor_report, args.quant_report_topk)
        if master_process:
            quant_report_path = f"quant_report_{args.run_id}.json"
            write_json_artifact(
                quant_report_path,
                quant_report_payload(
                    args=args,
                    quant_stats=quant_stats,
                    quant_raw_bytes=quant_raw_bytes,
                    quant_file_bytes=quant_file_bytes,
                    tensor_report=tensor_report,
                    code_bytes=code_bytes,
                    checkpoint_source=checkpoint_source,
                ),
            )
            log0(f"quant_report_path:{quant_report_path}")

    def emit_eval_sweep_artifact(
        *,
        checkpoint_source: str | None,
        float_results: list[dict[str, float | int | str]],
        quant_results: list[dict[str, float | int | str]],
    ) -> None:
        if not master_process or len(eval_seq_lens) <= 1:
            return
        eval_sweep_path = f"eval_sweep_{args.run_id}.json"
        write_json_artifact(
            eval_sweep_path,
            {
                "run_id": args.run_id,
                "checkpoint_source": checkpoint_source,
                "float_results": float_results,
                "quantized_results": quant_results,
            },
        )
        log0(f"eval_sweep_path:{eval_sweep_path}")

    if args.eval_only:
        if not args.load_path:
            raise ValueError("LOAD_PATH is required when EVAL_ONLY=1")
        load_path = Path(args.load_path).expanduser().resolve()
        if not load_path.is_file():
            raise FileNotFoundError(load_path)
        checkpoint_source = str(load_path)
        log0(f"eval_only:1 load_path:{checkpoint_source}")

        float_results: list[dict[str, float | int | str]] = []
        quant_results: list[dict[str, float | int | str]] = []

        if load_path.suffix == ".pt":
            state_dict = torch.load(load_path, map_location="cpu")
            base_model.load_state_dict(state_dict, strict=True)
            float_results = evaluate_sequence_lengths("float_eval")

            quant_obj, quant_stats, tensor_report = quantize_state_dict_int8_with_report(base_model.state_dict())
            quant_buf = io.BytesIO()
            torch.save(quant_obj, quant_buf)
            quant_raw = quant_buf.getvalue()
            quant_blob = zlib.compress(quant_raw, level=9)
            emit_quant_report(
                checkpoint_source=checkpoint_source,
                quant_stats=quant_stats,
                tensor_report=tensor_report,
                quant_raw_bytes=len(quant_raw),
                quant_file_bytes=len(quant_blob),
            )
            base_model.load_state_dict(dequantize_state_dict_int8(quant_obj), strict=True)
            quant_results = evaluate_sequence_lengths("final_int8_zlib_roundtrip")
            log_quant_gap(float_results, quant_results)
        elif load_path.suffixes[-2:] == [".int8", ".ptz"] or load_path.name.endswith(".int8.ptz"):
            quant_blob_disk = load_path.read_bytes()
            quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
            tensor_report = build_quant_report_from_quant_obj(quant_state)
            quant_stats = summarize_quant_report(tensor_report)
            emit_quant_report(
                checkpoint_source=checkpoint_source,
                quant_stats=quant_stats,
                tensor_report=tensor_report,
                quant_raw_bytes=len(zlib.decompress(quant_blob_disk)),
                quant_file_bytes=len(quant_blob_disk),
            )
            base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
            quant_results = evaluate_sequence_lengths("final_int8_zlib_roundtrip")
        else:
            raise ValueError(f"Unsupported LOAD_PATH for EVAL_ONLY=1: {load_path}")

        emit_eval_sweep_artifact(
            checkpoint_source=checkpoint_source,
            float_results=float_results,
            quant_results=quant_results,
        )
        if distributed:
            dist.destroy_process_group()
        return

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.shared_mode:
        # In shared mode, per-layer scalars live on GPT, not on blocks.
        # Block-level scalars (attn_scale, mlp_scale, resid_mix, q_gain) are dead in shared mode;
        # the GPT-level batched tensors are the real trainable parameters.
        scalar_params = [
            base_model.layer_attn_scales,
            base_model.layer_mlp_scales,
            base_model.layer_resid_mixes,
            base_model.layer_q_gains,
        ]
    elif base_model.skip_weights is not None and base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        torch._dynamo.reset()
        model = build_train_model()

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                base_model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                default_val_tokens,
                args.eval_seq_len,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} eval_seq_len:{args.eval_seq_len} "
                f"val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Activate QAT (fake quantization in CastedLinear.forward) in late training
        global _qat_active
        if args.qat_enabled and not _qat_active and step >= int(args.iterations * args.qat_start_frac):
            _qat_active = True
            log0(f"qat:activated at step:{step}")

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    # Deactivate QAT for evaluation
    _qat_active = False

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    float_results = evaluate_sequence_lengths("final_float")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats, tensor_report = quantize_state_dict_int8_with_report(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    quant_file_bytes = len(quant_blob)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
    emit_quant_report(
        checkpoint_source=None,
        quant_stats=quant_stats,
        tensor_report=tensor_report,
        quant_raw_bytes=quant_raw_bytes,
        quant_file_bytes=quant_file_bytes,
    )
    log0(
        f"Serialized model int8+zlib: {quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} "
        f"payload_ratio:{quant_stats['baseline_tensor_bytes'] / max(quant_stats['int8_payload_bytes'], 1):.2f}x)"
    )
    log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    quant_results = evaluate_sequence_lengths("final_int8_zlib_roundtrip")
    if len(quant_results) == 1:
        q_item = quant_results[0]
        log0(
            f"final_int8_zlib_roundtrip_exact eval_seq_len:{q_item['eval_seq_len']} "
            f"val_loss:{float(q_item['val_loss']):.8f} val_bpb:{float(q_item['val_bpb']):.8f}"
        )
    log_quant_gap(float_results, quant_results)

    # Sliding window eval on the quantized model (the actual competition metric)
    if args.eval_stride > 0:
        log0(f"sliding_window_eval stride:{args.eval_stride} batch_seqs:{args.eval_batch_seqs}")
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        slide_nl = _eval_num_layers  # deeper eval if shared mode
        slide_loss, slide_bpb = eval_val_sliding(
            args, base_model, rank, world_size, device,
            full_val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, batch_seqs=args.eval_batch_seqs, num_layers=slide_nl,
        )
        torch.cuda.synchronize()
        slide_ms = 1000.0 * (time.perf_counter() - t_slide)
        log0(
            f"final_int8_zlib_sliding val_loss:{slide_loss:.8f} val_bpb:{slide_bpb:.8f} "
            f"eval_time:{slide_ms:.0f}ms"
        )

    emit_eval_sweep_artifact(
        checkpoint_source=None,
        float_results=float_results,
        quant_results=quant_results,
    )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
