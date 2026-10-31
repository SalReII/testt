"""Q/K/V assembly for attention: LoRA add + rotary embedding + scatter to the padded layout.

Base Soup gets q, k and v from bitsandbytes as fp32 (the input was the fp32 norm output), adds
the LoRA outputs, transposes, applies the rotary embedding with seven PyTorch ops per tensor in
fp32 (two muls, a slice-negate-concat, an add), and SDPA's autocast then casts q, k and v to
fp16 again.

Here the fused q/k/v GEMM output (fp16, unpadded tokens) is read once per token by one Triton
program that adds the LoRA outputs in fp32, rotates q and k in fp32 and writes fp16 q/k/v
directly into zero-initialised [batch, heads, seq, head_dim] tensors at the token's padded
position, the layout SDPA takes. Rounding happens once, where base Soup's autocast rounds.
Backward is the transposed rotation, one program per token, writing the fp16 gradient of the
q/k/v GEMM output; the LoRA branches get views of the same buffer (in base Soup both receive
the same fp16 cast of the fp32 gradient).
"""

import torch
import triton
import triton.language as tl

from ._common import next_pow2


@triton.jit(do_not_specialize=["T", "has_lq", "has_lv"])
def rope_pack_fwd_kernel(qkv_ptr, lq_ptr, lv_ptr, cos_ptr, sin_ptr, pos_ptr, q_ptr, k_ptr, v_ptr,
                         T, has_lq, has_lv,
                         NH: tl.constexpr, NKV: tl.constexpr, HALF: tl.constexpr,
                         NH_P: tl.constexpr, NKV_P: tl.constexpr):
    tok = tl.program_id(0).to(tl.int64)
    pos = tl.load(pos_ptr + tok).to(tl.int64)
    b = pos // T
    t = pos % T
    D = 2 * HALF
    W = (NH + 2 * NKV) * D
    half = tl.arange(0, HALF)[None, :]
    c1 = tl.load(cos_ptr + t * D + half)
    c2 = tl.load(cos_ptr + t * D + HALF + half)
    s1 = tl.load(sin_ptr + t * D + half)
    s2 = tl.load(sin_ptr + t * D + HALF + half)
    row = qkv_ptr + tok * W

    # queries (+ LoRA), rotated
    hq = tl.arange(0, NH_P)[:, None]
    mq = (hq < NH) & (half < HALF)
    off = hq * D + half
    x1 = tl.load(row + off, mask=mq, other=0.0).to(tl.float32)
    x2 = tl.load(row + off + HALF, mask=mq, other=0.0).to(tl.float32)
    if has_lq:
        x1 = x1 + tl.load(lq_ptr + tok * (NH * D) + off, mask=mq, other=0.0).to(tl.float32)
        x2 = x2 + tl.load(lq_ptr + tok * (NH * D) + off + HALF, mask=mq, other=0.0).to(tl.float32)
    dst = q_ptr + ((b * NH + hq) * T + t) * D + half
    tl.store(dst, (x1 * c1 - x2 * s1).to(tl.float16), mask=mq)
    tl.store(dst + HALF, (x2 * c2 + x1 * s2).to(tl.float16), mask=mq)

    # keys, rotated
    hk = tl.arange(0, NKV_P)[:, None]
    mk = (hk < NKV) & (half < HALF)
    off = NH * D + hk * D + half
    x1 = tl.load(row + off, mask=mk, other=0.0).to(tl.float32)
    x2 = tl.load(row + off + HALF, mask=mk, other=0.0).to(tl.float32)
    dst = k_ptr + ((b * NKV + hk) * T + t) * D + half
    tl.store(dst, (x1 * c1 - x2 * s1).to(tl.float16), mask=mk)
    tl.store(dst + HALF, (x2 * c2 + x1 * s2).to(tl.float16), mask=mk)

    # values (+ LoRA)
    off = (NH + NKV) * D + hk * D + half
    x1 = tl.load(row + off, mask=mk, other=0.0).to(tl.float32)
    x2 = tl.load(row + off + HALF, mask=mk, other=0.0).to(tl.float32)
    if has_lv:
        lrow = lv_ptr + tok * (NKV * D) + hk * D + half
        x1 = x1 + tl.load(lrow, mask=mk, other=0.0).to(tl.float32)
        x2 = x2 + tl.load(lrow + HALF, mask=mk, other=0.0).to(tl.float32)
    dst = v_ptr + ((b * NKV + hk) * T + t) * D + half
    tl.store(dst, x1.to(tl.float16), mask=mk)
    tl.store(dst + HALF, x2.to(tl.float16), mask=mk)


@triton.jit(do_not_specialize=["T", "has_dq", "has_dk", "has_dv"])
def rope_pack_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, pos_ptr, dqkv_ptr,
                         T, has_dq, has_dk, has_dv,
                         NH: tl.constexpr, NKV: tl.constexpr, HALF: tl.constexpr,
                         NH_P: tl.constexpr, NKV_P: tl.constexpr):
    tok = tl.program_id(0).to(tl.int64)
    pos = tl.load(pos_ptr + tok).to(tl.int64)
    b = pos // T
    t = pos % T
    D = 2 * HALF
    W = (NH + 2 * NKV) * D
    half = tl.arange(0, HALF)[None, :]
    c1 = tl.load(cos_ptr + t * D + half)
    c2 = tl.load(cos_ptr + t * D + HALF + half)
    s1 = tl.load(sin_ptr + t * D + half)
    s2 = tl.load(sin_ptr + t * D + HALF + half)
    row = dqkv_ptr + tok * W

    hq = tl.arange(0, NH_P)[:, None]
    mq = (hq < NH) & (half < HALF)
    g1 = tl.zeros([NH_P, HALF], dtype=tl.float32)
    g2 = tl.zeros([NH_P, HALF], dtype=tl.float32)
    if has_dq:
        src = dq_ptr + ((b * NH + hq) * T + t) * D + half
        g1 = tl.load(src, mask=mq, other=0.0).to(tl.float32)
        g2 = tl.load(src + HALF, mask=mq, other=0.0).to(tl.float32)
    off = hq * D + half
    tl.store(row + off, (g1 * c1 + g2 * s2).to(tl.float16), mask=mq)
    tl.store(row + off + HALF, (g2 * c2 - g1 * s1).to(tl.float16), mask=mq)

    hk = tl.arange(0, NKV_P)[:, None]
    mk = (hk < NKV) & (half < HALF)
    g1 = tl.zeros([NKV_P, HALF], dtype=tl.float32)
    g2 = tl.zeros([NKV_P, HALF], dtype=tl.float32)
    if has_dk:
        src = dk_ptr + ((b * NKV + hk) * T + t) * D + half
        g1 = tl.load(src, mask=mk, other=0.0).to(tl.float32)
        g2 = tl.load(src + HALF, mask=mk, other=0.0).to(tl.float32)
    off = NH * D + hk * D + half
    tl.store(row + off, (g1 * c1 + g2 * s2).to(tl.float16), mask=mk)
    tl.store(row + off + HALF, (g2 * c2 - g1 * s1).to(tl.float16), mask=mk)

    g1 = tl.zeros([NKV_P, HALF], dtype=tl.float32)
    g2 = tl.zeros([NKV_P, HALF], dtype=tl.float32)
    if has_dv:
        src = dv_ptr + ((b * NKV + hk) * T + t) * D + half
        g1 = tl.load(src, mask=mk, other=0.0).to(tl.float32)
        g2 = tl.load(src + HALF, mask=mk, other=0.0).to(tl.float32)
    off = (NH + NKV) * D + hk * D + half
    tl.store(row + off, g1.to(tl.float16), mask=mk)
    tl.store(row + off + HALF, g2.to(tl.float16), mask=mk)


def _consts(n_heads, n_kv, head_dim):
    return dict(NH=n_heads, NKV=n_kv, HALF=head_dim // 2, NH_P=next_pow2(n_heads),
                NKV_P=next_pow2(n_kv))


class RopePack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, lora_q, lora_v, cos, sin, pos_idx, batch, seq, n_heads, n_kv, head_dim):
        rows = qkv.shape[0]
        dev = qkv.device
        q = torch.zeros((batch, n_heads, seq, head_dim), dtype=torch.float16, device=dev)
        k = torch.zeros((batch, n_kv, seq, head_dim), dtype=torch.float16, device=dev)
        v = torch.zeros((batch, n_kv, seq, head_dim), dtype=torch.float16, device=dev)
        if rows:
            rope_pack_fwd_kernel[(rows,)](
                qkv, lora_q if lora_q is not None else qkv, lora_v if lora_v is not None else qkv,
                cos, sin, pos_idx, q, k, v, seq, int(lora_q is not None), int(lora_v is not None),
                **_consts(n_heads, n_kv, head_dim))
        ctx.save_for_backward(cos, sin, pos_idx)
        ctx.dims = (rows, seq, n_heads, n_kv, head_dim)
        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        cos, sin, pos_idx = ctx.saved_tensors
        rows, seq, n_heads, n_kv, head_dim = ctx.dims
        need_qkv, need_lq, need_lv = ctx.needs_input_grad[:3]
        if not (need_qkv or need_lq or need_lv):
            return (None,) * 11
        width = (n_heads + 2 * n_kv) * head_dim
        dqkv = torch.empty((rows, width), dtype=torch.float16, device=pos_idx.device)
        grads = [g.contiguous() if g is not None else None for g in (grad_q, grad_k, grad_v)]
        if rows:
            rope_pack_bwd_kernel[(rows,)](
                grads[0] if grads[0] is not None else dqkv,
                grads[1] if grads[1] is not None else dqkv,
                grads[2] if grads[2] is not None else dqkv,
                cos, sin, pos_idx, dqkv, seq,
                int(grads[0] is not None), int(grads[1] is not None), int(grads[2] is not None),
                **_consts(n_heads, n_kv, head_dim))
        q_cols = n_heads * head_dim
        v_start = (n_heads + n_kv) * head_dim
        return ((dqkv if need_qkv else None),
                (dqkv[:, :q_cols] if need_lq else None),
                (dqkv[:, v_start:] if need_lv else None),
                None, None, None, None, None, None, None, None)


def rope_pack(qkv, lora_q, lora_v, cos, sin, pos_idx, batch, seq, n_heads, n_kv, head_dim):
    return RopePack.apply(qkv.contiguous(),
                          lora_q.contiguous() if lora_q is not None else None,
                          lora_v.contiguous() if lora_v is not None else None,
                          cos.contiguous(), sin.contiguous(), pos_idx, batch, seq,
                          n_heads, n_kv, head_dim)
