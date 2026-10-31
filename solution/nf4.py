"""NF4 (bitsandbytes 4-bit, double quant) weight unpacking in Triton, and the frozen 4-bit linear
that uses it.

bitsandbytes stores every frozen projection as NF4: two 4-bit codes per byte (high nibble first),
one absmax per 64 values, and with double quant ("nested") the absmax itself is an 8-bit code
into a 256-entry table, scaled per 256 absmax blocks and shifted by one offset:

    absmax[b] = code2[absmax_u8[b]] * absmax2[b // 256] + offset
    w[i]      = nf4[q[i]] * absmax[i // 64]          (fp32, rounded once to fp16)

bitsandbytes does this in two CUDA launches (absmax first, then the 4-bit values) and then runs
`F.linear`. Here one Triton program unpacks 1024 bytes into 2048 fp16 values with the absmax
lookup inline, and several projections are unpacked into one buffer so that, for example,
q/k/v or gate/up run as a single GEMM. The arithmetic is the same fp32 product, rounded once to
fp16, so the values are the ones bitsandbytes produces.
"""

import torch
import triton
import triton.language as tl

from ._common import triton_ok

BYTES_PER_PROGRAM = 1024


@triton.jit(do_not_specialize=["n_bytes"])
def nf4_unpack_kernel(packed_ptr, absmax_u8_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr,
                      out_ptr, n_bytes,
                      QBLOCK: tl.constexpr, QBLOCK2: tl.constexpr, NBYTES: tl.constexpr):
    pid = tl.program_id(0)
    byte_idx = pid * NBYTES + tl.arange(0, NBYTES)
    inside = byte_idx < n_bytes
    packed = tl.load(packed_ptr + byte_idx, mask=inside, other=0).to(tl.int32)
    # One absmax block covers QBLOCK values = QBLOCK // 2 bytes; both nibbles share it.
    block = byte_idx // (QBLOCK // 2)
    code = tl.load(absmax_u8_ptr + block, mask=inside, other=0).to(tl.int32)
    scale = tl.load(code2_ptr + code, mask=inside, other=0.0) \
        * tl.load(absmax2_ptr + block // QBLOCK2, mask=inside, other=0.0)
    scale = scale + tl.load(offset_ptr)
    first = tl.load(nf4_ptr + (packed >> 4), mask=inside, other=0.0) * scale
    second = tl.load(nf4_ptr + (packed & 15), mask=inside, other=0.0) * scale
    values = tl.interleave(first, second)
    out_idx = pid * (2 * NBYTES) + tl.arange(0, 2 * NBYTES)
    tl.store(out_ptr + out_idx, values.to(tl.float16), mask=out_idx < 2 * n_bytes)


def _quant_state(weight):
    return getattr(weight, "quant_state", None)


def supports_fast_unpack(weight):
    """True for the layout the kernel handles: NF4, nested absmax with 256-blocks, even sizes."""
    state = _quant_state(weight)
    if state is None or getattr(state, "quant_type", None) != "nf4" or not state.nested:
        return False
    if state.state2 is None or state.state2.blocksize != 256 or state.blocksize % 2:
        return False
    shape = tuple(state.shape)
    if len(shape) != 2 or (shape[0] * shape[1]) % state.blocksize:
        return False
    return weight.data.dtype == torch.uint8 and weight.data.is_contiguous()


def _on_device(weight, device):
    state = _quant_state(weight)
    tensors = (weight.data, state.absmax, state.code, state.state2.code, state.state2.absmax)
    return all(t.device == device for t in tensors) and \
        (not torch.is_tensor(state.offset) or state.offset.device == device)


def unpack_into(weight, out):
    """Write the fp16 values of one bitsandbytes 4-bit weight ([N, K]) into `out` ([N, K])."""
    state = _quant_state(weight)
    n_bytes = weight.data.numel()
    if triton_ok(out) and supports_fast_unpack(weight) and _on_device(weight, out.device):
        grid = (triton.cdiv(n_bytes, BYTES_PER_PROGRAM),)
        nf4_unpack_kernel[grid](
            weight.data, state.absmax, state.state2.code, state.state2.absmax,
            state.offset if torch.is_tensor(state.offset)
            else torch.tensor([float(state.offset)], device=out.device),
            state.code, out, n_bytes,
            QBLOCK=state.blocksize, QBLOCK2=state.state2.blocksize, NBYTES=BYTES_PER_PROGRAM,
            enable_fp_fusion=False)  # mul, add, mul rounded separately, as bitsandbytes does
        return out
    import bitsandbytes.functional as bnb_functional

    out.copy_(bnb_functional.dequantize_4bit(weight.data, state).to(out.dtype).view_as(out))
    return out


def weight_shape(weight):
    return tuple(_quant_state(weight).shape)


def unpack_concat(weights, device):
    """fp16 [sum N_i, K]: the weights stacked by rows, each unpacked in place."""
    rows = [weight_shape(w)[0] for w in weights]
    cols = weight_shape(weights[0])[1]
    out = torch.empty((sum(rows), cols), dtype=torch.float16, device=device)
    start = 0
    for weight, n in zip(weights, rows):
        unpack_into(weight, out[start:start + n])
        start += n
    return out


class FrozenNF4Linear(torch.autograd.Function):
    """y = x @ W^T (+ b) for frozen 4-bit weights stacked by rows, fp16 in and out.

    Nothing large is saved: the weights are unpacked again in backward (as bitsandbytes does),
    and no input is kept because W does not train."""

    @staticmethod
    def forward(ctx, x, bias, weights):
        weight = unpack_concat(weights, x.device)
        out = torch.nn.functional.linear(x, weight, bias)
        ctx.weights = weights
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = None
        if ctx.needs_input_grad[0]:
            weight = unpack_concat(ctx.weights, grad_out.device)
            grad_x = grad_out.to(torch.float16) @ weight
        return grad_x, None, None


def frozen_linear(x, weights, bias=None):
    return FrozenNF4Linear.apply(x, bias, tuple(weights))
