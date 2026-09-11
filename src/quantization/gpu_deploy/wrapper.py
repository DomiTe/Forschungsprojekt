import torch
import torch.nn as nn
import triton
import triton.language as tl

from Forschungsprojekt.src.quantization.real_quant_attempt.kernel import _int8_gemm_kernel

def po2_int8_linear(x: torch.Tensor, weight: torch.Tensor, scale_x: torch.Tensor, scale_w: torch.Tensor) -> torch.Tensor:
    """
    Executes INT8 GEMM on A100 Tensor Cores.
    x: [M, K] of type torch.int8
    weight: [K, N] of type torch.int8
    scale_x: [M] FP32 row-wise activation scales
    scale_w: [N] FP32 col-wise weight scales
    """
    assert x.dtype == torch.int8
    assert weight.dtype == torch.int8
    assert x.is_contiguous()
    assert weight.is_contiguous()

    M, K = x.shape
    K_w, N = weight.shape
    assert K == K_w

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    _int8_gemm_kernel[grid](
        x, weight, out,
        scale_x, scale_w,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
    )

    return out