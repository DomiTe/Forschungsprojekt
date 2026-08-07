import io
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torchao.quantization import (
    quantize_,
    Int8DynamicActivationInt8WeightConfig,
    IntxWeightOnlyConfig,
    PerAxis,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)

# torchao's quantize_() silently no-ops on layer types it doesn't recognize,
# leaving plain fp32 weights behind. Any parameter whose runtime type isn't
# one of these is a torchao quantized tensor subclass (e.g.
# LinearActivationQuantizedTensor, IntxUnpackedToInt8Tensor).
_UNQUANTIZED_PARAM_TYPES = ("Parameter", "Tensor")


def apply_int8_quantization(model: nn.Module) -> list[dict]:
    """
    Applies the best available torchao (0.17) int8 quantization to every
    Conv2d and Linear layer in `model`, in place.

    Investigation: torchao 0.17's dynamic-activation-int8 configs
    (Int8DynamicActivationInt8WeightConfig, Int8DynamicActivationIntxWeightConfig)
    only handle 2-d (Linear) weights. quantize_()'s default filter_fn
    (_is_linear) already excludes Conv2d, but even forcing a wider filter_fn
    doesn't help: Int8DynamicActivationIntxWeightConfig asserts
    weight.dim() == 2 outright, and Int8DynamicActivationInt8WeightConfig's
    transform reads weight.shape[-1] as "in_features" to decide whether
    quantization is worthwhile — for a Conv2d weight (out_ch, in_ch, kH, kW)
    that's actually the kernel width (typically 3), which trips its
    "skip if in_features <= 16" heuristic and hands back the *original* fp32
    weight completely unconverted, no error raised. (Verified directly: a
    Conv2d run through Int8DynamicActivationInt8WeightConfig with a widened
    filter_fn still reports `type(weight).__name__ == "Parameter"`
    afterward.) Its resulting LinearActivationQuantizedTensor also only
    implements dispatch for aten.linear/aten.mm, not aten.conv2d, so even a
    correctly-shaped conversion couldn't be used in a conv forward pass.
    This combination is exactly why conv-heavy models previously showed
    ~1.00x size reduction while still passing a type check that happened to
    only inspect a Linear weight.

    The one config in this torchao version that correctly quantizes Conv2d
    weights is IntxWeightOnlyConfig(weight_dtype=torch.int8): its
    IntxUnpackedToInt8Tensor explicitly implements aten.conv2d.default /
    F.conv2d dispatch (torchao/quantization/quantize_/workflows/intx/
    intx_unpacked_to_int8_tensor.py). It is weight-only, though — at
    forward time it dequantizes the weight and runs a normal fp32 conv, so
    it shrinks Conv2d storage (~4x) without accelerating conv compute. That
    is a fundamentally different tradeoff than the dynamic-activation path
    Linear layers get, so Conv2d and Linear are quantized separately here
    and the distinction is surfaced via the per-layer audit rather than
    folded into one number.

    Returns the per-layer audit (see audit_layer_quantization).
    """
    quantize_(
        model,
        Int8DynamicActivationInt8WeightConfig(),
        filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear),
    )
    quantize_(
        model,
        IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
        filter_fn=lambda mod, fqn: isinstance(mod, nn.Conv2d),
    )
    return audit_layer_quantization(model)


def audit_layer_quantization(model: nn.Module) -> list[dict]:
    """
    Walks every Conv2d/Linear module and classifies its weight as a torchao
    quantized tensor subclass or a plain fp32 Parameter/Tensor.

    Returns a list of dicts with keys: fqn, layer_type ("Conv2d"/"Linear"),
    weight_type (runtime type name), is_quantized.
    """
    details = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
            weight_type = type(module.weight).__name__
            details.append({
                "fqn": name or "<root>",
                "layer_type": "Conv2d" if isinstance(module, nn.Conv2d) else "Linear",
                "weight_type": weight_type,
                "is_quantized": weight_type not in _UNQUANTIZED_PARAM_TYPES,
            })
    return details


def summarize_quantization_audit(details: list) -> dict:
    """Aggregates audit_layer_quantization's output by layer type."""
    conv = [d for d in details if d["layer_type"] == "Conv2d"]
    linear = [d for d in details if d["layer_type"] == "Linear"]
    conv_quantized_count = sum(d["is_quantized"] for d in conv)
    linear_quantized_count = sum(d["is_quantized"] for d in linear)
    return {
        "conv_total": len(conv),
        "conv_quantized_count": conv_quantized_count,
        "conv_quantized": len(conv) > 0 and conv_quantized_count == len(conv),
        "linear_total": len(linear),
        "linear_quantized_count": linear_quantized_count,
        "linear_quantized": len(linear) > 0 and linear_quantized_count == len(linear),
    }


def log_quantization_audit(details: list, label: str) -> dict:
    """
    Logs the per-layer Conv2d/Linear quantization audit plus a summary.
    Callers should rank-0-guard this (the quantization itself must still
    run on every rank via apply_int8_quantization).

    Returns the summary dict from summarize_quantization_audit, for CSV
    reporting.
    """
    for d in details:
        status = "quantized" if d["is_quantized"] else "FP32 (NOT quantized)"
        logger.info(
            f"[Benchmark] {label} {d['fqn']} ({d['layer_type']}): "
            f"weight type = {d['weight_type']} [{status}]"
        )

    summary = summarize_quantization_audit(details)
    logger.info(
        f"[Benchmark] {label} quantization audit: "
        f"Conv2d {summary['conv_quantized_count']}/{summary['conv_total']} quantized "
        f"(weight-only int8), Linear {summary['linear_quantized_count']}/{summary['linear_total']} "
        f"quantized (dynamic-activation int8)"
    )
    if summary["conv_total"] > 0 and not summary["conv_quantized"]:
        logger.error(
            f"[Benchmark] {label}: {summary['conv_total'] - summary['conv_quantized_count']} "
            f"of {summary['conv_total']} Conv2d layer(s) remain fp32 — torchao could not "
            f"quantize them in this run."
        )
    return summary


def model_size_bytes(model: nn.Module) -> int:
    """
    Serializes model.state_dict() via torch.save and measures the resulting
    byte size.

    Summing param.numel() * param.element_size() is not reliable for
    torchao quantized tensors — the wrapper's reported element_size() does
    not always reflect the packed int8 storage underneath. Serialized size
    is the faithful "what would actually be written to disk" measurement,
    and correctly shows the ~3-4x reduction from int8 (not exactly 4x:
    biases/BatchNorm stay fp32, and scale/zero_point add small overhead).
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def benchmark_latency(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    """
    Benchmarks median per-call latency and throughput at a single batch size.

    CUDA kernel launches are async, so a timer without torch.cuda.synchronize()
    would just measure how fast Python can enqueue work rather than actual
    GPU time — we synchronize once after warmup (so the first timed sample
    isn't contaminated by pending warmup work) and again after every timed
    call. Reports the median (not mean) of `iters` runs to stay robust
    against one-off stalls (e.g. a GC pause or a stray autotune).

    Returns:
        dict with keys: latency_ms (median), throughput_ips
    """
    model.eval()
    dummy_input = torch.randn(input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies_ms = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    median_latency_ms = float(np.median(latencies_ms))
    throughput_ips = input_shape[0] * 1000.0 / median_latency_ms

    return {
        "latency_ms": median_latency_ms,
        "throughput_ips": throughput_ips,
    }


def _linear_fit_with_r2(batch_sizes: tuple, latencies_ms: list) -> dict:
    """
    Fits latency ~= intercept + slope * batch via least squares.

    intercept_ms is the fixed per-call overhead (kernel launch, dispatch)
    and should land in a similar place for both precisions; slope_ms_per_sample
    is the true marginal per-sample compute cost, which is where int8 should
    show its advantage. Small models at small inputs are often
    overhead-dominated (the line is nearly flat), which is expected and
    surfaces as a low R^2 rather than a bug.
    """
    batches = np.asarray(batch_sizes, dtype=np.float64)
    latencies = np.asarray(latencies_ms, dtype=np.float64)

    slope, intercept = np.polyfit(batches, latencies, 1)

    predicted = intercept + slope * batches
    ss_res = float(np.sum((latencies - predicted) ** 2))
    ss_tot = float(np.sum((latencies - latencies.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "intercept_ms": float(intercept),
        "slope_ms_per_sample": float(slope),
        "r2": r2,
    }


def compare_fp32_vs_int8(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    base_input_shape: tuple,
    device: torch.device,
    batch_sizes: tuple = DEFAULT_BATCH_SIZES,
) -> dict:
    """
    Compares an fp32 model against its int8 counterpart: model size, a
    per-batch-size latency/throughput sweep, and an overhead/compute
    decomposition fit across that sweep.

    Callers are responsible for ensuring int8_model actually holds torchao
    quantized weights (see src.quantization.deploy.build_int8_model's
    assertion) — this function does not re-check, since fp32_model
    legitimately has none.

    base_input_shape is (channels, height, width); each batch size is
    prepended to form the actual input passed to benchmark_latency. The
    default batch range is intentionally wide so the linear fit below is
    well-conditioned.

    Returns:
        dict with keys: fp32_size_bytes, int8_size_bytes, size_reduction_x,
        sweep (list of per-batch-size dicts), fp32_fit, int8_fit,
        compute_speedup_x (fp32_slope / int8_slope — the overhead-free
        compute speedup, the fairest single number since it factors out
        fixed per-call cost), decomposition_reliable (False if either fit's
        R^2 < 0.9, meaning the linear overhead/compute split shouldn't be
        trusted even though the raw sweep numbers are still valid).
    """
    fp32_size_bytes = model_size_bytes(fp32_model)
    int8_size_bytes = model_size_bytes(int8_model)
    size_reduction_x = fp32_size_bytes / int8_size_bytes

    sweep = []
    fp32_latencies_ms = []
    int8_latencies_ms = []
    for batch in batch_sizes:
        input_shape = (batch,) + tuple(base_input_shape)

        fp32_result = benchmark_latency(fp32_model, input_shape, device)
        int8_result = benchmark_latency(int8_model, input_shape, device)

        fp32_latencies_ms.append(fp32_result["latency_ms"])
        int8_latencies_ms.append(int8_result["latency_ms"])

        sweep.append({
            "batch": batch,
            "fp32_latency_ms": fp32_result["latency_ms"],
            "int8_latency_ms": int8_result["latency_ms"],
            "fp32_throughput_ips": fp32_result["throughput_ips"],
            "int8_throughput_ips": int8_result["throughput_ips"],
            "speedup_x": fp32_result["latency_ms"] / int8_result["latency_ms"],
        })

    fp32_fit = _linear_fit_with_r2(batch_sizes, fp32_latencies_ms)
    int8_fit = _linear_fit_with_r2(batch_sizes, int8_latencies_ms)
    decomposition_reliable = fp32_fit["r2"] >= 0.9 and int8_fit["r2"] >= 0.9

    for label, fit in (("fp32", fp32_fit), ("int8", int8_fit)):
        if fit["r2"] < 0.9:
            logger.warning(
                f"[Benchmark] {label} latency-vs-batch fit has R^2={fit['r2']:.3f} < 0.9; "
                f"overhead/compute decomposition may be unreliable (raw sweep is still valid)."
            )

    compute_speedup_x = fp32_fit["slope_ms_per_sample"] / int8_fit["slope_ms_per_sample"]

    return {
        "fp32_size_bytes": fp32_size_bytes,
        "int8_size_bytes": int8_size_bytes,
        "size_reduction_x": size_reduction_x,
        "sweep": sweep,
        "fp32_fit": fp32_fit,
        "int8_fit": int8_fit,
        "compute_speedup_x": compute_speedup_x,
        "decomposition_reliable": decomposition_reliable,
    }
