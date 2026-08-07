"""
int8_profile.py — diagnoses why torchao int8 inference underperforms fp32
for conv-dominated CNNs on this GPU.

Two hypotheses under test:
  H1: no optimized int8 Conv2d kernel; torchao falls back to a path that
      dequantizes to fp32, computes, and (for weight-only) skips
      re-quantization -- strictly more work than plain fp32 for zero
      compute benefit.
  H2: dynamic activation quantization overhead (a per-forward-pass
      activation min/max reduction + quantize step, on every Linear layer)
      exceeds any int8 compute savings.

Method: reuse benchmark.py's benchmark_latency (warmup + cuda synchronize +
median) for the throughput evidence, and torch.profiler for kernel-level
evidence. Two int8 configs are compared against fp32:
  - int8_weight_only: quantize_(model, Int8WeightOnlyConfig()) with the
    default filter_fn, which only matches nn.Linear -- Conv2d is left
    untouched fp32. No runtime activation quantization at all. Isolates H2.
  - int8_dynamic_act: "the" deployed config, obtained from the shared
    src.quantization.deploy.build_int8_model builder (dynamic-activation
    int8 on Linear, weight-only int8 on Conv2d via IntxWeightOnlyConfig --
    see apply_int8_quantization's docstring in benchmark.py). This is the
    config actually responsible for the observed slowdown; reusing the
    shared builder (rather than reconstructing it here) keeps this
    diagnosis grounded in what's really deployed and guarantees Conv2d
    actually got quantized (the builder asserts this).

Never loads a saved int8 checkpoint. int8_weight_only is a deliberately
different, Linear-only diagnostic config (not "the" deployed model) built
directly off of build_int8_model's baked_model, so it's the one int8
variant here that isn't itself "the deployed model" and legitimately lives
outside the shared builder.
"""

import os
import copy
import logging

import torch
import torch.nn as nn
from torchao.quantization import (
    quantize_,
    Int8WeightOnlyConfig,
)

from src.analysis.benchmark import benchmark_latency, DEFAULT_BATCH_SIZES
from src.quantization import deploy

logger = logging.getLogger(__name__)

PROFILE_BATCH_SIZE = 32
PROFILE_WARMUP = 10

SWEEP_CSV_FIELDNAMES = [
    "model", "dataset", "stage", "config", "batch",
    "fp32_latency_ms", "int8_latency_ms",
    "fp32_throughput_ips", "int8_throughput_ips", "speedup_x",
]

# Heuristic keyword classification of profiler kernel names. Exact cuDNN/
# cutlass/torchao kernel names vary by version, so this is pattern matching,
# not an exhaustive whitelist -- the report also embeds the raw top-10
# kernel table so a human can sanity-check the classification directly.
QUANT_KEYWORDS = ("quant", "dequant", "choose_qparams", "fake_quant")
COMPUTE_KEYWORDS = (
    "gemm", "conv", "mm", "addmm", "cutlass", "sgemm", "cudnn_convolution",
    "matmul", "bmm", "winograd",
)


def get_gpu_capability_info(device: torch.device) -> dict:
    """
    Logs and returns the GPU name/compute-capability, plus a static note on
    int8 Conv2d support in torchao 0.17. This note isn't a per-run
    measurement -- it's grounded in what's already been verified elsewhere
    in this codebase (see apply_int8_quantization's docstring in
    benchmark.py): Int8DynamicActivationInt8WeightConfig's quantize_()
    default filter_fn (_is_linear) only matches nn.Linear, and its tensor
    subclass only implements aten.linear/aten.mm dispatch -- there is no
    aten.conv2d support at all, independent of GPU architecture. The one
    config that does quantize Conv2d weights, IntxWeightOnlyConfig, is
    weight-only: it dequantizes to fp32 and runs a standard conv at forward
    time, saving storage but not compute.
    """
    if device.type != "cuda":
        logger.warning("[Int8Diag] Not running on CUDA -- GPU kernel dispatch cannot be assessed.")
        return {"device_name": "cpu", "compute_capability": None, "sm": None}

    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    sm = f"sm{capability[0]}{capability[1]}"

    logger.info(f"[Int8Diag] GPU: {name} (compute capability {capability[0]}.{capability[1]}, {sm})")
    logger.info(
        "[Int8Diag] torchao 0.17: Int8DynamicActivationInt8WeightConfig's default filter_fn "
        "only matches nn.Linear (aten.linear/aten.mm dispatch) -- it does not implement "
        "aten.conv2d at all. The only Conv2d-capable int8 config, IntxWeightOnlyConfig, is "
        "weight-only and dequantizes to fp32 before a standard fp32 conv at forward time. "
        "There is no optimized int8 GEMM/conv kernel path for Conv2d in this torchao "
        "version, on this or any GPU architecture."
    )

    return {"device_name": name, "compute_capability": capability, "sm": sm}


def profile_kernels(model: nn.Module, input_shape: tuple, device: torch.device,
                     warmup: int = PROFILE_WARMUP) -> dict:
    """
    Runs `warmup` untimed forward passes, then a single profiled forward
    pass under torch.profiler with CPU(+CUDA) activity tracking. Classifies
    each kernel event by name (see QUANT_KEYWORDS/COMPUTE_KEYWORDS) to
    attribute total device time to quantize/dequantize ops vs actual
    matmul/conv compute -- a high quantize/dequantize fraction is direct
    evidence for H2.

    Returns dict: top_kernels (formatted table string, top 10 by device
    time), quant_time_us, compute_time_us, other_time_us, total_time_us,
    quant_fraction.
    """
    model.eval()
    dummy_input = torch.randn(input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.no_grad(), torch.profiler.profile(activities=activities) as prof:
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

    averages = prof.key_averages()

    def _event_time_us(evt) -> float:
        # PyTorch renamed the *_cuda_time_total attributes to
        # *_device_time_total around the 2.1 release; fall back across
        # whichever this torch build exposes.
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            val = getattr(evt, attr, None)
            if val is not None:
                return float(val)
        return 0.0

    quant_time_us = 0.0
    compute_time_us = 0.0
    total_time_us = 0.0

    for evt in averages:
        t = _event_time_us(evt)
        total_time_us += t
        key_lower = evt.key.lower()
        if any(kw in key_lower for kw in QUANT_KEYWORDS):
            quant_time_us += t
        elif any(kw in key_lower for kw in COMPUTE_KEYWORDS):
            compute_time_us += t

    other_time_us = max(0.0, total_time_us - quant_time_us - compute_time_us)
    quant_fraction = quant_time_us / total_time_us if total_time_us > 0 else 0.0

    sort_key = "self_device_time_total" if any(
        hasattr(evt, "self_device_time_total") for evt in averages
    ) else "self_cuda_time_total"
    try:
        top_kernels = averages.table(sort_by=sort_key, row_limit=10)
    except (KeyError, AssertionError):
        top_kernels = averages.table(sort_by="self_cpu_time_total", row_limit=10)

    return {
        "top_kernels": top_kernels,
        "quant_time_us": quant_time_us,
        "compute_time_us": compute_time_us,
        "other_time_us": other_time_us,
        "total_time_us": total_time_us,
        "quant_fraction": quant_fraction,
    }


def run_int8_perf_diagnosis(
    model_name: str,
    dataset_name: str,
    stage: str,
    channels: int,
    num_classes: int,
    image_size: int,
    fp32_checkpoint_path: str,
    checkpoint_path: str,
    device: torch.device,
    batch_sizes: tuple = DEFAULT_BATCH_SIZES,
    profile_batch_size: int = PROFILE_BATCH_SIZE,
) -> dict:
    """
    Reconstructs the fp32 baseline (loaded from its saved checkpoint, same
    as _run_benchmark_int8_only) and the deployed int8 models fresh from a
    saved PTQ/QAT checkpoint -- never loads a saved int8 checkpoint. Runs
    the 3-way throughput sweep (fp32 / int8_weight_only / int8_dynamic_act)
    and a single-forward-pass kernel profile for fp32 vs int8_dynamic_act.

    Returns dict: sweep_rows (list of per-config/per-batch dicts),
    fp32_profile, int8_profile (both from profile_kernels), conv_quantized,
    linear_quantized (bools, from build_int8_model's quantization audit --
    lets the report state plainly whether Conv2d was actually touched by
    the dynamic-act config on this run; build_int8_model already asserts
    both are True, so these are always True here, but are threaded through
    for the report to state explicitly rather than assume).
    """
    from src.model_cnn.train import build_model

    if not os.path.exists(fp32_checkpoint_path):
        raise FileNotFoundError(f"No saved FP32 baseline at {fp32_checkpoint_path}")

    fp32_model = build_model(
        num_classes=num_classes, model_name=model_name,
        channels=channels, image_size=image_size,
    ).to(device)
    fp32_model.load_state_dict(
        torch.load(fp32_checkpoint_path, map_location=device, weights_only=True)
    )
    fp32_model.eval()

    # baked_model / dynamic_act_model ("the" deployed int8 model) both come
    # from the shared builder -- see its module docstring for why every
    # path that reconstructs an int8 model must go through it rather than
    # calling quantize_() independently.
    baked_model, dynamic_act_model, audit_details = deploy.build_int8_model(
        model_name=model_name,
        dataset_name=dataset_name,
        stage=stage,
        checkpoint_path=checkpoint_path,
        device=device,
        num_classes=num_classes,
        channels=channels,
        image_size=image_size,
    )

    # int8_weight_only is a deliberately different, Linear-only config
    # (Int8WeightOnlyConfig's default filter_fn leaves Conv2d untouched) --
    # an intentional diagnostic comparison against "the" deployed config,
    # not a second reconstruction of it, so it's built directly here rather
    # than through the shared builder (which always applies the full,
    # Conv2d-inclusive scheme).
    weight_only_model = copy.deepcopy(baked_model)
    quantize_(weight_only_model, Int8WeightOnlyConfig())
    weight_only_model.eval()

    conv_quantized = any(d["layer_type"] == "Conv2d" and d["is_quantized"] for d in audit_details)
    linear_quantized = any(d["layer_type"] == "Linear" and d["is_quantized"] for d in audit_details)

    base_input_shape = (channels, image_size, image_size)

    sweep_rows: list[dict] = []
    for batch in batch_sizes:
        input_shape = (batch,) + base_input_shape
        fp32_result = benchmark_latency(fp32_model, input_shape, device)

        for config_name, int8_model in (
            ("int8_weight_only", weight_only_model),
            ("int8_dynamic_act", dynamic_act_model),
        ):
            int8_result = benchmark_latency(int8_model, input_shape, device)
            sweep_rows.append({
                "config": config_name,
                "batch": batch,
                "fp32_latency_ms": fp32_result["latency_ms"],
                "int8_latency_ms": int8_result["latency_ms"],
                "fp32_throughput_ips": fp32_result["throughput_ips"],
                "int8_throughput_ips": int8_result["throughput_ips"],
                "speedup_x": fp32_result["latency_ms"] / int8_result["latency_ms"],
            })

    profile_input_shape = (profile_batch_size,) + base_input_shape
    fp32_profile = profile_kernels(fp32_model, profile_input_shape, device)
    int8_profile_result = profile_kernels(dynamic_act_model, profile_input_shape, device)

    del baked_model, weight_only_model, dynamic_act_model

    return {
        "sweep_rows": sweep_rows,
        "fp32_profile": fp32_profile,
        "int8_profile": int8_profile_result,
        "conv_quantized": conv_quantized,
        "linear_quantized": linear_quantized,
    }


def format_sweep_table(sweep_rows: list[dict]) -> str:
    """Renders run_int8_perf_diagnosis's sweep_rows as a plain-text table."""
    header = f"{'config':<18}{'batch':>7}{'fp32_ms':>10}{'int8_ms':>10}{'fp32_ips':>12}{'int8_ips':>12}{'speedup_x':>11}"
    lines = [header, "-" * len(header)]
    for row in sweep_rows:
        lines.append(
            f"{row['config']:<18}{row['batch']:>7}{row['fp32_latency_ms']:>10.4f}"
            f"{row['int8_latency_ms']:>10.4f}{row['fp32_throughput_ips']:>12.1f}"
            f"{row['int8_throughput_ips']:>12.1f}{row['speedup_x']:>11.3f}"
        )
    return "\n".join(lines)


def build_int8_perf_report(
    model_name: str,
    dataset_name: str,
    stage: str,
    sweep_rows: list[dict],
    fp32_profile: dict,
    int8_profile: dict,
    conv_quantized: bool,
    linear_quantized: bool,
) -> str:
    """
    Assembles one (model, dataset, stage) section of the diagnosis report:
    the 3-way throughput table, top-10 kernels for fp32 vs int8, the
    quantize/dequantize time fraction, and a stated conclusion on which
    hypothesis the evidence supports for this run. Reports honestly: if
    neither int8 config beats fp32, the conclusion says so plainly.
    """
    dynamic_speedups = [r["speedup_x"] for r in sweep_rows if r["config"] == "int8_dynamic_act"]
    weightonly_speedups = [r["speedup_x"] for r in sweep_rows if r["config"] == "int8_weight_only"]
    avg_dynamic = sum(dynamic_speedups) / len(dynamic_speedups) if dynamic_speedups else float("nan")
    avg_weightonly = sum(weightonly_speedups) / len(weightonly_speedups) if weightonly_speedups else float("nan")

    conclusion = [
        f"Mean speedup_x across batch sizes: int8_weight_only={avg_weightonly:.3f}, "
        f"int8_dynamic_act={avg_dynamic:.3f} (>1.0 = faster than fp32).",
        f"Dynamic-act audit: Conv2d quantized={conv_quantized}, Linear quantized={linear_quantized}.",
    ]
    if avg_weightonly >= 0.95 and avg_dynamic < avg_weightonly - 0.05:
        conclusion.append(
            "H2 CONFIRMED: weight-only (no runtime activation quantization) tracks fp32 "
            "speed, while dynamic-activation is markedly slower -- the per-forward-pass "
            "activation min/max + quantize step is the dominant added cost here, not the "
            "underlying weight precision."
        )
    if int8_profile["quant_fraction"] > 0.15:
        conclusion.append(
            f"H1/H2 evidence from kernel profile: {int8_profile['quant_fraction']:.1%} of "
            "int8 device time is spent in quantize/dequantize ops rather than matmul/conv "
            "compute."
        )
    if not conv_quantized:
        conclusion.append(
            "Conv2d was not quantized under the dynamic-act config on this run -- Conv2d "
            "compute is identical to fp32; any int8 slowdown here comes from Linear-layer "
            "dynamic activation quantization plus fixed per-call overhead, not from a conv "
            "kernel fallback."
        )
    if avg_dynamic < 1.0 and avg_weightonly < 1.0:
        conclusion.append(
            "Neither int8 config is faster than fp32 for this model/dataset/stage: no "
            "configuration tested here delivers a throughput benefit."
        )
    elif avg_weightonly >= 1.0:
        conclusion.append(
            "int8_weight_only is faster than fp32 here. It is not accelerating Conv2d "
            "compute (Conv2d is left fp32 by its default filter_fn) but avoids the "
            "dynamic-act config's per-call activation quantization while still shrinking "
            "weight storage -- see apply_int8_quantization's docstring in benchmark.py."
        )

    lines = [
        "=" * 78,
        f"{model_name} / {dataset_name} / {stage}",
        "=" * 78,
        "",
        "--- Throughput sweep: fp32 vs int8_weight_only vs int8_dynamic_act ---",
        format_sweep_table(sweep_rows),
        "",
        "--- Top-10 CUDA kernels: fp32 ---",
        fp32_profile["top_kernels"],
        "",
        "--- Top-10 CUDA kernels: int8 (dynamic-act config) ---",
        int8_profile["top_kernels"],
        "",
        "--- Quantize/dequantize time attribution (int8, dynamic-act) ---",
        f"quantize/dequantize: {int8_profile['quant_time_us']:.1f} us "
        f"({int8_profile['quant_fraction']:.1%} of total device time)",
        f"matmul/conv compute:  {int8_profile['compute_time_us']:.1f} us",
        f"other:                {int8_profile['other_time_us']:.1f} us",
        f"total:                {int8_profile['total_time_us']:.1f} us",
        "",
        f"--- Conclusion ({model_name}/{dataset_name}/{stage}) ---",
        *conclusion,
        "",
    ]
    return "\n".join(lines)


def write_report(path: str, gpu_info: dict, sections: list[str]) -> None:
    """Writes the combined header + all per-run sections to a plain-text report."""
    if gpu_info.get("compute_capability"):
        cap = gpu_info["compute_capability"]
        cap_line = f"Compute capability: {cap[0]}.{cap[1]} ({gpu_info['sm']})"
    else:
        cap_line = "Compute capability: n/a (not running on CUDA)"

    header = [
        "torchao int8 vs fp32 performance diagnosis",
        "=" * 78,
        f"GPU: {gpu_info.get('device_name')}",
        cap_line,
        "",
        "torchao 0.17 int8 Conv2d support: Int8DynamicActivationInt8WeightConfig's default",
        "filter_fn only matches nn.Linear (aten.linear/aten.mm dispatch); it does not",
        "implement aten.conv2d at all, so Conv2d layers are left untouched fp32 under",
        "quantize_(model, Int8DynamicActivationInt8WeightConfig()) with no filter_fn",
        "override. The only config that quantizes Conv2d weights, IntxWeightOnlyConfig, is",
        "weight-only: it dequantizes to fp32 and runs a standard fp32 conv at forward time,",
        "saving storage but not compute. There is no optimized int8 GEMM/conv kernel path",
        "for Conv2d in this torchao version.",
        "",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(header))
        f.write("\n".join(sections))

    logger.info(f"[Int8Diag] Report written -> {path}")
