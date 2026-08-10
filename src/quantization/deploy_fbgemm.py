"""
deploy_fbgemm.py -- converts PoT-quantized checkpoints to real INT8 via
torch.ao.quantization's eager-mode PTQ with the fbgemm backend, and
benchmarks accuracy/size/throughput on CPU.

Why this exists alongside src/quantization/deploy.py: torchao 0.17 has no
INT8 convolution kernel on the GPU path -- its only Conv2d-capable config
(IntxWeightOnlyConfig) dequantizes to fp32 and runs a standard fp32 conv, so
INT8 inference measured 3-7x *slower* than fp32 on the A100 despite a real
~3.9x size reduction (see apply_int8_quantization's docstring in
src/analysis/benchmark.py). fbgemm, by contrast, does implement quantized
Conv2d/Linear kernels for x86 with AVX-512 VNNI, which the target
workstation CPU (AMD Ryzen 7 7700X, Zen 4) has. This module tests whether a
real throughput benefit is achievable there -- CPU vs CPU, on the same
machine, same thread count, honestly reported either way.

Reuses (does not duplicate) the existing PoT reconstruction pipeline:
    build_model -> fuse_model_architectures -> replace_layers_for_quantization
    -> load_state_dict -> bake_pot_into_standard_layers
(same pipeline src.quantization.deploy.build_int8_model uses for the GPU/
torchao path -- it isn't called directly here because that function always
finishes with apply_int8_quantization, which is torchao/GPU-specific).

Runs as a single local process (`python -m src.main --deploy-cpu-fbgemm
...`), no torchrun/SLURM/torch.distributed involved anywhere in this
module.
"""

import os
import re
import csv
import copy
import types
import logging

import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.nn.quantized import FloatFunctional
from torchvision.models.resnet import BasicBlock, Bottleneck

from src.utility.config import RESULTS_DIR, RUN_ID, CSV_DIR, LOG_DIR, DATASET_SPECS
from src.utility.utils import get_data_loaders
from src.analysis.benchmark import model_size_bytes, benchmark_latency

logger = logging.getLogger(__name__)

MODELS = ["cnn", "resnet18_no_weights", "resnet50_no_weights"]
DATASETS = ["CIFAR10"] # "CIFAR10", "IMAGENET100"
STAGES = ["PTQ", "QAT"]

NUM_CALIBRATION_BATCHES = 50
LATENCY_WARMUP = 20
LATENCY_ITERS = 100
SWEEP_BATCH_SIZES = (1, 8, 32, 64, 128)

SUMMARY_FIELDNAMES = [
    "model", "dataset", "stage", "backend", "num_threads",
    "fp32_acc", "int8_acc", "acc_delta",
    "fp32_size_mb", "int8_size_mb", "size_reduction_x",
    "conv_quantized", "linear_quantized", "eval_subset_batches",
]
SWEEP_FIELDNAMES = [
    "model", "dataset", "stage", "batch",
    "fp32_latency_ms", "int8_latency_ms",
    "fp32_throughput_ips", "int8_throughput_ips", "speedup_x",
]


class FbgemmBuildError(RuntimeError):
    pass


def _save_csv(results: list[dict], path: str, fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"[DeployFbgemm] CSV saved -> {path}")


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_checkpoint_dir(checkpoint_dir: str | None, load_run_id: str | None) -> str:
    if checkpoint_dir:
        resolved = checkpoint_dir
    else:
        run_id = load_run_id or RUN_ID
        resolved = os.path.join(RESULTS_DIR, run_id, "quantized_models")
    logger.info(f"[DeployFbgemm] Checkpoint directory: {resolved}")
    return resolved


def _checkpoint_path(checkpoint_dir: str, stage: str, model_name: str, dataset_name: str) -> str:
    path = os.path.join(checkpoint_dir, f"{stage.lower()}_po2_{model_name}_{dataset_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[DeployFbgemm] Missing checkpoint for {stage} {model_name}/{dataset_name}: {path}"
        )
    return path


# ---------------------------------------------------------------------------
# Model reconstruction: checkpoint -> baked fp32 PoT model
# ---------------------------------------------------------------------------

def _build_baked_model(
    model_name: str,
    checkpoint_path: str,
    num_classes: int,
    channels: int,
    image_size: int,
) -> nn.Module:
    # Deferred imports: bake_pot_into_standard_layers lives in src.main,
    # which imports this module transitively (via src.main's own
    # --deploy-cpu-fbgemm dispatch) -- importing at module scope would be
    # circular. Mirrors the same pattern in src.quantization.deploy.
    from src.main import bake_pot_into_standard_layers
    from src.model_cnn.train import build_model
    from src.quantization.quantizer import fuse_model_architectures, replace_layers_for_quantization

    device = torch.device("cpu")

    quant_model = build_model(
        num_classes=num_classes, model_name=model_name,
        channels=channels, image_size=image_size,
    )
    fuse_model_architectures(quant_model, model_name)
    replace_layers_for_quantization(quant_model)
    quant_model = quant_model.to(device)
    quant_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    quant_model.eval()

    baked_model = bake_pot_into_standard_layers(quant_model)
    baked_model.eval()
    return baked_model


# ---------------------------------------------------------------------------
# fbgemm eager-mode PTQ
# ---------------------------------------------------------------------------

def _fuse_leftover_conv_bn(model: nn.Module, model_name: str) -> None:
    """
    fuse_model_architectures (src/quantization/quantizer.py) fuses conv1/bn1
    and conv2/bn2 per residual block -- complete for resnet18's 2-conv
    BasicBlock, but resnet50's 3-conv Bottleneck also has a conv3/bn3 pair
    it never fuses. That gap is invisible to the torchao weight-only
    PTQ/QAT path this codebase validates elsewhere (it only requires
    Conv2d/Linear module types to exist, not fused BatchNorm), which is why
    it was never hit before. Eager static quantization does require it: a
    plain fp32 BatchNorm2d sitting right after a quantized Conv2d has no
    quantized counterpart in convert()'s module mapping and errors on a
    quantized-tensor input.

    Folded here, at deploy time, on the already PoT-baked conv3 weight:
    this means deployed conv3's weight in resnet50 is no longer an exact
    power of two (it gets scaled per-output-channel by bn3's trained
    gamma/sqrt(running_var+eps)), unlike every other Conv2d/Linear in this
    pipeline. That is a real, reported deviation, not a hidden one -- see
    the module docstring -- and it is the only way to make resnet50 run
    under eager fbgemm quantization at all, since bn3 was never folded
    away during PTQ/QAT training the way conv1/bn1 and conv2/bn2 were.
    """
    if "resnet50" not in model_name:
        return
    fused_any = False
    for container_name, container in model.named_children():
        if not container_name.startswith("layer"):
            continue
        for block in container:
            if hasattr(block, "conv3") and hasattr(block, "bn3"):
                torch.ao.quantization.fuse_modules(block, [["conv3", "bn3"]], inplace=True)
                fused_any = True
    if fused_any:
        logger.warning(
            f"[DeployFbgemm] {model_name}: folded leftover conv3/bn3 BatchNorm into conv3 "
            "for eager quantization compatibility -- conv3's deployed weight is no longer "
            "an exact power of two in this model (see _fuse_leftover_conv_bn docstring)."
        )


def _basic_block_forward(self, x):
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    if self.downsample is not None:
        identity = self.downsample(x)
    out = self.add_func.add(out, identity)
    return self.relu(out)


def _bottleneck_forward(self, x):
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.downsample is not None:
        identity = self.downsample(x)
    out = self.add_func.add(out, identity)
    return self.relu(out)


def _patch_residual_add_for_quantization(model: nn.Module) -> None:
    """
    torchvision's BasicBlock/Bottleneck (resnet18_no_weights/resnet50_no_weights)
    compute the residual connection as a bare `out += identity`. There is no
    QuantizedCPU kernel for aten::add.out at all -- confirmed empirically:
    running the converted model raises NotImplementedError naming exactly
    that op/backend combination the first time a residual add executes on
    two int8 tensors. This is not a Conv2d/Linear problem convert() can fix;
    a bare Python `+=` is invisible to prepare()/convert(), which only
    instrument modules.

    This is exactly the problem src/model_cnn/resnet18.py's own from-scratch
    ResNet18 already documents solving with FloatFunctional ("PyTorch's PTQ
    observer cannot instrument a bare tensor `+`"). Same fix, applied here at
    deploy time via a bound-method patch instead of a source change, since
    the checkpoint's saved state_dict has to keep matching plain torchvision
    BasicBlock/Bottleneck's parameter names -- FloatFunctional carries no
    parameters/buffers of its own, so attaching one post-load_state_dict
    changes nothing about what got loaded.
    """
    for module in model.modules():
        if isinstance(module, BasicBlock):
            module.add_func = FloatFunctional()
            module.forward = types.MethodType(_basic_block_forward, module)
        elif isinstance(module, Bottleneck):
            module.add_func = FloatFunctional()
            module.forward = types.MethodType(_bottleneck_forward, module)


def _resolve_module_name(model: nn.Module, layer_name: str) -> str | None:
    """
    Resolves a bare layer name (e.g. "conv1") to its actual dotted path
    inside `model`, accounting for two independent structural changes
    _apply_fbgemm_ptq can make relative to the original checkpoint's naming:

    - torch.ao.quantization.QuantWrapper, applied to architectures with no
      native QuantStub/DeQuantStub (resnet18_no_weights / resnet50_no_weights),
      shifts every submodule one level down under "module." -- "conv1"
      becomes "module.conv1".
    - _ExcludedLayerWrapper (see below), applied per excluded layer, takes
      over the excluded layer's original name and holds the real leaf
      Conv2d/Linear one level further down at its own ".module" attribute
      -- so once found, if the resolved module is an _ExcludedLayerWrapper,
      this drills in one more level to reach the actual leaf.

    Returns None if no candidate form exists.
    """
    names = dict(model.named_modules())
    if layer_name in names:
        resolved = layer_name
    elif f"module.{layer_name}" in names:
        resolved = f"module.{layer_name}"
    else:
        return None

    if isinstance(names[resolved], _ExcludedLayerWrapper):
        resolved = f"{resolved}.module"
    return resolved


class _ExcludedLayerWrapper(nn.Module):
    """
    Wraps a single Conv2d/Linear that should stay FP32 while everything
    around it gets quantized. Setting a submodule's .qconfig to None makes
    prepare()/convert() skip *converting* it, but does nothing about the
    tensor dtype at its boundary -- static quantization threads quint8
    tensors directly between quantized ops with no implicit dequant/quant
    in between, so an excluded module sitting downstream of the model's
    QuantStub (or of another quantized layer) still receives a quint8
    tensor and crashes on the first quantized-vs-float op (confirmed
    empirically: "RuntimeError: Input type (c10::quint8) and bias type
    (float) should be the same" when conv1 -- the very first layer after
    the model's own QuantStub -- was excluded with a bare qconfig=None).
    DeQuantStub/QuantStub are the standard library's mechanism for exactly
    this: explicit, calibrated type-conversion boundaries. They inherit
    the model-wide qconfig via propagation (this wrapper and they are never
    given an explicit .qconfig of their own), while the wrapped module's
    own .qconfig is forced to None so it alone stays FP32.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.dequant = DeQuantStub()
        self.module = module
        self.quant = QuantStub()

    def forward(self, x):
        x = self.dequant(x)
        x = self.module(x)
        x = self.quant(x)
        return x


def _replace_module_by_path(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _apply_fbgemm_ptq(
    baked_model: nn.Module,
    calibration_loader,
    label: str,
    model_name: str,
    num_calib_batches: int = NUM_CALIBRATION_BATCHES,
    excluded_layers: frozenset[str] | None = None,
) -> tuple[nn.Module, object]:
    """
    Eager-mode static PTQ with the fbgemm backend.

    torch.ao.quantization.prepare/convert only touch modules -- the input
    to the very first quantized module and the output of the very last one
    still need an explicit QuantStub/DeQuantStub boundary to actually
    become/leave a quantized tensor. The CNN model already carries these
    (src/model_cnn/model.py), but the torchvision-based resnet18_no_weights/
    resnet50_no_weights architectures do not, so those get wrapped in
    torch.ao.quantization.QuantWrapper -- the standard library utility for
    exactly this case -- rather than double-wrapping models that already
    have their own stubs.

    excluded_layers (used by src/analysis/layer_ablation.py): bare layer
    names to leave in FP32. Each is replaced by an _ExcludedLayerWrapper
    (DeQuantStub -> module -> QuantStub) before prepare(), with the
    wrapped module's own .qconfig forced to None -- prepare()/convert()
    then skip converting it while still inserting real quantize/dequantize
    ops at its boundary, so tensor dtypes stay consistent with the
    quantized layers around it. See _ExcludedLayerWrapper's docstring for
    why a bare qconfig=None (no wrapper) is not sufficient.
    """
    baked_model.eval()
    _fuse_leftover_conv_bn(baked_model, model_name)
    _patch_residual_add_for_quantization(baked_model)

    has_stubs = any(isinstance(m, (QuantStub, DeQuantStub)) for m in baked_model.modules())
    model_to_quantize = baked_model if has_stubs else torch.ao.quantization.QuantWrapper(baked_model)
    model_to_quantize.eval()

    if excluded_layers:
        for layer_name in excluded_layers:
            resolved_name = _resolve_module_name(model_to_quantize, layer_name)
            if resolved_name is None:
                raise FbgemmBuildError(
                    f"{label}: cannot exclude '{layer_name}' from quantization -- no such "
                    f"module in {model_name} (checked '{layer_name}' and 'module.{layer_name}')."
                )
            leaf_module = dict(model_to_quantize.named_modules())[resolved_name]
            if not isinstance(leaf_module, (nn.Conv2d, nn.Linear)):
                raise FbgemmBuildError(
                    f"{label}: cannot exclude '{resolved_name}' -- expected nn.Conv2d/nn.Linear, "
                    f"got {type(leaf_module).__name__}."
                )
            wrapper = _ExcludedLayerWrapper(leaf_module)
            wrapper.module.qconfig = None
            _replace_module_by_path(model_to_quantize, resolved_name, wrapper)
            logger.info(
                f"[DeployFbgemm] {label}: wrapped '{resolved_name}' in dequant/quant stubs "
                f"and excluded it from quantization (module.qconfig=None)"
            )

    qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    model_to_quantize.qconfig = qconfig
    logger.info(f"[DeployFbgemm] {label}: qconfig = {qconfig}")

    torch.ao.quantization.prepare(model_to_quantize, inplace=True)

    batches_processed = 0
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            if batches_processed >= num_calib_batches:
                break
            model_to_quantize(inputs)
            batches_processed += 1
    logger.info(f"[DeployFbgemm] {label}: calibrated on {batches_processed} batches")

    int8_model = torch.ao.quantization.convert(model_to_quantize, inplace=False)
    int8_model.eval()
    return int8_model, qconfig


def _audit_quantized_modules(model: nn.Module, label: str) -> tuple[bool, bool]:
    """
    Asserts at least one Conv2d and one Linear actually became a
    torch.ao.nn.quantized module -- never silently benchmark an fp32 model
    under an "int8" label. Raises FbgemmBuildError naming the run if not.
    """
    conv_hit = None
    linear_hit = None
    for name, module in model.named_modules():
        if conv_hit is None and isinstance(module, nnq.Conv2d):
            conv_hit = name
        if linear_hit is None and isinstance(module, nnq.Linear):
            linear_hit = name

    if conv_hit is None:
        raise FbgemmBuildError(
            f"{label}: no Conv2d layer was converted to torch.ao.nn.quantized.Conv2d -- "
            f"fbgemm PTQ produced an fp32 conv stack under an 'int8' label."
        )
    if linear_hit is None:
        raise FbgemmBuildError(
            f"{label}: no Linear layer was converted to torch.ao.nn.quantized.Linear -- "
            f"fbgemm PTQ produced an fp32 linear stack under an 'int8' label."
        )

    logger.info(
        f"[DeployFbgemm] {label}: confirmed quantized -- "
        f"Conv2d '{conv_hit}' -> {type(dict(model.named_modules())[conv_hit]).__name__}, "
        f"Linear '{linear_hit}' -> {type(dict(model.named_modules())[linear_hit]).__name__}"
    )
    return True, True


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def _evaluate_accuracy(model: nn.Module, loader, eval_subset_batches: int | None = None) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if eval_subset_batches is not None and i >= eval_subset_batches:
                break
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Throughput sweep (CPU: no cuda synchronize, fixed thread count)
# ---------------------------------------------------------------------------

def _benchmark_sweep(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    base_input_shape: tuple,
    batch_sizes: tuple = SWEEP_BATCH_SIZES,
) -> list[dict]:
    device = torch.device("cpu")
    rows = []
    for batch in batch_sizes:
        input_shape = (batch,) + base_input_shape
        fp32_result = benchmark_latency(fp32_model, input_shape, device, warmup=LATENCY_WARMUP, iters=LATENCY_ITERS)
        int8_result = benchmark_latency(int8_model, input_shape, device, warmup=LATENCY_WARMUP, iters=LATENCY_ITERS)
        rows.append({
            "batch": batch,
            "fp32_latency_ms": fp32_result["latency_ms"],
            "int8_latency_ms": int8_result["latency_ms"],
            "fp32_throughput_ips": fp32_result["throughput_ips"],
            "int8_throughput_ips": int8_result["throughput_ips"],
            "speedup_x": fp32_result["latency_ms"] / int8_result["latency_ms"],
        })
    return rows


# ---------------------------------------------------------------------------
# CPU info log
# ---------------------------------------------------------------------------

def _write_cpu_info(path: str, num_threads: int) -> None:
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
    except OSError:
        cpuinfo = ""

    model_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    cpu_model = model_match.group(1).strip() if model_match else "unknown"

    flags_match = re.search(r"flags\s*:\s*(.+)", cpuinfo)
    flags = flags_match.group(1).split() if flags_match else []
    has_avx512 = any(flag.startswith("avx512") for flag in flags)
    has_vnni = "avx512_vnni" in flags

    lines = [
        f"CPU model: {cpu_model}",
        f"Threads used (torch.set_num_threads): {num_threads}",
        f"AVX-512 available: {has_avx512}",
        f"AVX-512 VNNI available: {has_vnni}",
        f"torch.__version__: {torch.__version__}",
        f"Active quantized engine: {torch.backends.quantized.engine}",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"[DeployFbgemm] CPU info written -> {path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_deploy_cpu_fbgemm(
    load_run_id: str | None,
    checkpoint_dir: str | None,
    eval_subset: int | None,
) -> None:
    torch.backends.quantized.engine = "fbgemm"
    num_threads = os.cpu_count() or 1
    torch.set_num_threads(num_threads)
    logger.info(f"[DeployFbgemm] backend=fbgemm num_threads={num_threads}")

    resolved_checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir, load_run_id)

    # CSV_DIR/LOG_DIR (src.utility.config) are already results/<RUN_ID>/{csv,logs} --
    # the same per-run directories every other mode in src.main writes to.
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    _write_cpu_info(os.path.join(LOG_DIR, "cpu_info.txt"), num_threads)

    summary_rows: list[dict] = []
    sweep_rows: list[dict] = []

    for dataset_name in DATASETS:
        specs = DATASET_SPECS[dataset_name]
        channels, image_size = specs["channels"], specs["image_size"]

        logger.info(f"[DeployFbgemm] Loading dataset: {dataset_name}")
        train_loader, val_loader, loaded_num_classes = get_data_loaders(dataset_name)

        subset_for_this_dataset = eval_subset if dataset_name == "IMAGENET100" else None

        for model_name in MODELS:
            for stage in STAGES:
                label = f"{stage} {model_name}/{dataset_name}"
                logger.info(f"[DeployFbgemm] --- {label} ---")

                checkpoint_path = _checkpoint_path(resolved_checkpoint_dir, stage, model_name, dataset_name)

                baked_model = _build_baked_model(
                    model_name=model_name,
                    checkpoint_path=checkpoint_path,
                    num_classes=loaded_num_classes,
                    channels=channels,
                    image_size=image_size,
                )

                int8_model, qconfig = _apply_fbgemm_ptq(
                    copy.deepcopy(baked_model), train_loader, label, model_name,
                )
                _audit_quantized_modules(int8_model, label)

                fp32_acc = _evaluate_accuracy(baked_model, val_loader, subset_for_this_dataset)
                int8_acc = _evaluate_accuracy(int8_model, val_loader, subset_for_this_dataset)
                acc_delta = int8_acc - fp32_acc
                logger.info(
                    f"[DeployFbgemm] {label}: fp32_acc={fp32_acc:.2f}% int8_acc={int8_acc:.2f}% "
                    f"delta={acc_delta:+.2f}%"
                )

                fp32_size_bytes = model_size_bytes(baked_model)
                int8_size_bytes = model_size_bytes(int8_model)
                size_reduction_x = fp32_size_bytes / int8_size_bytes
                logger.info(
                    f"[DeployFbgemm] {label}: fp32_size={fp32_size_bytes / 1024**2:.2f}MB "
                    f"int8_size={int8_size_bytes / 1024**2:.2f}MB reduction={size_reduction_x:.2f}x"
                )

                summary_rows.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "stage": stage,
                    "backend": "fbgemm",
                    "num_threads": num_threads,
                    "fp32_acc": fp32_acc,
                    "int8_acc": int8_acc,
                    "acc_delta": acc_delta,
                    "fp32_size_mb": fp32_size_bytes / 1024**2,
                    "int8_size_mb": int8_size_bytes / 1024**2,
                    "size_reduction_x": size_reduction_x,
                    "conv_quantized": True,
                    "linear_quantized": True,
                    "eval_subset_batches": subset_for_this_dataset if subset_for_this_dataset is not None else "",
                })

                base_input_shape = (channels, image_size, image_size)
                for row in _benchmark_sweep(baked_model, int8_model, base_input_shape):
                    sweep_rows.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "stage": stage,
                        **row,
                    })

                del baked_model, int8_model

                # Re-saved after every combo (not just at the end) so a late
                # failure on a slow CPU run (IMAGENET100 at 224x224, batch
                # sweep up to 128) doesn't discard everything completed so far.
                _save_csv(summary_rows, os.path.join(CSV_DIR, "fbgemm_summary.csv"), SUMMARY_FIELDNAMES)
                _save_csv(sweep_rows, os.path.join(CSV_DIR, "fbgemm_sweep.csv"), SWEEP_FIELDNAMES)

    logger.info("[DeployFbgemm] === Deploy-CPU-fbgemm complete ===")
