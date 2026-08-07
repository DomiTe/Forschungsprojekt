"""
deploy.py — the single source of truth for reconstructing "the deployed
int8 model" from a saved PTQ/QAT checkpoint.

Root cause this module fixes: torchao 0.17's Int8DynamicActivationInt8Weig
htConfig has a default filter_fn (_is_linear) that only matches nn.Linear.
Calling quantize_(model, Int8DynamicActivationInt8WeightConfig()) on a
whole model -- as validate_pot.py and the deploy path each did
independently -- silently leaves every Conv2d layer as a plain fp32
nn.Parameter (confirmed: type(model.conv1.weight) is torch.nn.Parameter
after that call, not a torchao quantized tensor subclass). benchmark.py's
apply_int8_quantization already had the fix: a split config (dynamic-
activation int8 for Linear, weight-only int8 for Conv2d via
IntxWeightOnlyConfig -- see its docstring in benchmark.py for the full
investigation of why torchao 0.17 needs two different configs), but that
fix only lived on the benchmark path. Every other path that reconstructed
an int8 model duplicated the reconstruction logic instead of reusing it,
so the unfixed config re-appeared independently -- in validate_pot.py's
per-layer PoT check (every Conv2d layer reporting a meaningless
rel_error=0.0, bit-identical to the fp32 reference) and in the deploy
path's saved "deployed_full_*.pt" checkpoints.

build_int8_model is now the ONLY function that constructs a deployed int8
model; validate_pot.py, the deploy path, and the benchmark path all call
it. It ends with a hard assertion that at least one Conv2d weight and at
least one Linear weight actually became a torchao quantized tensor
subclass, so a partially-quantized model can never again be handed back
silently under an "int8" label.
"""

import os
import copy
import logging

import torch
import torch.nn as nn

from src.analysis.benchmark import apply_int8_quantization

logger = logging.getLogger(__name__)


class Int8BuildError(RuntimeError):
    pass


def build_int8_model(
    model_name: str,
    dataset_name: str,
    stage: str,
    checkpoint_path: str,
    device: torch.device,
    num_classes: int,
    channels: int,
    image_size: int,
) -> tuple[nn.Module, nn.Module, list[dict]]:
    """
    The one, complete reconstruction of the deployed int8 model from a
    saved PTQ/QAT checkpoint:

        build_model -> fuse_model_architectures -> replace_layers_for_quantization
        -> load_state_dict -> bake_pot_into_standard_layers -> apply_int8_quantization

    Never loads a saved int8 checkpoint -- torchao quantized tensors don't
    reliably round-trip through load_state_dict into a fresh skeleton, so
    every deployed int8 model is rebuilt from its PTQ/QAT checkpoint fresh
    every time this is called.

    Returns (baked_model, int8_model, audit_details):
      baked_model   -- fp32 ground-truth PoT weights (exact powers of two),
                        plain nn.Conv2d/nn.Linear, never quantized.
      int8_model    -- baked_model's deployment copy, actually run through
                        apply_int8_quantization.
      audit_details -- the per-layer quantization audit from
                        audit_layer_quantization (via apply_int8_quantization),
                        for callers that want to log/summarize it themselves.

    Raises Int8BuildError, naming model/dataset/stage, if quantization did
    not actually convert at least one Conv2d weight AND at least one Linear
    weight to a torchao quantized tensor subclass. This is a hard failure,
    not a warning: silently returning a partially-quantized model under an
    "int8" label is exactly the bug this function exists to make
    impossible. Do not loosen this check.
    """
    # Deferred import: bake_pot_into_standard_layers lives in main.py, which
    # imports this module (transitively, via validate_pot/benchmark
    # callers) at top level -- importing it at module scope here would be
    # circular. By the time this function actually runs, src.main has
    # finished loading.
    from src.main import bake_pot_into_standard_layers
    from src.model_cnn.train import build_model
    from src.quantization.quantizer import fuse_model_architectures, replace_layers_for_quantization

    label = f"{stage} {model_name}/{dataset_name}"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No saved {stage} model at {checkpoint_path}")

    quant_model = build_model(
        num_classes=num_classes,
        model_name=model_name,
        channels=channels,
        image_size=image_size,
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

    int8_model = copy.deepcopy(baked_model)
    audit_details = apply_int8_quantization(int8_model)
    int8_model.eval()

    _assert_fully_quantized(audit_details, label)

    del quant_model
    return baked_model, int8_model, audit_details


def _assert_fully_quantized(audit_details: list[dict], label: str) -> None:
    conv_hit = next((d for d in audit_details if d["layer_type"] == "Conv2d" and d["is_quantized"]), None)
    linear_hit = next((d for d in audit_details if d["layer_type"] == "Linear" and d["is_quantized"]), None)

    if conv_hit is None:
        raise Int8BuildError(
            f"{label}: no Conv2d layer was quantized -- int8 construction produced an "
            f"fp32 conv stack under an 'int8' label. This is the exact bug build_int8_model "
            f"exists to prevent; do not loosen this check to make it pass."
        )
    if linear_hit is None:
        raise Int8BuildError(
            f"{label}: no Linear layer was quantized -- int8 construction produced an "
            f"fp32 linear stack under an 'int8' label."
        )

    logger.info(
        f"[BuildInt8] {label}: confirmed quantized -- Conv2d '{conv_hit['fqn']}' weight type = "
        f"{conv_hit['weight_type']}; Linear '{linear_hit['fqn']}' weight type = "
        f"{linear_hit['weight_type']}"
    )
