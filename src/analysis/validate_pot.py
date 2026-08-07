"""
validate_pot.py — functional PoT-preservation check for deployed int8 models.

torchao 0.17's quantized tensor subclasses (LinearActivationQuantizedTensor,
IntxUnpackedToInt8Tensor, ...) do not support .dequantize() in this version
(NotImplementedError), so the deployed int8 weights cannot be pulled back out
and compared numerically against the PoT weights. Instead this module proves
the claim functionally: for every Conv2d/Linear layer, it feeds the SAME
random input through (a) the full-precision PoT weights (F.conv2d/F.linear,
using the layer's real hyperparameters) and (b) the deployed quantized
layer's own forward(), and compares the outputs. A layer that doesn't carry
the PoT weights will produce a wildly different output; one that does will
match up to int8 rounding + dynamic-activation-quantization error.

Model construction lives in src.quantization.deploy.build_int8_model, not
here -- this module used to reconstruct the int8 model itself via a bare
quantize_(model, Int8DynamicActivationInt8WeightConfig()) call, which
silently left every Conv2d layer unquantized (torchao 0.17's default
filter_fn only matches nn.Linear) and made every Conv2d layer report a
meaningless rel_error=0.0 (bit-identical to the fp32 reference -- not
"validated", just never quantized). Callers now get (baked_model,
int8_model) from build_int8_model, which applies the Conv2d-inclusive
config and asserts both layer types actually got quantized before handing
the model back.
"""

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_BATCH_N = 32
DEFAULT_SPATIAL_SIZE = 16

# rel_error == 0.0 exactly: not_quantized -- a genuinely quantized layer
# always carries some int8-rounding + dynamic-activation-quantization
# error, so bit-identical output means this layer was never touched by
# quantization, not that it matched perfectly. rel_error < ELEVATED:
# validated. ELEVATED <= rel_error < FAILED: elevated (warn, likely a
# wide-dynamic-range channel losing precision). >= FAILED: the layer does
# not carry the PoT weights.
REL_ERROR_ELEVATED_THRESHOLD = 0.05
REL_ERROR_FAILED_THRESHOLD = 0.20

CSV_FIELDNAMES = ["model", "dataset", "stage", "layer", "rel_error", "max_abs_error", "status"]


def classify_status(rel_error: float) -> str:
    # Exact-zero is checked before the "validated" band on purpose: a
    # layer that was never quantized computes with the identical fp32 PoT
    # weight on both sides of the comparison, producing bit-identical
    # (not just small) error. Treating that as "validated" is precisely
    # the blind spot that let unquantized Conv2d layers pass silently.
    if rel_error == 0.0:
        return "not_quantized"
    elif rel_error < REL_ERROR_ELEVATED_THRESHOLD:
        return "validated"
    elif rel_error < REL_ERROR_FAILED_THRESHOLD:
        return "elevated"
    else:
        return "failed"


def compare_pot_vs_int8_layers(
    baked_model: nn.Module,
    int8_model: nn.Module,
    device: torch.device,
    seed: int = DEFAULT_SEED,
    batch_n: int = DEFAULT_BATCH_N,
    spatial_size: int = DEFAULT_SPATIAL_SIZE,
) -> list[dict]:
    """
    Per-layer functional PoT-preservation check.

    For every Conv2d/Linear layer, generates one fixed-seed random input of
    the layer's native shape and runs it through both the full-precision PoT
    reference (F.conv2d/F.linear using baked_model's weight/bias and the
    layer's actual stride/padding/dilation/groups) and the deployed
    int8_model layer's own forward(). Layers are matched by name: walking
    int8_model.named_modules() and looking up the same name in baked_model
    works because quantize_ mutates weights in place rather than
    restructuring the module tree, so the two trees stay structurally
    identical.

    Nonzero error is expected (int8 rounding + torchao's dynamic activation
    quantization both contribute) -- this is not a bit-exactness check, see
    classify_status for the tolerance bands.

    Returns a list of dicts: layer, layer_type, rel_error, max_abs_error, status.
    """
    baked_layers = dict(baked_model.named_modules())
    results: list[dict] = []

    # A CPU generator (moved to device afterward) keeps the sequence
    # reproducible regardless of device type/backend.
    generator = torch.Generator()
    generator.manual_seed(seed)

    with torch.no_grad():
        for name, module in int8_model.named_modules():
            baked_module = baked_layers.get(name)
            if not isinstance(baked_module, (nn.Conv2d, nn.Linear)):
                continue

            weight = baked_module.weight.detach().clone()
            bias = baked_module.bias.detach().clone() if baked_module.bias is not None else None

            if isinstance(baked_module, nn.Linear):
                layer_type = "Linear"
                x = torch.randn(batch_n, baked_module.in_features, generator=generator).to(device)
                reference = F.linear(x, weight, bias)
            else:
                layer_type = "Conv2d"
                x = torch.randn(
                    batch_n, baked_module.in_channels, spatial_size, spatial_size, generator=generator
                ).to(device)
                reference = F.conv2d(
                    x, weight, bias,
                    baked_module.stride, baked_module.padding,
                    baked_module.dilation, baked_module.groups,
                )

            deployed = module(x)

            diff = deployed.float() - reference.float()
            rel_error = diff.norm().item() / (reference.float().norm().item() + 1e-12)
            max_abs_error = diff.abs().max().item()

            results.append({
                "layer": name,
                "layer_type": layer_type,
                "rel_error": rel_error,
                "max_abs_error": max_abs_error,
                "status": classify_status(rel_error),
            })

    return results


def log_and_summarize_pot_validation(
    layer_results: list[dict],
    model_name: str,
    dataset_name: str,
    stage: str,
) -> dict:
    """
    Logs each layer's result at a severity matching its status, then logs
    and returns a summary: fraction of layers validated, and the worst
    layer + its error. Callers should rank-0-guard this call.

    not_quantized layers are logged as ERROR (a hard failure), and are
    called out explicitly in the summary line by name -- not_quantized
    always carries rel_error=0.0, so "worst layer by rel_error" alone would
    never surface them, silently hiding the exact failure mode this check
    exists to catch (see classify_status).
    """
    label = f"{stage} {model_name}/{dataset_name}"

    for r in layer_results:
        msg = (
            f"[ValidatePoT] {label} {r['layer']} ({r['layer_type']}): "
            f"rel_error={r['rel_error']:.4f} max_abs_error={r['max_abs_error']:.4g} "
            f"[{r['status']}]"
        )
        if r["status"] == "not_quantized":
            logger.error(msg + " -- exact-zero error: this layer was never quantized")
        elif r["status"] == "failed":
            logger.error(msg + " -- deployed layer does not carry the PoT weights")
        elif r["status"] == "elevated":
            logger.warning(msg)
        else:
            logger.info(msg)

    if not layer_results:
        logger.warning(f"[ValidatePoT] {label}: no Conv2d/Linear layers found to validate.")
        return {"validated_fraction": 0.0, "worst_layer": None, "worst_rel_error": None, "not_quantized_count": 0}

    n_validated = sum(1 for r in layer_results if r["status"] == "validated")
    not_quantized_layers = [r["layer"] for r in layer_results if r["status"] == "not_quantized"]
    validated_fraction = n_validated / len(layer_results)
    worst = max(layer_results, key=lambda r: r["rel_error"])

    summary_msg = (
        f"[ValidatePoT] {label} summary: {n_validated}/{len(layer_results)} "
        f"({validated_fraction:.1%}) layers validated | worst layer: {worst['layer']} "
        f"(rel_error={worst['rel_error']:.4f}, status={worst['status']})"
    )

    if not_quantized_layers:
        shown = ", ".join(not_quantized_layers[:5])
        more = ", ..." if len(not_quantized_layers) > 5 else ""
        logger.error(
            summary_msg + f" | {len(not_quantized_layers)} layer(s) NOT QUANTIZED "
            f"(exact rel_error=0.0): {shown}{more}"
        )
    else:
        logger.info(summary_msg)

    return {
        "validated_fraction": validated_fraction,
        "worst_layer": worst["layer"],
        "worst_rel_error": worst["rel_error"],
        "not_quantized_count": len(not_quantized_layers),
    }


def build_csv_rows(
    layer_results: list[dict],
    model_name: str,
    dataset_name: str,
    stage: str,
) -> list[dict]:
    return [
        {
            "model": model_name,
            "dataset": dataset_name,
            "stage": stage,
            "layer": r["layer"],
            "rel_error": r["rel_error"],
            "max_abs_error": r["max_abs_error"],
            "status": r["status"],
        }
        for r in layer_results
    ]


def plot_pot_weight_histogram(
    baked_model: nn.Module,
    model_name: str,
    dataset_name: str,
    stage: str,
    save_dir: str,
) -> str | None:
    """
    Saves a histogram of one small Conv2d/Linear layer's baked PoT weight
    values, to document that what got deployed really is a power-of-two
    comb (weight_fake_quant snaps every value to +/- 2^k). This only ever
    reads baked_model, which is plain fp32 -- it does NOT attempt to
    histogram the deployed int8_model's weights, since extracting those is
    blocked (see module docstring). The functional check in
    compare_pot_vs_int8_layers is the evidence for what was deployed; this
    plot is purely documentation of the PoT structure feeding into it.

    Returns the saved path, or None if the model has no Conv2d/Linear layers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = [
        (name, module) for name, module in baked_model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    if not candidates:
        return None

    name, module = min(candidates, key=lambda nm: nm[1].weight.numel())
    weights = module.weight.detach().cpu().flatten().numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(weights, bins=128)
    ax.set_title(f"Baked PoT weights: {model_name}/{dataset_name} {stage} {name}")
    ax.set_xlabel("weight value (exact powers of two)")
    ax.set_ylabel("count")

    os.makedirs(save_dir, exist_ok=True)
    safe_name = name.replace(".", "_") or "root"
    save_path = os.path.join(
        save_dir, f"pot_weights_{model_name}_{dataset_name}_{stage}_{safe_name}.png"
    )
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
