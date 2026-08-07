import math
import torch.nn as nn
import logging

from src.quantization.quantizer import QuantizedConv2d, QuantizedLinear

logger = logging.getLogger(__name__)


def _fp32_weights(model: nn.Module) -> dict:
    return {
        name: param for name, param in model.named_parameters()
        if param.requires_grad and "weight" in name and param.dim() >= 2
    }


def _quantized_effective_weights(model: nn.Module) -> dict:
    results = {}
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            results[f"{name}.weight"] = module.weight_fake_quant(module.weight)
    return results


def compute_layerwise_quant_error(fp32_model: nn.Module, quant_model: nn.Module) -> dict:
    """
    Compares FP32 weights against the effective post-quantization weights of a quantized model.

    Returns:
        dict of layer_name -> {"mse": float, "sqnr": float}
    """
    fp32_params = _fp32_weights(fp32_model)
    quant_params = _quantized_effective_weights(quant_model)

    results = {}
    for name, fp32_weight in fp32_params.items():
        if name not in quant_params:
            logger.warning(f"Layer '{name}' not found in quantized model, skipping.")
            continue

        quant_weight = quant_params[name].to(fp32_weight.device)

        diff = fp32_weight - quant_weight
        mse = (diff ** 2).mean().item()
        signal_power = (fp32_weight ** 2).mean().item()

        if mse <= 1e-12:
            sqnr = float("inf")
        else:
            sqnr = 10 * math.log10(signal_power / mse)

        results[name] = {"mse": mse, "sqnr": sqnr}

    return results