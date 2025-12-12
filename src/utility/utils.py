import torch
from torchvision import datasets, transforms
from typing import Tuple
import os
import copy
from src.utility.logging import get_logger 
from src.quantizer import Quantization
from src.utility.config import PIN_MEMORY, DATA_DIR, BATCH_SIZE, TEST_BATCH_SIZE
from src.evaluation.evaluate_model import evaluate
from src.layers import replace_layers_with_quantizable, calibrated_model_activation, QuantizedConv2d, QuantizedLinear

logger = get_logger(__name__)

def get_data_loaders() -> Tuple[
    torch.utils.data.DataLoader, torch.utils.data.DataLoader
]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    kwargs = {"num_workers": 0, "pin_memory": PIN_MEMORY} if PIN_MEMORY else {}
    train_dataset = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs
    )
    test_dataset = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs
    )
    return train_loader, test_loader


def get_model_size(model_path: str) -> float:
    size = os.path.getsize(model_path)
    return size / (1024 * 1024)

def layer_weight_mse(model_float, model_quant):
    errors = {}
    for (name_f, mod_f), (name_q, mod_q) in zip(model_float.named_modules(), model_quant.named_modules()):
        if isinstance(mod_f, (torch.nn.Conv2d, torch.nn.Linear)) and isinstance(mod_q, (QuantizedConv2d, QuantizedLinear)):
            float_weight = mod_f.weight.data.cpu().float()
            if getattr(mod_q, 'weight_int', None) is None:
                dequant_weight = mod_q.weight.data.cpu().float()
            else:
                dequant_weight = Quantization.dequantize_from_int(mod_q.weight_int.cpu(), mod_q.scale.cpu(), mod_q.zero_point.cpu(), method=mod_q.quant_method)
            mse = torch.mean((float_weight - dequant_weight) ** 2).item()
            errors[name_f] = mse
    return errors

def per_layer_sensitivity_analysis(model_base, test_loader, device, bits=8, method='symmetric'):
    results = []
    
    # Identify layers first
    quantizable_names = [n for n, m in model_base.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]

    logger.info(f"Starting Sensitivity Analysis on {len(quantizable_names)} layers...")

    for layer_name in quantizable_names:
        # 1. Fresh Copy
        model_copy = copy.deepcopy(model_base).to('cpu')
        model_copy = replace_layers_with_quantizable(model_copy)
        
        # 2. Target Specific Layer
        target_found = False
        for name, module in model_copy.named_modules():
            if name == layer_name and hasattr(module, 'quantized_storage'):
                # Quantize Weights for THIS layer
                module.quantized_storage(num_bits=bits, method=method)
                target_found = True
                break
        
        if not target_found:
            continue

        # 3. Calibrate Activations (Required for Full Quantization)
        # We perform a quick calibration on the modified model
        train_loader, _ = get_data_loaders()
        calibrated_model_activation(model_copy, train_loader, device='cpu', num_batches=10, method=method, num_bits=bits)

        # 4. Evaluate
        model_copy.to(device)
        acc, inf_time = evaluate(model_copy, test_loader, f"Sensitivity: {layer_name}")
        
        results.append({
            "layer": layer_name,
            "accuracy": acc,
            "inference_time": inf_time
        })
        
        # Cleanup to save RAM
        del model_copy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results
