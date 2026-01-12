import torch
from src.layers import QuantizedLayerMixin
from src.utility.quantizer import Quantization
import logging

logger = logging.getLogger(__name__)


def calibrate_model(model, loader, num_batches=10, device=torch.device('cpu')):
    """
    Runs data through the model to determine min/max ranges for activations.
    """
    model.eval()
    logger.info(f"Start Calibration ({num_batches} batches)...")
    
    # 1. Register Hooks to catch input data at every Quantized Layer
    stats = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            # input is a tuple, we want the tensor (input[0])
            x = input[0].detach()
            # Track Global Min/Max for this layer
            if name not in stats:
                stats[name] = {"min": x.min(), "max": x.max()}
            else:
                stats[name]["min"] = torch.min(stats[name]["min"], x.min())
                stats[name]["max"] = torch.max(stats[name]["max"], x.max())
        return hook

    # Attach hooks to all layers that inherit from our Mixin
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayerMixin):
            hooks.append(module.register_forward_hook(get_hook(name)))

    # 2. Run Data
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches: break
            images = images.to(device)
            model(images)

    # 3. Remove Hooks
    for h in hooks: h.remove()

    # 4. Calculate Scale & Zero Point from the collected stats
    # We apply the same method (symmetric/affine) to activations as defined in the layer
    for name, module in model.named_modules():
        if name in stats and isinstance(module, QuantizedLayerMixin):
            x_min = stats[name]["min"]
            x_max = stats[name]["max"]
            
            # Using the layer's stored settings (method/bits) to calculate activation params
            if module.quant_method == 'symmetric':
                _, _, scale, zp = Quantization.symmetric_quantization(
                    torch.tensor([x_min, x_max]), num_bits=module.num_bits
                )
            elif module.quant_method == 'affine':
                dummy = torch.tensor([x_min, x_max])
                _, _, scale, zp = Quantization.affine_quantization(dummy, num_bits=module.num_bits)
            else: # power2 or others
                 # Often activations use symmetric for simplicity, but let's stick to method
                _, _, scale, zp = Quantization.symmetric_quantization(
                    torch.tensor([x_min, x_max]), num_bits=module.num_bits
                )

            # SAVE params into the layer buffers
            module.act_scale.copy_(scale)
            module.act_zero_point.copy_(zp)
            module.activation_calibrated = True
            
    logger.info("Calibration complete. Activation params updated.")