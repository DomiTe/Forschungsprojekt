import torch
import logging
from src.layers import QuantizedLayerMixin
from src.utility.quantizer import Quantization

logger = logging.getLogger(__name__)

def calibrate_model(model, loader, num_batches=20, device=torch.device('cpu'), percentile=0.9999):
    """
    Hardware-Aware Symmetric Calibration for CPU.
    Calculates scale based on Symmetric Abs Max, but sets Zero Point to 128
    to map the floating-point zero to the midpoint of the unsigned quint8 range.
    """
    model.eval()
    logger.info(f"Start Calibration (Hardware-Aware Symmetric, {num_batches} batches)...")
    
    stats = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            x_in = input[0].detach()
            if x_in.numel() > 0:
                update_stats(stats, name, "in", x_in, percentile)
            x_out = output.detach()
            if x_out.numel() > 0:
                update_stats(stats, name, "out", x_out, percentile)
        return hook

    def update_stats(stats_dict, layer_name, side, tensor, p):
        t_flat = tensor.view(-1).float()
        # Efficiency: Cap samples for quantile calculation
        if t_flat.numel() > 200000:
            indices = torch.randperm(t_flat.numel())[:200000]
            t_flat = t_flat[indices]
            
        batch_min = torch.quantile(t_flat, 1.0 - p)
        batch_max = torch.quantile(t_flat, p)

        key = f"{layer_name}_{side}"
        if key not in stats_dict:
            stats_dict[key] = {"min": batch_min, "max": batch_max}
        else:
            stats_dict[key]["min"] = torch.min(stats_dict[key]["min"], batch_min)
            stats_dict[key]["max"] = torch.max(stats_dict[key]["max"], batch_max)

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayerMixin):
            hooks.append(module.register_forward_hook(get_hook(name)))

    # Run data through model
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches: break
            images = images.to(device)
            model(images)

    # Remove hooks
    for h in hooks: h.remove()

    # Apply Stats
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayerMixin):
            
            def apply_hw_symmetric(target_scale, target_zp, key):
                if key in stats:
                    s = stats[key]
                    # Find symmetric absolute max
                    abs_max = torch.max(s["max"].abs(), s["min"].abs())
                    
                    # Scale for a symmetric range mapped to 255 levels
                    # range is [-abs_max, abs_max], distance is 2 * abs_max
                    scale = (2 * abs_max) / 255.0
                    
                    # Midpoint for unsigned hardware compatibility
                    zp = 128.0
                    
                    target_scale.copy_(scale)
                    target_zp.copy_(zp)

            apply_hw_symmetric(module.act_scale, module.act_zero_point, f"{name}_in")
            apply_hw_symmetric(module.out_scale, module.out_zero_point, f"{name}_out")
            
            module.activation_calibrated = True
            
    logger.info("Calibration complete. Midpoint symmetry applied for CPU.")