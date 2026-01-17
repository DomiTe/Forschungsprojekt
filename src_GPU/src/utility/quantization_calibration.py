import torch
from src.layers import QuantizedLayerMixin
from src.utility.quantizer import Quantization
import logging

logger = logging.getLogger(__name__)

def calibrate_model(model, loader, num_batches=20, device=torch.device('cpu'), percentile=0.9999):
    """
    Strict Symmetric Calibration.
    Forces Zero Point = 0 for all layers.
    """
    model.eval()
    logger.info(f"Start Calibration (Symmetric, {num_batches} batches) on {device}...")
    
    stats = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            # Capture Input
            x_in = input[0].detach()
            if x_in.numel() > 0:
                update_stats(stats, name, "in", x_in, percentile)
            
            # Capture Output
            x_out = output.detach()
            if x_out.numel() > 0:
                update_stats(stats, name, "out", x_out, percentile)
        return hook

    def update_stats(stats_dict, layer_name, side, tensor, p):
        t_flat = tensor.view(-1)
        MAX_SAMPLES = 200000 
        if t_flat.numel() > MAX_SAMPLES:
            step = t_flat.numel() // MAX_SAMPLES
            t_flat = t_flat[::step]
            
        t_flat = t_flat.float()
        batch_min = torch.quantile(t_flat, 1.0 - p)
        batch_max = torch.quantile(t_flat, p)

        key = f"{layer_name}_{side}"
        if key not in stats_dict:
            stats_dict[key] = {"min": batch_min, "max": batch_max}
        else:
            stats_dict[key]["min"] = torch.min(stats_dict[key]["min"], batch_min)
            stats_dict[key]["max"] = torch.max(stats_dict[key]["max"], batch_max)

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayerMixin):
            hooks.append(module.register_forward_hook(get_hook(name)))

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches: break
            images = images.to(device)
            model(images)

    for h in hooks: h.remove()

    # Apply Stats (Strict Symmetric)
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayerMixin):
            
            def set_symmetric_params(target_scale, target_zp, key_in_stats):
                if key_in_stats in stats:
                    x_min = stats[key_in_stats]["min"]
                    x_max = stats[key_in_stats]["max"]
                    # Symmetric = Max(Abs(Min), Abs(Max))
                    abs_max = torch.max(x_max.abs(), x_min.abs())
                    
                    _, _, scale, zp = Quantization.symmetric_quantization(
                        torch.tensor([-abs_max, abs_max], device=device), 
                        num_bits=module.num_bits
                    )
                    target_scale.copy_(scale)
                    target_zp.copy_(zp) # Should be 0.0

            # 1. Input Scale
            set_symmetric_params(module.act_scale, module.act_zero_point, f"{name}_in")
            
            # 2. Output Scale
            set_symmetric_params(module.out_scale, module.out_zero_point, f"{name}_out")
            
            module.activation_calibrated = True
            count += 1
            
    logger.info(f"Calibration complete. {count} Layers updated (Strict Symmetric).")