import torch
import torch.nn.functional as F

def calculate_fidelity_metrics(baseline_logits, quant_logits):
    """
    Calculates metrics comparing how much the quantized model 
    deviates from the baseline.
    """
    # 1. MSE (Direct difference)
    mse = F.mse_loss(baseline_logits, quant_logits).item()
    
    # 2. SQNR (Signal-to-Quantization-Noise Ratio)
    signal_power = torch.mean(baseline_logits**2)
    noise_power = torch.mean((baseline_logits - quant_logits)**2)
    sqnr = 10 * torch.log10(signal_power / (noise_power + 1e-10)).item()
    
    # 3. KL-Divergence (Probability distribution shift)
    p = F.softmax(baseline_logits, dim=1)
    log_q = F.log_softmax(quant_logits, dim=1)
    kl_div = F.kl_div(log_q, p, reduction='batchmean').item()
    
    return mse, sqnr, kl_div