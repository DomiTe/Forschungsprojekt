import torch
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

class CustomeAffineObserver(MinMaxObserver):
    """
    Implements Affine Quantization (Asymmetric).
    Maps [min, max] -> [0, 255] directly.
    """
    def calculate_qparams(self):
        # 1. Get min/max from the observed data
        min_val, max_val = self.min_val, self.max_val
        
        # Guard against empty tensors
        if min_val == float("inf") or max_val == float("-inf"):
            return torch.tensor([1.0]), torch.tensor([0])
            
        # 2. Get q_min and q_max (e.g., 0 and 255)
        q_min, q_max = self.quant_min, self.quant_max
        
        # 3. Calculate Scale (S)
        # Formula: S = (x_max - x_min) / (q_max - q_min)
        numerator = max_val - min_val
        denominator = float(q_max - q_min)
        
        scale = numerator / denominator
        scale = torch.max(scale, torch.tensor(1e-6)) # Avoid division by zero
        
        # 4. Calculate Zero Point (Z)
        # Formula: Z = round(q_min - x_min / S)
        zero_point = q_min - (min_val / scale)
        zero_point = torch.round(zero_point)
        
        # Clamp Z to valid int8 range so PyTorch doesn't crash
        zero_point = torch.clamp(zero_point, q_min, q_max)
        
        return scale, zero_point.int()

# ==========================================
# 1. SYMMETRIC ACTIVATION
# ==========================================
class CustomeSymmetricActivationObserver(MinMaxObserver):
    """
    For ACTIVATIONS (Per-Tensor, Unsigned quint8).
    Uses 'Midpoint Quantization' to handle negative inputs on CPU.
    """
    def calculate_qparams(self):
        min_val, max_val = self.min_val, self.max_val
        
        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
        
        # 2. Scale Calculation (Full Symmetric Range)
        # Maps [-max_abs, +max_abs] to 254 levels
        # Formula: Scale = Total_Range / Levels
        scale = (2 * max_abs) / 254.0
        scale = torch.max(scale, torch.tensor(1e-6))
        
        # 3. Force Midpoint Zero Point
        # Maps real 0.0 -> integer 128.
        # Required for FBGEMM to handle negative inputs with unsigned int8.
        zero_point = torch.tensor(128).int()

        return scale, zero_point

# ==========================================
# 2. SYMMETRIC WEIGHT
# ==========================================
class CustomeSymmetricWeightObserver(PerChannelMinMaxObserver):
    """
    For WEIGHTS (Per-Channel, Signed qint8).
    Strictly follows Standard Symmetric mapping [-127, 127].
    """
    def calculate_qparams(self):
        min_val, max_val = self.min_val, self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
             return torch.tensor([1.0]), torch.tensor([0])
        
        # 1. Find absolute max to center range at 0
        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))

        # 2. Scale Calculation (Signed Int8)
        # Maps [0, max_abs] to [0, 127]. 
        # Denominator is 127.0 for qint8.
        denominator = 127.0 
        scale = max_abs / denominator
        scale = torch.max(scale, torch.tensor(1e-6))
        
        # 3. Force Zero Point to 0
        # Strict definition of Symmetric Quantization.
        zero_point = torch.zeros_like(scale).int()
        
        return scale, zero_point.int()

# ==========================================
# 3. PoT OBSERVERS
# ==========================================
class CustomePoTActivationObserver(MinMaxObserver):
    """
    For ACTIVATIONS (Power-of-Two).
    Combines PoT Scaling with Midpoint ZP for CPU compatibility.
    """
    def calculate_qparams(self):
        min_val, max_val = self.min_val, self.max_val
        if min_val == float("inf") or max_val == float("-inf"):
            return torch.tensor([1.0]), torch.tensor([0])
            
        # 1. Percentile Clipping 
        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))

        # 2. Calculate Initial Scale
        # Maps symmetric range to 254 levels
        scale = (2 * max_abs) / 254.0
        scale = torch.max(scale, torch.tensor(1e-6))
        
        # 3. Power-of-Two Rounding
        # Force scale to be 2^k (e.g., 2^-5, 2^-4)
        scale = torch.pow(2.0, torch.round(torch.log2(scale)))
        
        # 4. Force Midpoint Zero Point
        # Essential for accuracy on signed inputs with quint8.
        zero_point = torch.tensor(128).int()

        return scale, zero_point

class CustomePoTWeightObserver(PerChannelMinMaxObserver):
    """
    For WEIGHTS (Power-of-Two, Per-Channel).
    Constrains symmetric scales to nearest 2^k.
    """
    def calculate_qparams(self):
        min_val, max_val = self.min_val, self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
             return torch.tensor([1.0]), torch.tensor([0])

        # 1. Find absolute max
        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
        
        # 2. Calculate Signed Scale
        scale = max_abs / 127.0
        scale = torch.max(scale, torch.tensor(1e-6))
        
        # 3. Power-of-Two Rounding
        # Rounds log2(scale) to nearest integer k, then 2^k
        scale = torch.pow(2.0, torch.round(torch.log2(scale)))
        
        # 4. Force Zero Point to 0
        return scale, torch.zeros_like(scale).int()