import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class Quantization:
    """
    Core Fake Quantization logic.
    Simulates INT8 behavior using Float32 math (Fake Quantization).
    """

    @staticmethod
    def _core_quantization(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, q_min: int, q_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. Quantize: (x / S) + Z
        2. Clamp: [q_min, q_max]
        3. De-quantize: (x_q - Z) * S
        """
        # 1. Quantize
        # Add epsilon to scale to prevent division by zero
        scale = scale + 1e-6
        x_int = torch.round((tensor / scale) + zero_point)
        
        # 2. Clamp
        x_int = torch.clamp(x_int, q_min, q_max)
        
        # 3. De-Quantize
        x_dequant = (x_int - zero_point) * scale

        return x_dequant, x_int

    @staticmethod
    def affine_quantization(tensor: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Asymmetric (Affine). Maps [min, max] -> [0, 2^b - 1].
        """
        t_min, t_max = tensor.min(), tensor.max()
        q_min, q_max = 0, (2 ** num_bits) - 1

        scale = (t_max - t_min) / float(q_max - q_min)
        scale = torch.max(scale, torch.tensor(1e-6))

        zero_point = q_min - (t_min / scale)
        zero_point = torch.clamp(torch.round(zero_point), q_min, q_max)

        x_dequant, x_int = Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)
        return x_dequant, x_int, scale, zero_point

    @staticmethod
    def symmetric_quantization(tensor: torch.Tensor, num_bits: int = 8, is_activation: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Symmetric Quantization.
        - Weights: Signed [-127, 127], ZP=0
        - Activations (if positive): Unsigned [0, 255], ZP=0 (Your "Unnormalized" trick)
        """
        t_abs = torch.abs(tensor).max()
        
        if is_activation:
            # Unsigned Symmetric for Activations (Unnormalized 0-1 input)
            q_min, q_max = 0, (2 ** num_bits) - 1
            # Map [0, max] -> [0, 255]
            scale = t_abs / float(q_max) 
        else:
            # Signed Symmetric for Weights
            q_min = -(2 ** (num_bits - 1)) + 1 # -127
            q_max = (2 ** (num_bits - 1)) - 1  # 127
            # Map [-max, max] -> [-127, 127]
            scale = t_abs / float(q_max)

        zero_point = torch.tensor(0.0).to(tensor.device)
        
        x_dequant, x_int = Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)
        return x_dequant, x_int, scale, zero_point

    @staticmethod
    def power_of_two_quantization(tensor: torch.Tensor, num_bits: int = 8, is_activation: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Power-of-Two (PoT).
        Same limits as Symmetric, but Scale is rounded to nearest 2^k.
        """
        t_abs = torch.abs(tensor).max()
        
        if is_activation:
            q_min, q_max = 0, (2 ** num_bits) - 1
            scale_ideal = t_abs / float(q_max)
        else:
            q_min = -(2 ** (num_bits - 1)) + 1
            q_max = (2 ** (num_bits - 1)) - 1
            scale_ideal = t_abs / float(q_max)

        # Force Scale to 2^k
        # log2(0) protection
        scale_ideal = torch.max(scale_ideal, torch.tensor(1e-6))
        scale = 2 ** torch.round(torch.log2(scale_ideal))
        
        zero_point = torch.tensor(0.0).to(tensor.device)

        x_dequant, x_int = Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)
        return x_dequant, x_int, scale, zero_point
    
    @staticmethod
    def quantize_with_params(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, method: str, num_bits: int = 8):
        """
        Wendet Quantisierung mit BEREITS BEKANNTEN Parametern an.
        Wird im Forward-Pass der Layer benutzt (Inference).
        """
        # Bestimme Limits basierend auf der Methode
        if method == 'affine':
            q_min = 0
            q_max = (2 ** num_bits) - 1
        else:
            # Symmetrisch & Power-of-2 nutzen signed range
            q_max = (2 ** (num_bits - 1)) - 1
            q_min = -q_max
            
        # Sicherheitscheck für Scale
        if scale.numel() == 1 and scale.item() == 0:
            scale = torch.tensor(1.0, device=tensor.device)

        return Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)