import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class Quantization:
    @staticmethod
    def _core_quantization(tensor, scale, zero_point, q_min, q_max):
        x_int = torch.round((tensor / scale) + zero_point)
        x_int = torch.clamp(x_int, q_min, q_max)
        x_dequant = (x_int - zero_point) * scale

        return x_dequant, x_int
    
    @staticmethod
    def affine_quantization(
        tensor: torch.Tensor, num_bits: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x_min = tensor.min()
        x_max = tensor.max()
        q_min = 0
        q_max = 2**num_bits - 1

        if x_max == x_min:
            logger.warning("Tensor min/max are identical in affine quantization.")
            return tensor, tensor.to(torch.uint8), torch.tensor(1.0), torch.tensor(0.0)

        scale = (x_max - x_min) / float(q_max - q_min)
        zero_point = torch.round(q_min - (x_min / scale)).to(torch.int32)

        logger.debug(
            f"Affine Quantization: Min={x_min:.6f}, Max={x_max:.6f}, "
            f"Scale={scale:.8f}, Zero Point:{zero_point.item()}"
        )

        x_dequant, x_int = Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)

        return x_dequant, x_int.to(torch.uint8), scale, zero_point.to(torch.int32)

    @staticmethod
    def symmetric_quantization(
        tensor: torch.Tensor, num_bits: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        q_max = (2 ** (num_bits - 1)) - 1
        q_min = -q_max
        max_val = torch.max(torch.abs(tensor))

        if max_val == 0:
            scale = torch.tensor(1.0)
            logger.debug("Symmetric Quantization: Max value is 0, setting scale to 1.0")
        else:
            scale = max_val / q_max

        logger.debug(
            f"Symmetric Quantization: Max value={max_val:.4f}, Scale={scale:.6f}, "
            f"Range=[{q_min}, {q_max}]"
        )

        zero_point = torch.tensor(0.0).to(tensor.device)

        x_dequant, x_int = Quantization._core_quantization(tensor, scale, zero_point, q_min, q_max)

        return x_dequant, x_int.to(torch.int8), scale, zero_point

    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, method: str= 'symmetric', num_bits: int = 8):
        if method == 'affine':
            return Quantization.affine_quantization(tensor, num_bits=num_bits)
        else:
            return Quantization.symmetric_quantization(tensor, num_bits=num_bits)
        
    @staticmethod
    def quantize_with_params(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, method: str = 'symmetric', num_bits = 8):
        if method == 'affine':
            q_min = 0
            q_max = 2**num_bits - 1
            if float(scale) == 0:
                logger.warning("Scale is zero in quantize_with_params (affine). replacing with 1.0")
                scale = torch.tensor(1.0, device= tensor.device)
            x_int = torch.round(tensor / scale)
            x_int = torch.clamp(x_int, q_min, q_max)
            x_dequant = x_int * scale
            return x_dequant, x_int.to(torch.int8)
        else:
            q_max = (2 ** (num_bits - 1)) - 1
            q_min = -q_max
            
            x_int = torch.round(tensor / scale)
            x_int = torch.clamp(x_int, q_min, q_max)
            x_dequant = x_int * scale
            return x_dequant, x_int.to(torch.int8)
        
    @staticmethod
    def dequantize_from_int(int_tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, method: str = 'symmetric'):
        if int_tensor is None:
            return None
        if method == 'affine':
            return (int_tensor.float() - float(zero_point)) * float(scale)
        else:
            return int_tensor.float() *float(scale)
                            