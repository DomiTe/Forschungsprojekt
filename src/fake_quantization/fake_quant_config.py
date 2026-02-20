import torch
from torch.ao.quantization import FakeQuantize

# Project Imports
from src.torch_quantization.custom_observer import (
    CustomeAffineObserver,
    CustomeSymmetricActivationObserver,
    CustomeSymmetricWeightObserver,
    CustomePoTActivationObserver,
    CustomePoTWeightObserver
)

# ==========================================
# FAKE QUANTIZATION CONFIGS (FOR ANALYSIS)
# ==========================================

def get_fake_quant_affine_config():
    """
    Fake Affine (Asymmetric) Quantization.
    Simulates [0, 255] range with zero-point shifting.
    """
    return torch.ao.quantization.QConfig(
        activation=FakeQuantize.with_args(
            observer=CustomeAffineObserver,
            dtype=torch.quint8,
            quant_min=0,
            quant_max=255,
            reduce_range=False,
            qscheme=torch.per_tensor_affine
        ),
        weight=FakeQuantize.with_args(
            observer=CustomeAffineObserver,
            dtype=torch.qint8,
            quant_min=-127,
            quant_max=127,
            reduce_range=False,
            qscheme=torch.per_tensor_affine,
            ch_axis=0
        )
    )

def get_fake_quant_symmetric_config():
    """
    Fake Symmetric Quantization.
    Simulates [-128, 127] range with zero-point fixed at 0.
    """
    return torch.ao.quantization.QConfig(
        activation=FakeQuantize.with_args(
            observer=CustomeSymmetricActivationObserver,
            dtype=torch.qint8, 
            quant_min=-128, 
            quant_max=127, 
            reduce_range=False,
            qscheme=torch.per_tensor_symmetric
        ),
        weight=FakeQuantize.with_args(
            observer=CustomeSymmetricWeightObserver,
            dtype=torch.qint8, 
            quant_min=-127, 
            quant_max=127, 
            reduce_range=False,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0
        )
    )

def get_fake_quant_pot_config():
    """
    Fake Power-of-Two (PoT) Quantization.
    Simulates symmetric quantization where the scale is constrained to 2^k.
    """
    return torch.ao.quantization.QConfig(
        activation=FakeQuantize.with_args(
            observer=CustomePoTActivationObserver,
            dtype=torch.qint8,
            quant_min=-128,
            quant_max=127,
            reduce_range=False,
            qscheme=torch.per_tensor_symmetric
        ),
        weight=FakeQuantize.with_args(
            observer=CustomePoTWeightObserver,
            dtype=torch.qint8,
            quant_min=-127,
            quant_max=127,
            reduce_range=False,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0
        )
    )