import torch
import logging


# PyTorch Quantization Imports
from torch.ao.quantization import (
    QConfig,
    fuse_modules,
)

# Project Imports
from src.model import CNN

from src.custom_observer import (
    CustomeAffineObserver,
    CustomeSymmetricActivationObserver,
    CustomeSymmetricWeightObserver,
    CustomePoTActivationObserver,
    CustomePoTWeightObserver
)

logger = logging.getLogger("Experiment")

def fuse_layers(model):
    """
    Fuses layers to improve inference speed and accuracy.
    Pattern: [Conv, ReLU] -> [FusedConvReLU]
    Note: If Batch Norm exists, fuse as ['conv', 'bn', 'relu'].
    """
    model.eval() # Fusion requires eval mode
    
    # Define candidates for fusion based on model architecture
    fusion_candidates = [
        ['conv1', 'relu1'],
        ['conv2', 'relu2'],
        ['conv3', 'relu3'],
        ['conv4', 'relu4'],
        ['fc1', 'relu5'] 
    ]
    
    # Only fuse layers that actually exist
    existing_fusions = [
        f for f in fusion_candidates 
        if hasattr(model, f[0]) and hasattr(model, f[1])
    ]
    
    if existing_fusions:
        fuse_modules(model, existing_fusions, inplace=True)
        logger.info(f"Fused layers: {existing_fusions}")
    else:
        logger.warning("No layers fused.")

    return model

# ==========================================
# QUANTIZATION CONFIGURATIONS
# ==========================================

def get_custome_affine_qconfig():
    """
    Affine (Asymmetric) Quantization.
    - Activations: [0, 255] (quint8)
    - Weights: [-128, 127] (qint8)
    - Zero Point: Calculated to map range precisely.
    """
    return QConfig(
        activation=CustomeAffineObserver.with_args(
            dtype=torch.quint8,
            quant_min=0,
            quant_max=255,
            reduce_range=False
        ),
        weight=CustomeAffineObserver.with_args(
            dtype=torch.qint8,
            quant_min=-128,
            quant_max=127,
            reduce_range=False
        )
    )

def get_custome_symmetric_qconfig():
    """
    Symmetric Quantization (Hardware-Aware).
    - Activations: Unsigned [0, 255] with ZP=128 (Midpoint) to handle negative inputs on CPU.
    - Weights: Signed [-127, 127] with ZP=0.
    """
    return torch.ao.quantization.QConfig(
        activation=CustomeSymmetricActivationObserver.with_args(
            dtype=torch.quint8, 
            quant_min=0, 
            quant_max=255, 
            reduce_range=False,
            qscheme=torch.per_tensor_symmetric
        ),
        weight=CustomeSymmetricWeightObserver.with_args(
            dtype=torch.qint8, 
            quant_min=-127, 
            quant_max=127, 
            reduce_range=False,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0
        )
    )

def get_custome_pot_qconfig():
    """
    Power-of-Two (PoT) Quantization.
    - Activations: Midpoint Power of Two (ZP=128) with PoT Scales.
    - Weights: Strict Symmetric (ZP=0) with PoT Scales (2^k).
    """
    return torch.ao.quantization.QConfig(
        activation=CustomePoTActivationObserver.with_args(
            dtype=torch.quint8,
            quant_min=0,
            quant_max=255,
            reduce_range=False,
            qscheme=torch.per_tensor_symmetric
        ),
        weight=CustomePoTWeightObserver.with_args(
            dtype=torch.qint8,
            quant_min=-127,
            quant_max=127,
            reduce_range=False,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0
        )
    )