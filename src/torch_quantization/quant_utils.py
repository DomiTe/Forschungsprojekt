import torch
import logging

from torch.ao.quantization import (
    QConfig,
    fuse_modules,
)

from src.model_cnn.model import CNN

from src.torch_quantization.custom_observer import (
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
    Pattern: [Conv, BatchNorm,  ReLU] -> [FusedConvBnReLU]
    """
    model.eval()
    
    # Define layers for fusion based on model architecture
    # Fully-Connected Layers wont be fused for better comparison of conv- and fc-layers
    # Because only fc1 would be fused with relu5 but since fc2 doesnt have a relu activation I wanted it to be uniform for the CNN-architecture
    fusion_candidates = [
        ['conv1', 'bn1', 'relu1'],
        ['conv2', 'bn2', 'relu2'],
        ['conv3', 'bn3', 'relu3'],
        ['conv4', 'bn4', 'relu4'],
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
            reduce_range=False,
            qscheme=torch.per_tensor_affine
        ),
        weight=CustomeAffineObserver.with_args(
            dtype=torch.qint8,
            quant_min=-127,
            quant_max=127,
            reduce_range=False,
            qscheme=torch.per_tensor_affine
        )
    )

def get_custome_symmetric_qconfig():
    """
    Symmetric Quantization (Hardware-Aware).
    - Activations: [-127, 127 ] with ZP=0.
    - Weights: Signed [-127, 127] with ZP=0.
    - Note: Due to Relu and Torch.AO. restriction we only look at Act >= 0 (uint8) values.
    """
    return torch.ao.quantization.QConfig(
        activation=CustomeSymmetricActivationObserver.with_args(
            dtype=torch.quint8, 
            quant_min=-128, 
            quant_max=127, 
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
    - Activations: [-127, 127 ] with ZP=0.
    - Weights: Strict Symmetric (ZP=0) with PoT Scales (2^k).
    - Note: Due to Relu and Torch.AO. restriction we only look at Act >= 0 (uint8) values.
    """
    return torch.ao.quantization.QConfig(
        activation=CustomePoTActivationObserver.with_args(
            dtype=torch.quint8,
            quant_min=-127,
            quant_max=127,
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