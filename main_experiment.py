import torch
import copy
import os
from src.quantizer import Quantization
from src.model import CNN
from src.utils import get_data_loaders, get_model_size
from src.train import train_model
from src.config import DEVICE, MODEL_SAVE_PATH
from src.evaluate_model import evaluate
from src.logging import get_logger
from src.logging_config import setup_logging
from src.layers import replace_layers_with_quantizable

logger = get_logger("Experiment")


def run_experiment():
    if torch.cuda.is_available():
        logger.info(f"Config: Using NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    else:
        logger.info("Config: Using CPU")

    logger.info("Initializing Data Loaders...")
    _, test_loader = get_data_loaders()

    if not os.path.exists(MODEL_SAVE_PATH):
        logger.warning(f"Model not found at {MODEL_SAVE_PATH}. Training new model.")
        model = train_model()
    else:
        logger.info(f"Loading existing model from {MODEL_SAVE_PATH}")
        model = CNN().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    acc_base, time_base = evaluate(model, test_loader, "Baseline Float32")
    size_base = get_model_size(MODEL_SAVE_PATH)

    # Quantization Affine
    logger.info("Applying 8-bit Affine Quantization... (Simulated)")
    quant_model = copy.deepcopy(model)
    quant_model.to(DEVICE)

    for name, module in quant_model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            logger.debug(f"Quantization layer (affine): {name}")
            float_weights = module.weight.data
            dequant_weights, _, _, _ = Quantization.affine_quantization(
                float_weights, num_bits=8
            )
            module.weight.data = dequant_weights.to(module.weight.dtype)

    acc_affine, time_affine = evaluate(quant_model, test_loader, "Quanitzed 8-bit")

    # Quantization Symmetric
    logger.info("Applying 8-bit Symmetric Quantization... (Simulated)")
    quant_model_sym = copy.deepcopy(model)
    quant_model_sym.to(DEVICE)

    for name, module in quant_model_sym.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            logger.debug(f"Quantizing layer (symmetric): {name}")
            float_weights = module.weight.data
            dequant_weights,_, _, _ = Quantization.symmetric_quantization(
                float_weights, num_bits=8
            )
            module.weight.data = dequant_weights.to(module.weight.dtype)
    acc_symm, time_symm = evaluate(quant_model_sym, test_loader, "Symmetric 8-bit")

    # Actual Storage Quantization Experiment

    logger.info("Starting Actual Storage Quantization (Int8)..")

    storage_model = CNN().to('cpu')
    storage_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))

    storage_model = replace_layers_with_quantizable(storage_model)
    logger.info("Model layers replaced with Quantizable versions")

    for name, module in storage_model.named_modules():
        if hasattr(module, 'quantized_storage'):
            module.quantized_storage(num_bits=8, method='affine')
    quantized_path = 'model_quantized_int8.pt'

    torch.save(storage_model.state_dict(), quantized_path)

    original_size = os.path.getsize(MODEL_SAVE_PATH) / 1024
    quantized_size = os.path.getsize(quantized_path) / 1024
    reduction = 100 * (1 - quantized_size / original_size)

    logger.info(f"Original Size: {original_size:.2f} KB")
    logger.info(f"Quantized Size: {quantized_size:.2f} KB")
    logger.info(f"Reduction: {reduction:.2f}%")

    storage_model.to(DEVICE)
    acc_storage, time_storage = evaluate(storage_model, test_loader, "Storage Quantization (Int8)")


    report = (
        "\n==========================================\n"
        "         Result Report                      \n"
        "==========================================\n"
        f"Model Size (Float32):   {size_base:.2f} MB\n"
        f"Theor. Size (Int8):      {size_base / 4:.2f} MB\n"
        "------------------------------------------\n"
        f"Accuracy Float32:         {acc_base:.2f}%\n"
        f"Accuracy Affine:        {acc_affine:.2f}%\n"
        f"Accuracy Symmetric:       {acc_symm:.2f}%\n"
        "=========================================="
    )
    logger.info("Experiment finished. Summary: " + report)


if __name__ == "__main__":
    setup_logging()
    run_experiment()


# TODO
# Logging + (Analysing Functions --> Research whats best)
# Moving to Uni GPU Servers


# Full Implementation of Quantisation -> Different and Unique Models based on Functions

# DONE UNTIL 8th of Dec

# Bigger Dataset for Training during christmas
# --> Helper Function for dataset integration
# --> making it actually work lol

# DONE UNTIL WEEK BEFORE CHRISTMAS
