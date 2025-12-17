import torch
import copy
import os
import csv
from src.quantizer import Quantization
from src.model import CNN
from src.utility.utils import get_data_loaders, get_model_size, layer_weight_mse, per_layer_sensitivity_analysis
from src.train import train_model
from src.utility.config import (
    DEVICE, 
    MODEL_SAVE_PATH,    
    SENSITIVITY_CSV_PATH, 
    MSE_CSV_PATH,
    QUANTIZED_SAVE_PATH
)
from src.evaluation.evaluate_model import evaluate
from src.utility.logging import get_logger
from src.utility.logging_config import setup_logging
from src.layers import replace_layers_with_quantizable, calibrated_model_activation


logger = get_logger("Experiment")

def run_experiment():
    if torch.cuda.is_available():
        logger.info(f"Config: Using NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    else:
        logger.info("Config: Using CPU")

    logger.info("Initializing Data Loaders...")

    train_loader, test_loader, num_classes = get_data_loaders()
    

    if not os.path.exists(MODEL_SAVE_PATH):
        logger.warning(f"Model not found at {MODEL_SAVE_PATH}. Training new model.")
        model = train_model()
    else:
        logger.info(f"Loading existing model from {MODEL_SAVE_PATH}")
        model = CNN(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    acc_base, time_base = evaluate(model, test_loader, "Baseline Float32")
    size_base = get_model_size(MODEL_SAVE_PATH)

    # Simulated quantization affine (weights dequantized) using quantization API
    logger.info("Applying 8-bit Affine Quantization... (Simulated)")
    quant_model = copy.deepcopy(model)
    quant_model.to(DEVICE)

    for name, module in quant_model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            logger.debug(f"Quantization layer (affine): {name}")
            float_weights = module.weight.data
            dequant_weights, _, _, _ = Quantization.quantize_tensor(
                float_weights, method='affine', num_bits=8)
            module.weight.data = dequant_weights.to(module.weight.dtype)

    acc_affine, time_affine = evaluate(quant_model, test_loader, "Simulated Affine 8-bit")

    # Simulated quantization symmetric (weights dequantized) using quantization API
    logger.info("Applying 8-bit Symmetric Quantization... (Simulated)")
    quant_model_sym = copy.deepcopy(model)
    quant_model_sym.to(DEVICE)

    for name, module in quant_model_sym.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            logger.debug(f"Quantizing layer (symmetric): {name}")
            float_weights = module.weight.data
            dequant_weights,_, _, _ = Quantization.quantize_tensor(
                float_weights, method='symmetric', num_bits=8)
            module.weight.data = dequant_weights.to(module.weight.dtype)

    acc_symm, time_symm = evaluate(quant_model_sym, test_loader, "Simulated Symmetric 8-bit")

    # Actual Storage Quantization Experiment

    logger.info("Starting Actual Storage Quantization (Int8)..")

    storage_model = CNN().to('cpu')
    storage_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))

    storage_model = replace_layers_with_quantizable(storage_model)
    logger.info("Model layers replaced with Quantizable versions")

    for name, module in storage_model.named_modules():
        if hasattr(module, 'quantized_storage'):
            module.quantized_storage(num_bits=8, method='affine')

    logger.info("Calibrating activations for quantized model (using train set subset)...")
    calibrated_model_activation(storage_model, train_loader, device='cpu', num_batches=50, method='affine', num_bits=8)

    logger.info(f"Saving quantized model to: {QUANTIZED_SAVE_PATH}")
    torch.save(storage_model.state_dict(), QUANTIZED_SAVE_PATH)

    original_size = os.path.getsize(MODEL_SAVE_PATH) / 1024
    quantized_size = os.path.getsize(QUANTIZED_SAVE_PATH) / 1024
    reduction = 100 * (1 - quantized_size / original_size) if original_size > 0 else 0.0

    logger.info(f"Original Size: {original_size:.2f} KB")
    logger.info(f"Quantized Size: {quantized_size:.2f} KB")
    logger.info(f"Reduction: {reduction:.2f}%")

    storage_model.to(DEVICE)
    acc_storage, time_storage = evaluate(storage_model, test_loader, "Storage Quantization (Int8)")

    logger.info("Computing per-layer weight MSE between float model and quantized storage model...")
    errors = layer_weight_mse(model, storage_model)

    # save errors to CSV
    with open(MSE_CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['layer', 'mse'])
        for k, v in errors.items():
            writer.writerow([k, v])

    logger.info("Running per-layer sensitivity analysis (quantize one layer at a time)")
    sensitivity_results = per_layer_sensitivity_analysis(model, test_loader, DEVICE, bits=8, method='affine')

    # save sensitivity to CSV
    with open(SENSITIVITY_CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['layer', 'accuracy', 'inference_time'])
        writer.writeheader()
        for entry in sensitivity_results:
            writer.writerow(entry)



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
# DONE UNTIL 22th of Dec:
# Better Evaluation
# Visualisation: 
# Jupyter Notebooks for results 
# learning curve during training with example results of accuracy
# -----------------------

# DURING CHRISTMAS:
# Bigger Dataset for Training during christmas
# --> Helper Function for dataset integration
# --> making it actually work lol


