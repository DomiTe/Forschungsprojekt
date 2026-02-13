import torch
import copy
import os
import logging
import warnings

# Project Imports
from src.model import CNN
from src.train import train_model
from src.evaluation.evaluate import evaluate

from src.utility.quantization_calibration import calibrate_model
from src.utility.utils import (
    get_data_loaders, 
    get_model_size, 
    save_csv, 
    setup_global_logging, 
    plot_training_curves
)
from src.utility.quant_utils import (
    fuse_layers,
    get_custome_affine_qconfig,
    get_custome_symmetric_qconfig,
    get_custome_pot_qconfig
)

from src.utility.config import (
    DEVICE, 
    QUANTIZED_MODELS, 
    BASELINE_MODEL_PATH,
    EXPERIMENT_CSV_PATH
)

logger = logging.getLogger("Experiment")


def run_experiment():
    # Suppress deprecation warnings for cleaner logs
    warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
    
    # 1. Setup Data & Backend
    logger.info("Loading datasets...")
    train_loader, test_loader, num_classes = get_data_loaders()
    torch.backends.quantized.engine = 'fbgemm' # Optimized for x86 CPUs

    # 2. Baseline Model (Float32)
    # Load existing model or train from scratch if missing
    baseline_path = BASELINE_MODEL_PATH
    model = CNN(num_classes=num_classes).to(DEVICE)

    if os.path.exists(baseline_path):
        logger.info(f"Load Baseline-Model: {baseline_path}")
        model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
    else:
        logger.info("No baseline found. Starting training...")
        model, history = train_model(train_loader, test_loader, num_classes)
        plot_training_curves(history)

    # 3. Evaluate Baseline
    logger.info("Evaluating Baseline (Float32)...")
    model.eval()
    acc_base, time_base = evaluate(model, test_loader, "Baseline Float32", device=torch.device('cpu'))
    size_base = get_model_size(model)
    logger.info(f"Baseline -> Acc: {acc_base:.2f}%, Time: {time_base:.4f}s, Size: {size_base:.2f}MB")

    results = [{
        "config_name": "Baseline (Float32)",
        "method": "float",
        "bits": 32,
        "accuracy": acc_base,
        "inference_time": time_base,
        "model_size_mb": size_base,
        "drop_percentage": 0.0
    }]

    # ---------------------------------------------------------
    # Experiment 1: Affine Quantization (Asymmetric)
    # ---------------------------------------------------------
    experiment_name = "Affine_PTQ"
    logger.info(f"--- Starting Experiment: {experiment_name} ---")

    # A. Preparation
    model_affine = copy.deepcopy(model).to('cpu')
    model_affine.eval()
    model_affine = fuse_layers(model_affine) # Fuse BEFORE config

    # B. Configuration & Observer Attachment
    model_affine.qconfig = get_custome_affine_qconfig()
    
    if hasattr(model_affine, 'conv1'):
        model_affine.conv1.qconfig = None
        model_affine.relu1.qconfig = None
        # model_sym.quant.qconfig = None

    # Letzte Schicht (fc2)
    if hasattr(model_affine, 'fc2'):
        model_affine.fc2.qconfig = None
        model_affine.relu2.qconfig = None 
    torch.ao.quantization.prepare(model_affine, inplace=True)
    
    # C. Calibration (Find Min/Max)
    logger.info("Calibrating model...")
    calibrate_model(model_affine, train_loader, num_batches=10, device='cpu')

    # D. Conversion (Float -> Int8)
    logger.info("Converting to INT8...")
    torch.ao.quantization.convert(model_affine, inplace=True)

    # E. Evaluation
    acc, time_inf = evaluate(model_affine, test_loader, experiment_name, device=torch.device('cpu'))
    drop = acc_base - acc
    
    logger.info(f"Result {experiment_name}: Acc={acc:.2f}% (Drop: {drop:.2f})")

    # F. Save Model
    save_path = os.path.join(QUANTIZED_MODELS, f"model_{experiment_name}.pt")
    torch.jit.save(torch.jit.script(model_affine), save_path)
    real_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    
    results.append({
        "config_name": experiment_name,
        "method": "affine",
        "bits": 8,
        "accuracy": acc,
        "inference_time": time_inf,
        "model_size_mb": real_size_mb,
        "drop_percentage": drop
    })

    save_csv(results, EXPERIMENT_CSV_PATH, list(results[0].keys()))

    # ---------------------------------------------------------
    # Experiment 2: Symmetric Quantization
    # ---------------------------------------------------------
    experiment_name = "Symmetric_PTQ"
    logger.info(f"--- Starting Experiment: {experiment_name} ---")

    # A. Preparation
    model_sym = copy.deepcopy(model).to("cpu")
    model_sym.eval()
    model_sym = fuse_layers(model_sym)

    # B. Configuration
    model_sym.qconfig = get_custome_symmetric_qconfig()
    
    if hasattr(model_sym, 'conv1'):
        model_sym.conv1.qconfig = None
        model_sym.relu1.qconfig = None
        # model_sym.quant.qconfig = None

    # Letzte Schicht (fc2)
    if hasattr(model_sym, 'fc2'):
        model_sym.fc2.qconfig = None
        model_sym.relu2.qconfig = None 
        
    torch.ao.quantization.prepare(model_sym, inplace=True)

    # C. Calibration
    logger.info("Calibrating Symmetric model...")
    calibrate_model(model_sym, train_loader, num_batches=10, device=torch.device('cpu'))

    # D. Conversion
    logger.info("Converting to INT8 (Symmetric)...")
    torch.ao.quantization.convert(model_sym, inplace=True)

    # E. Evaluation
    acc, time_inf = evaluate(model_sym, test_loader, experiment_name, device=torch.device('cpu'))
    drop = acc_base - acc
    
    logger.info(f"Result {experiment_name}: Acc={acc:.2f}% (Drop: {drop:.2f})")

    # F. Save
    save_path = os.path.join(QUANTIZED_MODELS, f"model_{experiment_name}.pt")
    torch.jit.save(torch.jit.script(model_sym), save_path)
    real_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    results.append({
        "config_name": experiment_name,
        "method": "symmetric",
        "bits": 8,
        "accuracy": acc,
        "inference_time": time_inf,
        "model_size_mb": real_size_mb,
        "drop_percentage": drop
    })

    save_csv(results, EXPERIMENT_CSV_PATH, list(results[0].keys()))

    # ---------------------------------------------------------
    # Experiment 3: Power-of-Two (PoT) Quantization
    # ---------------------------------------------------------
    experiment_name = "PoT_PTQ"
    logger.info(f"--- Starting Experiment: {experiment_name} ---")

    # A. Preparation
    model_pot = copy.deepcopy(model).to("cpu")
    model_pot.eval()
    model_pot = fuse_layers(model_pot)

    # B. Configuration
    model_pot.qconfig = get_custome_pot_qconfig()
    
    if hasattr(model_pot, 'conv1'):
        model_pot.conv1.qconfig = None
        model_pot.relu1.qconfig = None 
        # model_sym.quant.qconfig = None

    # Letzte Schicht (fc2)
    if hasattr(model_pot, 'fc2'):
        model_pot.fc2.qconfig = None
        model_pot.relu2.qconfig = None 
        
    torch.ao.quantization.prepare(model_pot, inplace=True)

    # C. Calibration
    logger.info("Calibrating PoT model...")
    calibrate_model(model_pot, train_loader, num_batches=10, device="cpu")

    # D. Conversion (Calculates 2^k scales)
    logger.info("Converting to INT8 (PoT)...")
    torch.ao.quantization.convert(model_pot, inplace=True)

    # E. Evaluation
    acc, time_inf = evaluate(model_pot, test_loader, experiment_name, device=torch.device('cpu'))
    drop = acc_base - acc
    
    logger.info(f"Result {experiment_name}: Acc={acc:.2f}% (Drop: {drop:.2f})")

    # F. Save
    save_path = os.path.join(QUANTIZED_MODELS, f"model_{experiment_name}.pt")
    torch.jit.save(torch.jit.script(model_pot), save_path)
    real_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    results.append({
        "config_name": experiment_name,
        "method": "pot_manual",
        "bits": 8,
        "accuracy": acc,
        "inference_time": time_inf,
        "model_size_mb": real_size_mb,
        "drop_percentage": drop
    })

    save_csv(results, EXPERIMENT_CSV_PATH, list(results[0].keys()))

    logger.info("=== Experiments completed successfully ===")

if __name__ == "__main__":
    setup_global_logging()
    run_experiment()