import torch
import copy
import os
import logging
import warnings

from torch.ao.quantization import (
    QConfig, 
    HistogramObserver, 
    PerChannelMinMaxObserver, 
    fuse_modules,
    MinMaxObserver, 
)

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
from src.utility.config import (
    DEVICE, 
    QUANTIZED_MODELS, 
    BASELINE_MODEL_PATH,
    EXPERIMENT_CSV_PATH
)

logger = logging.getLogger("Experiment")

class PowerOfTwoObserver(MinMaxObserver):
    """
    Custom Observer for Activations.
    Forces scale to be a power of 2 (2^k) and zero_point to be 0.
    """
    def calculate_qparams(self):
        # 1. Get standard min/max scale
        scale, zero_point = super().calculate_qparams()
        
        # 2. Force Scale to nearest Power of Two: 2^(round(log2(scale)))
        scale = 2.0 ** torch.round(torch.log2(scale))
        
        # 3. Force Zero Point to 0 (PoT is strictly symmetric)
        zero_point = torch.zeros_like(zero_point)
        
        return scale, zero_point

class PowerOfTwoWeightObserver(PerChannelMinMaxObserver):
    """
    Custom Observer for Weights (Per-Channel).
    Forces scale to be a power of 2 (2^k) and zero_point to be 0.
    """
    def calculate_qparams(self):
        # 1. Get standard per-channel scale
        scale, zero_point = super().calculate_qparams()
        
        # 2. Force Scale to nearest Power of Two
        scale = 2.0 ** torch.round(torch.log2(scale))
        
        # 3. Force Zero Point to 0
        zero_point = torch.zeros_like(zero_point)
        
        return scale, zero_point
    

def fuse_layers(model):
    """
    Fuses Conv+ReLU and Linear+ReLU layers to improve accuracy and speed.
    Ref: https://pytorch.org/docs/stable/quantization.html#module-torch.ao.quantization.fuse_modules
    """
    # Ensure model is in eval mode before fusion
    model.eval()
    
    # NOTE: Adjust these names to match your src/model.py definition exactly.
    # Example assumes: self.conv1, self.relu1, self.fc1, etc.
    fusion_candidates = [
        ['conv1', 'relu1'],
        ['conv2', 'relu2'],
        ['conv3', 'relu3'],
        ['conv4', 'relu4'],
        ['fc1', 'relu5'] # Assuming fc1 is followed by relu5
    ]
    
    # Filter out modules that don't exist in the model to prevent errors
    existing_fusions = [
        f for f in fusion_candidates 
        if hasattr(model, f[0]) and hasattr(model, f[1])
    ]
    
    if existing_fusions:
        fuse_modules(model, existing_fusions, inplace=True)
        logger.info(f"Fused layers: {existing_fusions}")
    else:
        logger.warning("No layers fused. Check layer names in 'fuse_layers' function.")

    return model
def get_pot_qconfig():
    """
    Returns QConfig for Power of Two (PoT) Quantization.
    Uses custom observers to enforce 2^k scaling.
    """
    return QConfig(
        # ACTIVATIONS:
        # Use our custom PoT class.
        # qscheme=symmetric is required because PoT implies ZP=0.
        activation=PowerOfTwoObserver.with_args(
            qscheme=torch.per_tensor_symmetric,
            dtype=torch.quint8
        ),
        
        # WEIGHTS:
        # Use our custom Per-Channel PoT class.
        # reduce_range=True is still needed for FBGEMM backend safety.
        weight=PowerOfTwoWeightObserver.with_args(
            qscheme=torch.per_channel_symmetric,
            dtype=torch.qint8,
            reduce_range=True
        )
    )

def get_symmetric_qconfig():
    """
    Returns QConfig for Symmetric Quantization (Manual/Eager Mode).
    """
    return QConfig(
        # ACTIVATIONS:
        # qint8 is signed [-128, 127]. 
        # Symmetric means the range is centered at 0 (zero_point=0).
        activation=HistogramObserver.with_args(
            qscheme=torch.per_tensor_symmetric,
            dtype=torch.quint8
        ),
        
        # WEIGHTS:
        # reduce_range=True is MANDATORY for standard x86 CPUs (FBGEMM backend).
        # It restricts weights to 7-bit to prevent overflow.
        weight=PerChannelMinMaxObserver.with_args(
            qscheme=torch.per_channel_symmetric,
            dtype=torch.qint8,
            reduce_range=True 
        )
    )

def get_affine_qconfig():
    """
    Returns QConfig for Affine (Asymmetric) Quantization.
    Activations: HistogramObserver (reduces outlier impact).
    Weights: PerChannelMinMaxObserver (standard for CNNs).
    Ref: https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig.QConfig.html
    """
    return QConfig(
        activation=HistogramObserver.with_args(
            qscheme=torch.per_tensor_affine, 
            dtype=torch.quint8
        ),
        weight=PerChannelMinMaxObserver.with_args(
            qscheme=torch.per_channel_affine, 
            dtype=torch.qint8,
            reduce_range=True
        )
    )

def run_experiment():
    # Filter out the specific deprecation warnings from PyTorch Quantization
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao.quantization")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.ao.quantization")
    # 1. Load Data
    logger.info("Loading datasets...")
    train_loader, test_loader, num_classes = get_data_loaders()
    torch.backends.quantized.engine = 'fbgemm' # or fbgemm, QNnpack

    # 2. Baseline Model Management
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
    acc_base, time_base = evaluate(model, test_loader, "Baseline Float32",device=torch.device('cpu'))
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

    # 4. Affine Quantization Experiment
    experiment_name = "Affine_PTQ"
    logger.info(f"--- Starting Experiment: {experiment_name} ---")

    # Step A: Create a clean copy for quantization
    # We use 'cpu' for quantization preparation as some observers are CPU-only
    model_affine = copy.deepcopy(model).to('cpu')
    model_affine.eval()

    # Step B: Fuse Layers
    # Fusing must happen BEFORE attaching observers
    model_affine = fuse_layers(model_affine)

    # Step C: Attach Configuration (Affine)
    model_affine.qconfig = get_affine_qconfig()
    
    # Step D: Prepare
    # Inserts observers into the model layers
    # Ref: https://pytorch.org/docs/stable/generated/torch.ao.quantization.prepare.prepare.html
    torch.ao.quantization.prepare(model_affine, inplace=True)
    
    # Step E: Calibrate
    # Passes data through model to calculate min/max ranges
    logger.info("Calibrating model...")
    calibrate_model(model_affine, train_loader, num_batches=10, device='cpu')

    # Step F: Convert
    # Freezes statistics and swaps float layers for int8 layers
    # Ref: https://pytorch.org/docs/stable/generated/torch.ao.quantization.convert.convert.html
    logger.info("Converting to INT8...")
    torch.ao.quantization.convert(model_affine, inplace=True)

    # 5. Evaluate Quantized Model
    acc, time_inf = evaluate(model_affine, test_loader, experiment_name, device=torch.device('cpu'))
    drop = acc_base - acc
    
    logger.info(f"Result {experiment_name}: Acc={acc:.2f}% (Drop: {drop:.2f})")

    # 6. Save Model
    save_path = os.path.join(QUANTIZED_MODELS, f"model_{experiment_name}.pt")
    # TorchScript is preferred for saving quantized models to preserve structure
    torch.jit.save(torch.jit.script(model_affine), save_path)
    
    real_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    logger.info(f"Saved model size: {real_size_mb:.2f} MB")
    
    results.append({
        "config_name": experiment_name,
        "method": "affine",
        "bits": 8,
        "accuracy": acc,
        "inference_time": time_inf,
        "model_size_mb": real_size_mb,
        "drop_percentage": drop
    })

    # 7. Save Results
    save_csv(results, EXPERIMENT_CSV_PATH, 
             ['config_name', 'method', 'bits', 'accuracy', 'inference_time', 'model_size_mb', 'drop_percentage'])

    # 5. Symmetric Quantization (Manual/Eager Mode)
    experiment_name = "Symmetric_PTQ"
    logger.info(f"--- Starting Experiment: {experiment_name} ---")

    # Step A: Clean Copy & CPU move
    # Quantization preparation typically happens on CPU
    model_sym = copy.deepcopy(model).to("cpu")
    model_sym.eval()

    # Step B: Manual Fusion
    # Crucial: Fusion must happen BEFORE setting the qconfig in Eager mode
    model_sym = fuse_layers(model_sym)

    # Step C: Attach Configuration
    # We assign the config directly to the model
    model_sym.qconfig = get_symmetric_qconfig()

    # Step D: Prepare
    # This inserts the Observers into the model layers
    # Warning: "prepare" is deprecated but correct for this manual workflow
    torch.ao.quantization.prepare(model_sym, inplace=True)

    # Step E: Calibrate
    logger.info("Calibrating Symmetric model...")
    calibrate_model(model_sym, train_loader, num_batches=10, device=torch.device('cpu'))

    # Step F: Convert
    # Freezes statistics and swaps float layers for int8 layers
    logger.info("Converting to INT8 (Symmetric)...")
    torch.ao.quantization.convert(model_sym, inplace=True)

    # Step G: Evaluate
    acc, time_inf = evaluate(model_sym, test_loader, experiment_name, device=torch.device('cpu'))
    drop = acc_base - acc
    
    logger.info(f"Result {experiment_name}: Acc={acc:.2f}% (Drop: {drop:.2f})")

    # Step H: Save
    save_path = os.path.join(QUANTIZED_MODELS, f"model_{experiment_name}.pt")
    # Use TorchScript to save the quantized model structure reliably
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

    # 6. Power of Two (PoT) Quantization (Manual/Eager)
    experiment_name = "PoT_PTQ"
    logger.info(f"--- Starting Experiment: {experiment_name} ---")

    # Step A: Clean Copy
    model_pot = copy.deepcopy(model).to("cpu")
    model_pot.eval()

    # Step B: Manual Fusion
    model_pot = fuse_layers(model_pot)

    # Step C: Attach PoT Config
    model_pot.qconfig = get_pot_qconfig()

    # Step D: Prepare
    # Uses our custom Observers to record min/max
    torch.ao.quantization.prepare(model_pot, inplace=True)

    # Step E: Calibrate
    logger.info("Calibrating PoT model...")
    calibrate_model(model_pot, train_loader, num_batches=10, device="cpu")

    # Step F: Convert
    # This calls our custom calculate_qparams(), rounding scales to 2^k
    logger.info("Converting to INT8 (PoT)...")
    torch.ao.quantization.convert(model_pot, inplace=True)

    # Step G: Evaluate
    acc, time_inf = evaluate(model_pot, test_loader, experiment_name, device=torch.device('cpu'))
    drop = acc_base - acc
    
    logger.info(f"Result {experiment_name}: Acc={acc:.2f}% (Drop: {drop:.2f})")

    # Step H: Save
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