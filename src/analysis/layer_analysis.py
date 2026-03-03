import torch
import torch.nn as nn
import numpy as np
import copy
import logging
import collections
import statistics

from src.evaluation.evaluate import evaluate
from src.torch_quantization.quantization_calibration import calibrate_model
from src.utility.utils import save_csv

logger = logging.getLogger("Experiment")

class LayerAnalyzer:
    def __init__(self, float_model, loader, device, qconfig):
        self.float_model = float_model
        self.loader = loader
        self.device = device
        self.qconfig = qconfig

    def compute_fidelity_metrics(self, x_float, x_quant):
        """
        Computes layer-wise fidelity metrics: MSE, SQNR, KL Divergence.
        """
        # Flatten and convert to numpy
        x_f = x_float.detach().cpu().flatten().numpy()
        x_q = x_quant.detach().cpu().flatten().numpy()
        
        # 1. MSE (Mean Squared Error)
        mse = np.mean((x_f - x_q) ** 2)
        
        # 2. SQNR (Signal to Quantization Noise Ratio) in dB
        signal_power = np.mean(x_f ** 2)
        noise_power = mse + 1e-12
        sqnr = 10 * np.log10(signal_power / noise_power)
        
        # 3. KL Divergence
        min_val = min(x_f.min(), x_q.min())
        max_val = max(x_f.max(), x_q.max())
        
        # 1024 bins for histogram
        bins = np.linspace(min_val, max_val, 1024)
        hist_f, _ = np.histogram(x_f, bins=bins, density=True)
        hist_q, _ = np.histogram(x_q, bins=bins, density=True)
        
        # Normalize to probability distributions
        hist_f += 1e-12; hist_f /= hist_f.sum()
        hist_q += 1e-12; hist_q /= hist_q.sum()
        
        # KL(P || Q)
        kl = np.sum(hist_f * np.log(hist_f / hist_q))
        
        return mse, sqnr, kl

    def run_layer_wise_analysis(self, output_csv="layer_analysis.csv"):
        """
        Runs Fidelity (Comparison) and Sensitivity (Ablation) analysis.
        """
        logger.info("--- Starting Comprehensive Layer-wise Analysis ---")
        
        # 1. Setup Fake Quantized Model (for Fidelity comparison)
        fake_quant_model = copy.deepcopy(self.float_model)
        fake_quant_model.qconfig = self.qconfig
        torch.ao.quantization.prepare(fake_quant_model, inplace=True)
        # Calibrate Fake Quant Model
        calibrate_model(fake_quant_model, self.loader, num_batches=5, device=self.device)
        
        # 2. Measure Baseline Global Metrics
        base_metrics = evaluate(self.float_model, self.loader, "Baseline", self.device)
        
        # 3. Identify Quantizable Layers
        layers_to_analyze = []
        for name, module in fake_quant_model.named_modules():
            # In fake quant, layers often have an 'activation_post_process' submodule
            if hasattr(module, 'activation_post_process'): 
                if not any(x in name for x in ["activation_post_process", "weight_fake_quant"]):
                    layers_to_analyze.append(name)
        
        logger.info(f"Analyzing {len(layers_to_analyze)} layers: {layers_to_analyze}")
        
        results = []
        
        # --- A. FIDELITY: Capture feature maps ---
        activations_float = {}
        activations_quant = {}
        
        def get_hook(store_dict, name):
            def hook(model, input, output):
                store_dict[name] = output
            return hook
            
        hooks = []
        # Hook Float Model
        for name, module in self.float_model.named_modules():
            if name in layers_to_analyze:
                hooks.append(module.register_forward_hook(get_hook(activations_float, name)))
        # Hook Fake Quant Model
        for name, module in fake_quant_model.named_modules():
            if name in layers_to_analyze:
                hooks.append(module.register_forward_hook(get_hook(activations_quant, name)))
        
        # Run ONE batch for fidelity stats
        inputs, _ = next(iter(self.loader))
        inputs = inputs.to(self.device)
        self.float_model(inputs)
        fake_quant_model(inputs)
        
        for h in hooks: h.remove()
        
        # --- B. SENSITIVITY Loop ---
        for name in layers_to_analyze:
            
            if "activation_post_process" in name or "weight_fake_quant" in name:
                continue
            
            logger.info(f"Analyzing Layer: {name}...")
            
            # 1. Fidelity Metrics
            mse, sqnr, kl = 0, 0, 0
            if name in activations_float and name in activations_quant:
                mse, sqnr, kl = self.compute_fidelity_metrics(
                    activations_float[name], activations_quant[name]
                )
            
            # 2. Sensitivity (Quantize ONLY this layer)
            temp_model = copy.deepcopy(self.float_model)
            temp_model.qconfig = None # Disable global
            
            # Apply config ONLY to target
            try:
                target_module = dict(temp_model.named_modules())[name]
                target_module.qconfig = self.qconfig
            except KeyError:
                logger.warning(f"Could not find layer {name} in float model. Skipping.")
                continue
            
            # Prepare & Calibrate
            torch.ao.quantization.prepare(temp_model, inplace=True)
            calibrate_model(temp_model, self.loader, num_batches=2, device=self.device)
            
            # Evaluate Global Impact
            metrics = evaluate(temp_model, self.loader, f"Sens: {name}", self.device)
            
            row = {
                "layer_name": name,
                "avg_mse": mse,
                "sqnr_db": sqnr,
                "kl_divergence": kl,
                "acc_drop": base_metrics['accuracy'] - metrics['accuracy'],
                "f1_drop": base_metrics['f1_score'] - metrics['f1_score'],
                "precision_drop": base_metrics['precision'] - metrics['precision'],
                "recall_drop": base_metrics['recall'] - metrics['recall'],
            }
            results.append(row)
            
        save_csv(results, output_csv, results[0].keys())
        return results

    def run_real_quant_analysis(self, quantized_model, output_csv="real_layer_analysis.csv"):
        """
        Analyzes fidelity for models converted with torch.ao.quantization.convert.
        """
        logger.info("--- Starting Real Quantized Layer-wise Analysis over full test set ---")
        
        activations_float = {}
        activations_real = {}

        def get_hook(store_dict, name, is_quantized=False):
            def hook(model, input, output):
                # dequantize to compare
                if is_quantized and hasattr(output, 'dequantize'):
                    store_dict[name] = output.dequantize().detach().cpu()
                else:
                    store_dict[name] = output.detach().cpu()
            return hook

        layers_to_hook = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']
        hooks = []

        # hook float model
        for name, module in self.float_model.named_modules():
            if name in layers_to_hook:
                hooks.append(module.register_forward_hook(get_hook(activations_float, name)))
                
        # hook quantized model
        for name, module in quantized_model.named_modules():
            if name in layers_to_hook:
                hooks.append(module.register_forward_hook(get_hook(activations_real, name, is_quantized=True)))

        # store metrics for all batches
        batch_metrics = collections.defaultdict(lambda: {"mse": [], "sqnr": [], "kl": []})

        # iterate over entire dataloader
        with torch.no_grad():
            for inputs, _ in self.loader:
                inputs = inputs.to(self.device)
                self.float_model(inputs)
                quantized_model(inputs)

                # calculate and store metrics for current batch
                for name in layers_to_hook:
                    if name in activations_float and name in activations_real:
                        mse, sqnr, kl = self.compute_fidelity_metrics(
                            activations_float[name], activations_real[name]
                        )
                        batch_metrics[name]["mse"].append(mse)
                        batch_metrics[name]["sqnr"].append(sqnr)
                        batch_metrics[name]["kl"].append(kl)

        # calculate final stats across all batches
        results = []
        for name in layers_to_hook:
            if name in batch_metrics:
                mse_list = batch_metrics[name]["mse"]
                sqnr_list = batch_metrics[name]["sqnr"]
                kl_list = batch_metrics[name]["kl"]
                
                # compute stats
                results.append({
                    "layer_name": name,
                    "avg_mse": statistics.mean(mse_list),
                    "batch_mse_std": statistics.stdev(mse_list) if len(mse_list) > 1 else 0.0,
                    "sqnr_db": statistics.mean(sqnr_list),
                    "batch_sqnr_std": statistics.stdev(sqnr_list) if len(sqnr_list) > 1 else 0.0,
                    "kl_divergence": statistics.mean(kl_list),
                    "batch_kl_std": statistics.stdev(kl_list) if len(kl_list) > 1 else 0.0
                })

        for h in hooks: h.remove()
        save_csv(results, output_csv, list(results[0].keys()))
        return results