# Investigating of Quantization Strategies for Deep Learning Models (RESEARCH PROJECT FOR MASTERS DEGREE AT HTW BERLIN)

## Project Overview

This project evaluates the impact of 8-bit Post-Training Quantization (PTQ) strategies on a Convolutional Neural Network (CNN). It explores the trade-offs between model compression, inference speed, and predictive accuracy across datasets of varying complexity.

### Key Research Questions

* **Methodological Impact**: How do affine, symmetric, and Power-of-Two (PoT) quantization schemes affect accuracy and latency across different data distributions? 

* **Layer Sensitivity**: In which specific network layers do quantization-induced performance losses primarily concentrate? 



## Methodologies & Implementation

* **Quantization Schemes**: Implementation of Affine (asymmetric), Symmetric, and simulated Power-of-Two (PoT) 8-bit quantization.

* **Architecture**: A 4-block CNN utilizing Batch Normalization to center activations, to make efficient symmetric and PoT quantization.

* **Fidelity Metrics**: Layer-wise analysis using Signal-to-Quantization-Noise-Ratio (SQNR), Mean Squared Error (MSE), and Kullback-Leibler (KL) Divergence.

* **Framework**: Built with PyTorch 2.9.1 (torch.ao) and benchmarked on x86 architecture (fbgemm backend).



## Key Results

### Global Performance Comparison (CIFAR-10)

* **Storage**: Consistently reduced model size by ~74.5% (from 5.53 MB to ~1.4 MB).
* **Latency**: Improved inference speed by ~75.9% on x86 hardware.
* **Accuracy**: Affine quantization proved most robust, maintaining accuracy within 0.05% of the FP32 baseline ($84.48\%$ vs. $84.43\%$).


### Layer-wise Fidelity Analysis

* **Error Localization**: Signal degradation is minimal in early convolutional layers but concentrates heavily in deep Fully-Connected (FC) layers.

* **Fidelity Collapse**: In sensitive datasets (e.g., Pokémon), SQNR for symmetric/PoT methods can drop to near 0 dB in final layers, leading to significant accuracy loss.



## Project Structure

```bash
├── main.py                             # Entry point for training and experiments
├── src
│   ├── analysis
│   │   └── layer_analysis.py           # Metrics for MSE, SQNR, and KL-Divergence
│   ├── evaluation
│   │   └── evaluate.py                 # Inference loops and latency benchmarking
│   ├── fake_quantization
│   │   └── fake_quant_config.py        # Simulation settings for bit-depth noise
│   ├── model_cnn
│   │   ├── model.py                    # CNN Architecture (Conv-BN-ReLU blocks)
│   │   └── train.py                    # Training loop implementation
│   ├── torch_quantization
│   │   ├── custom_observer.py          # Range observers for calibration
│   │   ├── quant_utils.py              # Helper functions for PTQ conversion
│   │   └── quantization_calibration.py # Calibration logic for scale/zero-point
│   └── utility
│       ├── config.py                   # Global dataset and method settings
│       └── utils.py                    # Data loading (CIFAR, Pokemon) and logging
├── results
│   ├── csv                             # Layer-wise fidelity data
│   ├── logs                            # Experiment logs
│   ├── models                          # Saved FP32 weights
│   ├── quantized_models                # Saved INT8 and Fake-Quant weights
│   └── tests                           # Results organized by dataset (CIFAR, Pokemon, etc.)
├── notebooks                           # Jupyter notebooks for prototyping
├── data                                # Raw dataset storage
└── pyproject.toml                      # Project dependencies (uv)
```


## Usage

### 1. Installation

Ensure Python 3.12.0+ is installed. Use `uv` to install dependencies:

```bash
uv sync
```

### 2. Execution

Run the main script to train the baseline and execute the PTQ pipeline:

```bash
uv run src/main.py
```

## References

* Krishnamoorthi, R. "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper." (2018). 


* Nagel, M., et al. "A White Paper on Neural Network Quantization." (2021). 


* Gholami, A., et al. "A Survey of Quantization Methods for Efficient Neural Network Inference." (2021).