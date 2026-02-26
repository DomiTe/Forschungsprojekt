TODO: REPORT WRITING

# Investigating Quantization Strategies for Deep Learning Models

This repository contains the implementation and experimental results for the research paper:

**"Investigating Quantization Strategies for Deep Learning Models: Schichtweise Fidelity-Analyse und Hardware-Simulation: Ein Vergleich von Affiner, Symmetrischer und PoT-Quantisierung"**

## Project Overview

This project evaluates the trade-offs between different **Post-Training Quantization (PTQ)** strategies on x86 hardware. It specifically investigates the "Fidelity-Loss" across network layers, analyzing how **Symmetric** and **Power-of-Two (PoT)** schemes compare against the **Affine** baseline when dealing with unnormalized data distributions.

### Key Research Questions

1. **RQ1:** How does the choice of quantization (Affine, Symmetric, Power-of-Two) affect the accuracy and latency of a CNN?
2. **RQ2:** In which network layers are performance losses concentrated, and how does layer sensitivity differ between robust (CIFAR) and sensitive datasets?

## Methodologies & Implementation

The project utilizes the **PyTorch 2.2 `fbgemm` backend** to simulate and execute quantization on x86 CPUs.

* **Affine (Asymmetric):** Mapping the range  to  using a floating-point Scale () and Integer Zero-Point ().
* **Symmetric:** Fixing  to optimize hardware execution, though potentially leading to "Bit-Waste" in asymmetric distributions.
* **Power-of-Two (PoT):** Restricting scaling factors to  to replace multiplications with bit-shifts.
* **Fidelity Tracking:** Use of **Forward Hooks** to capture inter-layer tensors and calculate MSE, SQNR, and KL-Divergence.

## Key Results

### Global Performance Comparison (CIFAR-10)

The following table summarizes the performance of a 4-block CNN architecture. Note that Affine quantization effectively maintained baseline accuracy by avoiding signal degradation in the input layer.

| Config | Method | Acc (%) | F1-Score | Time (s) | Drop (%) |
| --- | --- | --- | --- | --- | --- |
| **Baseline** | Float32 | 85.23 | 0.853 | 52.65 | 0.0 |
| **Affine_PTQ** | Affine | **85.44** | 0.855 | 14.35 | -0.21 |
| **Symmetric_PTQ** | Symmetric | 84.85 | 0.849 | 12.55 | 0.38 |
| **PoT_PTQ** | PoT | 83.82 | 0.839 | 12.55 | 1.41 |

### Layer-wise Fidelity Analysis

The analysis highlights the **"Input Shock"**—a massive drop in signal quality occurring in the first layer when using symmetric schemes on strictly positive data (e.g., after `QuantStub`).

| Layer | Method | MSE | SQNR (dB) | KL-Div |
| --- | --- | --- | --- | --- |
| **quant (Input)** | Affine |  | **114.6** | 0.0 |
| **quant (Input)** | Symmetric |  | **47.8** | 26.2 |
| **quant (Input)** | PoT |  | 47.4 | 29.0 |

> **Observation:** Symmetric quantization effectively uses only 7-bit resolution for unnormalized  data, explaining the superiority of Affine methods for the input stage.

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

```bash
uv sync
```

### 2. Execution

Ensure the dataset and method are set in `config.py`, then run:

```bash
uv run main.py
```

The script will perform a three-phase execution:

1. **Train/Load** FP32 Baseline.
2. **Calibrate** using the validation set to determine  and .
3. **Convert** to INT8 and evaluate layer-wise metrics.

## References

* **Nagel et al. (2021):** *A White Paper on Neural Network Quantization*.
* **Wu et al. (2020):** *Integer Quantization for Deep Learning Inference*.
* **Lee et al. (2022):** *Quantune: Post-training quantization with hardware-aware tuning*.

