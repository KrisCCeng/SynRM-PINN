# SynRM Flux Linkage Identification via PINN and PCGrad

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for the paper:  
**"High-Precision Standstill Flux Linkage Identification for Synchronous Reluctance Motors Based on Gradient Projection and Implicit Inductance Modeling"**

## 💡 Overview

Accurate flux linkage modeling is essential for the high-performance control of Synchronous Reluctance Motors (SynRM) due to their severe magnetic saturation and cross-coupling effects. However, traditional data-driven methods struggle with physical consistency, and conventional Physics-Informed Neural Networks (PINNs) easily fall into multi-task gradient conflicts when facing measurement noise.

This project proposes a novel high-precision standstill identification framework featuring:
1. **Implicit Secant Inductance Architecture (SCN):** Strictly satisfies the physical zero-point constraint and all-plane parity symmetry from the mathematical foundation.
2. **Gradient Surgery (PCGrad):** Eliminates gradient conflicts between data-fitting and Maxwell's reciprocity physical constraints without relying on noise-sensitive PDE loss.
3. **Robust Evaluation:** Outperforms traditional analytical formulas and baseline ANNs even under severe integral drift and rotor vibration noise.

## 📂 Repository Structure

    SynRM_PINN_Project/
    ├── configs/                   # Configuration files (hyperparameters, paths)
    │   └── pinn_train.yaml        # PINN training and evaluation config
    ├── data/                      # Dataset directory
    │   ├── processed/             # Experimental data (e.g., hysteresis voltage injection)
    │   └── simulation/            # Simulation data (2000/4000/8000 ideal points)
    ├── outputs/                   # Auto-generated results
    │   └── [Dataset_Name]/
    │       ├── models/            # Saved .pth model weights
    │       ├── logs/              # Training logs (.csv)
    │       └── evaluation_results/# Generated 3D flux surfaces, error heatmaps, loss curves
    ├── src/                       # Core source code
    │   ├── dataset.py             # Data loading and preprocessing
    │   ├── models.py              # Neural network definitions (ANN, SCN, PINN)
    │   ├── baselines.py           # Traditional methods (Polynomial, Look-Up Table)
    │   ├── train_engine.py        # Core training logic
    │   ├── losses.py              # Loss weights and PCGrad optimizer wrapper
    │   ├── utils.py               # Utilities (Seed, Logger)
    │   └── visualization.py       # Academic plotting functions
    ├── plot_waveforms.py          # Script for plotting oscilloscope-style waveforms
    ├── train.py                   # Main training entry point
    ├── evaluate.py                # Main evaluation and comparison entry point
    └── requirements.txt           # Python dependencies

## ⚙️ Installation

Clone this repository and install the required dependencies:

```bash
git clone [https://github.com/KrisCCeng/SynRM-PINN.git](https://github.com/KrisCCeng/SynRM-PINN.git)
cd SynRM-PINN
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Training the Models

You can train three different types of models:
* `ANN`: Traditional Black-box Neural Network.
* `SCN`: Structure-Constrained Network (Implicit Inductance + Absolute Symmetry).
* `PINN`: Fully Constrained Network (SCN + Maxwell Reciprocity via PCGrad).

Modify the `configs/pinn_train.yaml` file to set your `mode`, `data_dir`, and `w_phys` (physical loss weight). Then run:

```bash
python train.py --config pinn.yaml
```

*To perform ablation studies on the physical loss weight, simply change the `w_phys` parameter in the `.yaml` file.*

### 2. Evaluation and Comparison

Evaluate all trained models and analytical baselines. This script will automatically calculate MAE, RMSE, MAPE, and generate high-resolution `SVG` visualizations including 3D flux surfaces and error heatmaps:

```bash
python evaluate.py --config pinn.yaml
```

### 3. Plotting Experimental Waveforms

To plot oscilloscope-style waveform figures for the standstill experimental datasets:

```bash
python plot_waveforms.py --data_dir data/processed/ExDataset1
```

## 📊 Evaluation Metrics

The framework will automatically generate an `evaluation_metrics.csv` report and print a formatted table in the console, comparing the training time and accuracy (d/q-axis MAE, RMSE, MAPE) across the Traditional Polynomial model, ANN, SCN, and PINN.

## 📝 Citation

If you find this code or our paper useful for your research, please consider citing our work:

```bibtex
@article{Ye2026SynRM,
  title={High-Precision Standstill Flux Linkage Identification for Synchronous Reluctance Motors Based on Gradient Projection and Implicit Inductance Modeling},
  author={Cheng Ye and Yankai Jia and Yusheng Song and Changxing Guo and Chuanwen Shen},
  journal={Submitted to Journal},
  year={2026}
}
```
```
