# Hybrid Quantum-Classical CNN for Hyperspectral Nitrogen Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35+-purple.svg)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Quantum-Classical Hybrid Pipeline for Eggplant Nitrogen Stress Classification from Hyperspectral Imaging (HSI)**
>
> Achieves **94.50% test accuracy** with only **99 trainable parameters** (33× fewer than classical baseline) using an 8-qubit Variational Quantum Circuit (VQC) on NISQ-era simulators.

---

## 🏗️ Architecture Overview

### Classical Backbone (Feature Extractor)
- **Input**: 11×11×64 spatial-spectral patches (64 optimal bands from 260 HSI bands)
- **Dual-Branch**: Inception (multi-scale 1×1, 3×3, 5×5) + ResNet (residual highway)
- **Gated Attention Fusion**: Adaptive feature weighting → 112-dim `Attentive_Fusion` latent vector
- **Framework**: TensorFlow/Keras

### Quantum Classifier Head (VQC)
```
112-dim Attentive_Fusion
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  QUANTUM PIPELINE (This Repository's Core Contribution)     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Fisher Discriminant Ranking                                │
│  112-dim ──────────────────▶ Top-64 features               │
│       │                                                    │
│       ▼                                                    │
│  PCA Whitening (whiten=True)                                │
│  64-dim ──────────────────▶ 16-dim orthogonal (Cov = I₁₆)  │
│       │                                                    │
│       ▼                                                    │
│  MinMaxScaler [0, π]                                        │
│  16-dim ──────────────────▶ Angles for quantum encoding    │
│       │                                                    │
│       ▼                                                    │
│  Dense Angle Encoding (2 features/qubit)                    │
│  16 angles ─────────────▶ 8 qubits: RY(θ₁)RZ(θ₂) per qubit │
│       │                                                    │
│       ▼                                                    │
│  Variational Quantum Circuit (VQC)                          │
│  StronglyEntanglingLayers × 3 layers = 72 quantum params   │
│  Full all-to-all entanglement in ℋ = ℂ²⁵⁶                  │
│       │                                                    │
│       ▼                                                    │
│  Measurement: ⟨Z₀⟩...⟨Z₇⟩ ∈ [-1, 1]⁸                        │
│       │                                                    │
│       ▼                                                    │
│  Classical Readout: Linear(8→3) + Softmax = 27 params      │
│       │                                                    │
│       ▼                                                    │
│  Output: 3-class probabilities (Low N, Med N, High N)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Total Quantum Parameters**: 72 (VQC) + 27 (Readout) = **99 parameters**  
**Classical Baseline**: 3,267 parameters (Dense 64→32→3)  
**Compression Ratio**: **33× fewer parameters**

---

## 📊 Key Results

| Metric | Classical Dense | Quantum VQC | Advantage |
|--------|----------------|-------------|-----------|
| **Test Accuracy** | 94.48% | **94.50%** | +0.02% |
| **Weighted F1** | 0.9448 | **0.9450** | +0.0002 |
| **Cohen's κ** | 0.9149 | **0.9153** | +0.0004 |
| **Trainable Params** | 3,267 | **99** | **33× fewer** |
| **Train Samples** | 69,999 | **10,000** | **7× data efficient** |
| **Acc / 10k Params** | 2.89 | **95.45** | **33× efficient** |

### Noise Robustness (Gaussian Phase Jitter)
| Noise σ | Classical | Quantum | Winner |
|---------|-----------|---------|--------|
| 0.00 (Clean) | 94.48% | **94.50%** | Quantum |
| 0.05 | 90.07% | **90.66%** | Quantum |
| 0.10 | 81.35% | **82.21%** | Quantum |
| 0.20 | 67.39% | **68.01%** | Quantum |
| 0.50 | **50.21%** | 49.00% | Classical |

> **Finding**: Quantum VQC outperforms classical up to σ=0.20, consistent with **Noise-Induced Regularization** theory (Kuzmin et al., Adv. Quantum Tech. 2025).

---

## 📁 Repository Structure

```
qcnn/
├── src/qcnn/                 # Core Python package
│   ├── __init__.py
│   ├── config.py             # Hyperparameters & band selection
│   ├── data_loader.py        # HSI cube loading, patch extraction, splits
│   ├── classical_backbone.py # Dual-branch Inception-ResNet + Gated Fusion
│   ├── feature_extraction.py # Fisher → PCA Whitening → [0,π] scaling
│   ├── quantum_vqc.py        # PennyLane VQC (8 qubits, 3 layers)
│   ├── train_eval.py         # Training loops, evaluation, noise robustness
│   └── pipeline.py           # End-to-end orchestrator
├── configs/
│   └── eggplant_bands.txt    # 64 selected band indices
├── data/
│   └── eggplant_bands.txt    # Copy for data pipeline access
├── notebooks/
│   └── qcnn_pipeline_demo.ipynb  # Main demonstration notebook
├── results/
│   ├── figures/              # Generated plots (noise robustness, ablation)
│   ├── logs/                 # Training logs
│   └── features/             # Extracted .npy features & ablation CSVs
├── scripts/
│   └── run_pipeline.py       # CLI entry point
├── tests/                    # Unit tests (to be added)
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── LICENSE
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/qcnn-hsi.git
cd qcnn-hsi
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python -m src.qcnn.pipeline \
  --data_path /path/to/reflectance_260bands-003.npy \
  --label_path /path/to/Eggplant_Labels.npy \
  --wave_path /path/to/wavelengths_260bands.npy
```

### Or Use the Demo Notebook
```bash
jupyter lab notebooks/qcnn_pipeline_demo.ipynb
```

---

## 🔬 Reproducing Results

### Classical Baseline Only
```python
from src.qcnn import build_dual_branch_qcnn, run_qcnn_pipeline
# Trains CNN backbone, extracts Attentive_Fusion features
```

### Quantum Head Comparison
```python
from src.qcnn import HybridVQCClassifier, ClassicalDenseHead
# Fair 1:1 comparison on same 10k/69,999 training samples
```

### Noise Robustness Evaluation
```python
from src.qcnn import noise_robustness_mc
# Monte Carlo σ ∈ [0.0, 0.5], 5 trials each
```

---

## 📚 Theoretical Foundations

This work integrates four foundational papers:

| Paper | Venue | Contribution to This Work |
|-------|-------|---------------------------|
| **Quanv4EO** (Sebastianelli et al.) | IEEE TGRS 2022 | Quanvolutional layers for Earth Observation |
| **QEDNet** (Lin, Tang & Huete) | IEEE TGRS 2023 | Spatial-spectral continuity & Quantum Feature Empowerment |
| **Hybrid Pipeline** (Tomar et al.) | IEEE Access 2024 | 2-Stage: Classical Extractor → Quantum VQC |
| **Noise-Induced Regularization** (Kuzmin et al.) | Adv. Quantum Tech. 2025 | Unitary transforms as geometric regularizers |

---

## 🖥️ Hardware Feasibility (NISQ)

| Metric | Value | Threshold |
|--------|-------|-----------|
| **Qubits** | 8 | ✅ < 100 (IBM Heron/Eagle) |
| **CNOT Gates** | 24 | ✅ Low depth |
| **Circuit Depth** | ~18 | ✅ ≪ T₂ (80-150 μs) |
| **Execution Time** | ~0.9 μs | ✅ 1.1% of T₂ |

**Verdict**: Circuit executes **before decoherence** — deployable on real IBM Quantum hardware.

---

## 📝 Citation

```bibtex
@misc{qcnn-hsi-2025,
  title={Hybrid Quantum-Classical CNN for Hyperspectral Nitrogen Classification},
  author={Le, Huu Phuc and Le Nguyen, Minh Tan},
  year={2025},
  note={Cần Giờ Quantum Remote Sensing Project}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---
