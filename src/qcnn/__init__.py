"""
QCNN: Hybrid Quantum-Classical CNN for Hyperspectral Nitrogen Classification.

A two-stage pipeline:
1. Classical Dual-Branch Inception-ResNet + Gated Attention Fusion (Feature Extraction)
2. Quantum VQC with Dense Angle Encoding (Classification)
"""

from .config import (
    SEED,
    SELECTED_BANDS,
    N_BANDS,
    PATCH_SIZE,
    MAX_PATCHES,
    SPLIT_RATIO,
    N_QUBITS,
    N_VQC_LAYERS,
    N_CLASSES,
    CLASS_NAMES,
)

from .data_loader import (
    load_and_preprocess_cube,
    extract_patches,
    create_stratified_splits,
)

from .classical_backbone import (
    build_dual_branch_qcnn,
    simple_inception,
    simple_resblock,
)

from .feature_extraction import (
    compute_fisher_discriminant_scores,
    extract_and_reduce_features,
)

from .quantum_vqc import (
    build_quantum_circuit,
    HybridVQCClassifier,
)

from .train_eval import (
    ClassicalDenseHead,
    train_model,
    evaluate_model,
    get_predictions,
    noise_robustness_mc,
)

from .pipeline import run_qcnn_pipeline

__version__ = "0.1.0"
__author__ = "Huu Phuc Le & Minh Tan Le Nguyen"
__all__ = [
    # Config
    "SEED",
    "SELECTED_BANDS",
    "N_BANDS",
    "PATCH_SIZE",
    "MAX_PATCHES",
    "SPLIT_RATIO",
    "N_QUBITS",
    "N_VQC_LAYERS",
    "N_CLASSES",
    "CLASS_NAMES",
    # Data
    "load_and_preprocess_cube",
    "extract_patches",
    "create_stratified_splits",
    # Classical Backbone
    "build_dual_branch_qcnn",
    "simple_inception",
    "simple_resblock",
    # Feature Extraction
    "compute_fisher_discriminant_scores",
    "extract_and_reduce_features",
    # Quantum VQC
    "build_quantum_circuit",
    "HybridVQCClassifier",
    # Training & Evaluation
    "ClassicalDenseHead",
    "train_model",
    "evaluate_model",
    "get_predictions",
    "noise_robustness_mc",
    # Pipeline
    "run_qcnn_pipeline",
]