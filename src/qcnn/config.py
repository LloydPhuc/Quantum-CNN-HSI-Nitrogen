"""
Configuration parameters for QCNN-HSI Hybrid Quantum-Classical Framework.
Synced with Kaggle notebooks v10 (Feature Extraction) and v9 (Quantum Classifier).
"""

# Random Seed for Reproducibility
SEED = 42

# Selected 64 Optimal Bands (out of 260 HSI bands)
SELECTED_BANDS = [
    6, 29, 39, 65, 71, 73, 79, 83, 91, 93, 95, 108, 114, 116, 133, 139, 145, 147,
    151, 152, 156, 159, 163, 166, 170, 176, 177, 180, 181, 183, 185, 186, 189,
    190, 194, 197, 199, 204, 205, 206, 207, 211, 212, 214, 224, 228, 231, 232,
    237, 238, 242, 244, 245, 247, 248, 250, 251, 252, 253, 254, 255, 256, 257, 258
]
N_BANDS = len(SELECTED_BANDS)  # 64

# Spatial & Patch Extraction Config
PATCH_SIZE = 11
MAX_PATCHES = 100000  # RAM safety cap for 100k patches
SPLIT_RATIO = (0.70, 0.15, 0.15)  # Train / Val / Test (Stratified)

# CNN Backbone Training Config (TensorFlow/Keras)
CNN_BATCH_SIZE = 64
CNN_MAX_EPOCHS = 150
CNN_LR = 1e-3
CNN_PATIENCE_ES = 7
CNN_PATIENCE_LR = 4

# Dimensionality Reduction Config
FISHER_TOP_K = 64
N_PCA_FEATURES = 16  # PCA dimension for Quantum encoding

# Classical Dense Head Config (PyTorch)
CLASSICAL_HIDDEN_1 = 64
CLASSICAL_HIDDEN_2 = 32
CLASSICAL_DROPOUT_1 = 0.4
CLASSICAL_DROPOUT_2 = 0.3
CLASSICAL_LR = 0.001
CLASSICAL_WEIGHT_DECAY = 1e-4
CLASSICAL_EPOCHS = 100
CLASSICAL_PATIENCE = 10

# Quantum VQC Architecture Config (PyTorch + PennyLane)
N_QUBITS = 8          # 8 Qubits with Dense Angle Encoding (2 features/qubit)
N_VQC_LAYERS = 3      # StronglyEntanglingLayers depth
VQC_TRAIN_SIZE = None  # v9: None = use FULL training set (69,999 samples) for fair 1:1 comparison
VQC_BATCH_SIZE = 64
VQC_EPOCHS = 80
VQC_LR = 0.01
VQC_PATIENCE = 10

N_CLASSES = 3
CLASS_NAMES = ['Low N', 'Medium N', 'High N']

# Noise Robustness (Monte Carlo) Config
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
N_TRIALS = 5
