"""
PennyLane Variational Quantum Circuit (VQC) with PyTorch TorchLayer integration.
Synced with Kaggle notebook v9 (qcnn-quantum-classifier.ipynb, Cell 4).
"""

import torch
import torch.nn as nn
import pennylane as qml
from .config import N_QUBITS, N_VQC_LAYERS, N_CLASSES


def build_quantum_circuit():
    """Create a fresh QNode (device is created per call to avoid module-level state)."""
    dev = qml.device('default.qubit', wires=N_QUBITS)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def quantum_circuit(inputs, weights):
        # Dense Angle Encoding: 16 features -> 8 qubits (RY + RZ)
        for i in range(N_QUBITS):
            qml.RY(inputs[..., i], wires=i)
            qml.RZ(inputs[..., i + N_QUBITS], wires=i)
        # Variational Ansatz
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        # Multi-Qubit Pauli-Z expectation measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return quantum_circuit


class HybridVQCClassifier(nn.Module):
    """
    Hybrid Quantum-Classical Classifier using PennyLane TorchLayer.
    Architecture: Dense Angle Encoding (16 -> 8 qubits) -> StronglyEntanglingLayers(L=3)
                  -> 8x Pauli-Z expectation -> Linear readout (8 -> 3).
    Total parameters: 72 (quantum) + 27 (readout) = 99.
    """
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_VQC_LAYERS, n_classes=N_CLASSES):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(build_quantum_circuit(), weight_shapes)
        self.fc = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        q_out = self.qlayer(x)
        return self.fc(q_out)
