"""
End-to-End Orchestrator Pipeline for QCNN.
Synced with Kaggle notebooks v10 (Feature Extraction) + v9 (Quantum Classifier).
"""

import os
import time
import numpy as np
import tensorflow as tf
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .config import (
    CNN_BATCH_SIZE, CNN_MAX_EPOCHS, CNN_PATIENCE_ES, CNN_PATIENCE_LR,
    VQC_TRAIN_SIZE, VQC_BATCH_SIZE, VQC_EPOCHS, VQC_LR, VQC_PATIENCE,
    CLASSICAL_LR, CLASSICAL_WEIGHT_DECAY, CLASSICAL_EPOCHS, CLASSICAL_PATIENCE,
)
from .data_loader import load_and_preprocess_cube, extract_patches, create_stratified_splits
from .classical_backbone import build_dual_branch_qcnn
from .feature_extraction import extract_and_reduce_features
from .quantum_vqc import HybridVQCClassifier
from .train_eval import (
    ClassicalDenseHead, train_model, evaluate_model, noise_robustness_mc, BATCH_SIZE,
)


def run_qcnn_pipeline(path_X, path_y, path_wave=None, export_dir='qcnn_features'):
    """
    Stage 1 (matches qcnn-feature-extraction v10):
      Load cube -> band select -> Z-score -> patches -> stratified split
      -> Dual-Branch CNN (Keras) -> Attentive_Fusion 112d
      -> Fisher top-64 -> PCA whitening 16d -> MinMax [0, pi] -> export .npy

    Stage 2 (matches qcnn-quantum-classifier v9):
      Classical Dense Head vs Quantum VQC Head, fair 1:1 on full training set,
      then Monte Carlo noise robustness.
    """
    device = torch.device('cpu')
    print("🚀 Starting QCNN Hybrid Quantum-Classical Pipeline...")

    # ---------- Stage 1: Data + CNN backbone ----------
    X_cube, y_mask = load_and_preprocess_cube(path_X, path_y, path_wave)
    patches, labels = extract_patches(X_cube, y_mask)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_stratified_splits(patches, labels)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("🧠 Building and training Dual-Branch Gated Fusion CNN...")
    n_classes = len(np.unique(y_train))
    cnn = build_dual_branch_qcnn(n_classes=n_classes)
    cnn.summary()

    start = time.time()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=CNN_PATIENCE_ES,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=CNN_PATIENCE_LR, min_lr=1e-5),
    ]
    cnn.fit(X_train, y_train, validation_data=(X_val, y_val),
            batch_size=CNN_BATCH_SIZE, epochs=CNN_MAX_EPOCHS,
            callbacks=callbacks, verbose=1)
    print(f"⏱️ CNN training time: {time.time() - start:.1f}s")

    # ---------- Feature extraction + dimensionality reduction ----------
    print("🔬 Extracting Attentive_Fusion features + Fisher/PCA reduction...")
    feat_train_q, feat_val_q, feat_test_q = extract_and_reduce_features(
        cnn, X_train, y_train, X_val, X_test
    )

    os.makedirs(export_dir, exist_ok=True)
    np.save(f'{export_dir}/feat_train_q.npy', feat_train_q)
    np.save(f'{export_dir}/feat_val_q.npy', feat_val_q)
    np.save(f'{export_dir}/feat_test_q.npy', feat_test_q)
    np.save(f'{export_dir}/y_train.npy', y_train)
    np.save(f'{export_dir}/y_val.npy', y_val)
    np.save(f'{export_dir}/y_test.npy', y_test)
    print(f"💾 Features exported to {export_dir}/")

    # ---------- Stage 2: Classifier heads (fair 1:1 comparison) ----------
    train_dataset = TensorDataset(torch.tensor(feat_train_q, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(feat_val_q, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(feat_test_q, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    # Classical Dense Head
    print("🏛️ Training Classical Dense Head...")
    classical_model = ClassicalDenseHead(in_features=feat_train_q.shape[1], n_classes=n_classes).to(device)
    optimizer_c = optim.Adam(classical_model.parameters(), lr=CLASSICAL_LR, weight_decay=CLASSICAL_WEIGHT_DECAY)
    scheduler_c = optim.lr_scheduler.ReduceLROnPlateau(optimizer_c, mode='min', factor=0.5, patience=5)
    history_classical, classical_time = train_model(
        classical_model, train_loader, val_loader, optimizer_c, criterion, scheduler_c,
        epochs=CLASSICAL_EPOCHS, patience=CLASSICAL_PATIENCE, device=device
    )

    # Quantum VQC Head (full training set unless VQC_TRAIN_SIZE is set)
    print("⚛️ Training Quantum VQC Head...")
    if VQC_TRAIN_SIZE is not None and VQC_TRAIN_SIZE < len(feat_train_q):
        np.random.seed(42)
        sel = np.random.choice(len(feat_train_q), VQC_TRAIN_SIZE, replace=False)
        X_vqc, y_vqc = feat_train_q[sel], y_train[sel]
    else:
        X_vqc, y_vqc = feat_train_q, y_train
    vqc_train_loader = DataLoader(
        TensorDataset(torch.tensor(X_vqc, dtype=torch.float32), torch.tensor(y_vqc, dtype=torch.long)),
        batch_size=VQC_BATCH_SIZE, shuffle=True
    )
    vqc_model = HybridVQCClassifier(n_classes=n_classes).to(device)
    optimizer_v = optim.Adam(vqc_model.parameters(), lr=VQC_LR)
    scheduler_v = optim.lr_scheduler.ReduceLROnPlateau(optimizer_v, mode='min', factor=0.5, patience=5)
    history_vqc, vqc_time = train_model(
        vqc_model, vqc_train_loader, val_loader, optimizer_v, criterion, scheduler_v,
        epochs=VQC_EPOCHS, patience=VQC_PATIENCE, device=device
    )

    # ---------- Evaluation ----------
    res_c = evaluate_model(classical_model, test_loader, device)
    res_v = evaluate_model(vqc_model, test_loader, device)
    print("\n📊 TEST RESULTS")
    print(f"  Classical: Acc={res_c['accuracy']:.4f} | F1={res_c['f1']:.4f} | Kappa={res_c['kappa']:.4f} | {classical_time:.1f}s")
    print(f"  Quantum:   Acc={res_v['accuracy']:.4f} | F1={res_v['f1']:.4f} | Kappa={res_v['kappa']:.4f} | {vqc_time:.1f}s")

    # ---------- Noise Robustness ----------
    print("\n🔊 Running Monte Carlo noise robustness...")
    noise_summary = noise_robustness_mc(
        {'Classical': classical_model, 'Quantum': vqc_model},
        feat_test_q, y_test
    )

    return {
        'classical': {'model': classical_model, 'results': res_c, 'history': history_classical, 'time': classical_time},
        'vqc': {'model': vqc_model, 'results': res_v, 'history': history_vqc, 'time': vqc_time},
        'noise_summary': noise_summary,
        'features': (feat_train_q, feat_val_q, feat_test_q),
    }


if __name__ == '__main__':
    pass