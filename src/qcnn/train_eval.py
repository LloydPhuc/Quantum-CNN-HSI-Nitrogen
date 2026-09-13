"""
Training loops, classifier heads, evaluation metrics, and noise robustness analysis.
Synced with Kaggle notebook v9 (qcnn-quantum-classifier.ipynb, Cells 3, 5, 7).
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    cohen_kappa_score, f1_score
)
from .config import (
    CLASSICAL_HIDDEN_1, CLASSICAL_HIDDEN_2,
    CLASSICAL_DROPOUT_1, CLASSICAL_DROPOUT_2,
    N_CLASSES, NOISE_LEVELS, N_TRIALS, SEED
)

BATCH_SIZE = 64


class ClassicalDenseHead(nn.Module):
    """Classical Dense classifier: 16 -> 64 -> 32 -> 3 with Dropout (3,267 params)."""
    def __init__(self, in_features=16, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, CLASSICAL_HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(CLASSICAL_DROPOUT_1),
            nn.Linear(CLASSICAL_HIDDEN_1, CLASSICAL_HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(CLASSICAL_DROPOUT_2),
            nn.Linear(CLASSICAL_HIDDEN_2, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler,
                epochs=100, patience=10, device='cpu'):
    """
    PyTorch training loop with early stopping and LR scheduler.
    Uses deep clone for best state checkpoint (bug fix from v8).
    """
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_b.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_total_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                v_total_loss += loss.item() * X_b.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == y_b).sum().item()
                v_total += y_b.size(0)

        val_loss = v_total_loss / v_total
        val_acc = v_correct / v_total
        scheduler.step(val_loss)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    elapsed_time = time.time() - start_time
    model.load_state_dict(best_state)
    return history, elapsed_time


def get_predictions(model, loader, device='cpu'):
    """Run inference and return (predictions, labels) as numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            outputs = model(X_b)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.numpy())
    return np.array(all_preds), np.array(all_labels)


def evaluate_model(model, data_loader, device='cpu'):
    """Compute Accuracy, Weighted F1, and Cohen's Kappa."""
    y_pred, y_true = get_predictions(model, data_loader, device)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'y_pred': y_pred,
        'y_true': y_true
    }


def noise_robustness_mc(models, feat_test_q, y_test, noise_levels=None,
                        n_trials=N_TRIALS, batch_size=BATCH_SIZE, seed=SEED):
    """
    Monte Carlo noise robustness evaluation (Q1 academic standard).
    Adds Gaussian noise to quantum-encoded features, clipped to [0, pi].

    Args:
        models: dict of {name: model} to evaluate (e.g. {'classical': m1, 'vqc': m2})
        feat_test_q: test features in [0, pi] range
        y_test: test labels
    Returns:
        dict of {name: (mean_array, std_array)} across noise levels
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    results = {name: np.zeros((len(noise_levels), n_trials)) for name in models}

    for i, sigma in enumerate(noise_levels):
        for t in range(n_trials):
            np.random.seed(seed + t + int(sigma * 100))
            noise = np.random.normal(0, sigma, feat_test_q.shape).astype('float32')
            feat_noisy = np.clip(feat_test_q + noise, 0, np.pi)

            noisy_loader = DataLoader(
                TensorDataset(
                    torch.tensor(feat_noisy, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.long)
                ),
                batch_size=batch_size, shuffle=False
            )

            for name, model in models.items():
                preds, _ = get_predictions(model, noisy_loader)
                results[name][i, t] = accuracy_score(y_test, preds)

    summary = {name: (results[name].mean(axis=1), results[name].std(axis=1))
               for name in models}
    return summary
