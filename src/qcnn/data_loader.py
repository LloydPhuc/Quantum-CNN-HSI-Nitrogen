"""
Data loader and patch extraction module for Hyperspectral Images (HSI).
Synced with Kaggle notebook v10 (qcnn-feature-extraction.ipynb, Cell 4).
"""

import gc
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .config import SEED, SELECTED_BANDS, PATCH_SIZE, MAX_PATCHES, SPLIT_RATIO


def load_and_preprocess_cube(path_X, path_y, path_wave=None):
    """
    Load HSI cube, remap labels, select 64 bands, apply Z-score normalization.

    Label convention (from Eggplant_Labels.npy):
        0 = background, 1 = Low N, 2 = Medium N, 3 = High N.
    Output: background -> -1, classes remapped to 0/1/2.

    Z-score uses StandardScaler on the flattened cube (matches v10 notebook,
    verified as standard HSI single-scene practice, not a data leak).
    """
    print(f"📥 Loading HSI cube from {path_X}...")
    cube = np.load(path_X)
    labels = np.load(path_y)

    # Remap labels: background (0) -> -1, valid classes 1/2/3 -> 0/1/2
    valid_mask = labels > 0
    unique_valid = np.unique(labels[valid_mask])
    labels[~valid_mask] = -1
    for i, l in enumerate(unique_valid):
        labels[labels == l] = i

    # Band selection (260 -> 64 bands)
    print(f"Selecting {len(SELECTED_BANDS)} bands...")
    cube = cube[:, :, SELECTED_BANDS]
    gc.collect()

    # Z-score normalization (StandardScaler on flattened cube, per v10)
    print("Normalizing bands (Z-score)...")
    h, w, c = cube.shape
    cube_reshaped = cube.reshape(-1, c)
    scaler = StandardScaler()
    cube_normalized = scaler.fit_transform(cube_reshaped).reshape(h, w, c)
    del cube, cube_reshaped
    gc.collect()

    return cube_normalized, labels


def extract_patches(data, labels, patch_size=PATCH_SIZE, max_patches=MAX_PATCHES):
    """
    Extract spatial-spectral patches from labeled (non-background) pixels.
    Coordinates are subsampled BEFORE extraction to avoid OOM (v5 bug fix).
    """
    pad = patch_size // 2
    padded_data = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    valid_y, valid_x = np.where(labels >= 0)

    # Subsample coordinates before building patches to avoid OOM
    if len(valid_y) > max_patches:
        print(f"Subsampling coordinates from {len(valid_y)} to {max_patches}...")
        np.random.seed(SEED)
        idx = np.random.choice(len(valid_y), max_patches, replace=False)
        valid_y = valid_y[idx]
        valid_x = valid_x[idx]

    patches = []
    patch_labels = []
    for r, c in zip(valid_y, valid_x):
        patches.append(padded_data[r:r + patch_size, c:c + patch_size, :])
        patch_labels.append(labels[r, c])

    return np.array(patches, dtype=np.float32), np.array(patch_labels, dtype=np.int32)


def create_stratified_splits(patches, labels, split_ratio=SPLIT_RATIO):
    """
    Stratified 70/15/15 split into Train / Val / Test (matches v10 Cell 5).
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        patches, labels, test_size=split_ratio[2], stratify=labels, random_state=SEED
    )
    del patches, labels
    gc.collect()

    val_ratio = split_ratio[1] / (split_ratio[0] + split_ratio[1])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=SEED
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
