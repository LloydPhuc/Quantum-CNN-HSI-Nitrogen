"""
Feature extraction, Fisher discriminant score ranking, PCA whitening, and [0, pi] angle scaling.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from .config import SEED, FISHER_TOP_K, N_PCA_FEATURES

def compute_fisher_discriminant_scores(X, y):
    """
    Compute Fisher discriminant ratio for each feature column.
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    overall_mean = X.mean(axis=0)
    for j in range(n_features):
        between_var = 0.0
        within_var = 0.0
        for c in classes:
            Xc = X[y == c, j]
            nc = len(Xc)
            between_var += nc * (Xc.mean() - overall_mean[j]) ** 2
            within_var += nc * Xc.var()
        scores[j] = between_var / (within_var + 1e-10)
    return scores

def extract_and_reduce_features(cnn_model, X_train, y_train, X_val, X_test, n_components=N_PCA_FEATURES):
    """
    Full pipeline: Attentive_Fusion (112d) -> Fisher Ranking (top 64) -> PCA Whitening (16d) -> MinMaxScaler [0, pi].
    """
    latent_model = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("Attentive_Fusion").output
    )
    
    feat_train = latent_model.predict(X_train, verbose=0, batch_size=256)
    feat_val   = latent_model.predict(X_val, verbose=0, batch_size=256)
    feat_test  = latent_model.predict(X_test, verbose=0, batch_size=256)
    
    # 1. Fisher Filtering (112d -> 64d)
    fisher_scores = compute_fisher_discriminant_scores(feat_train, y_train)
    top_indices = np.argsort(fisher_scores)[::-1][:FISHER_TOP_K]
    
    feat_tr_f = feat_train[:, top_indices]
    feat_va_f = feat_val[:, top_indices]
    feat_te_f = feat_test[:, top_indices]
    
    # 2. PCA Whitening (64d -> 16d)
    scaler = StandardScaler()
    feat_tr_s = scaler.fit_transform(feat_tr_f)
    feat_va_s = scaler.transform(feat_va_f)
    feat_te_s = scaler.transform(feat_te_f)
    
    pca = PCA(n_components=n_components, whiten=True, random_state=SEED)
    feat_tr_pca = pca.fit_transform(feat_tr_s)
    feat_va_pca = pca.transform(feat_va_s)
    feat_te_pca = pca.transform(feat_te_s)
    
    # 3. Angle Scaling [0, pi]
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    feat_train_q = angle_scaler.fit_transform(feat_tr_pca).astype("float32")
    feat_val_q   = angle_scaler.transform(feat_va_pca).astype("float32")
    feat_test_q  = angle_scaler.transform(feat_te_pca).astype("float32")
    
    return feat_train_q, feat_val_q, feat_test_q
