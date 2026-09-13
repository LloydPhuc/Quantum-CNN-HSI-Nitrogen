"""
Dual-Branch Classical CNN Backbone with Inception + ResNet and Gated Attention Fusion.
Synced with Kaggle notebook v10 (qcnn-feature-extraction.ipynb, Cell 7).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from .config import PATCH_SIZE, N_BANDS, N_CLASSES, CNN_LR


def simple_inception(x, filters):
    """3-branch Inception module: 1x1, 1x1->3x3, 1x1->5x5."""
    b1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)

    b2 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    b2 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(b2)

    b3 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    b3 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(b3)

    return layers.Concatenate()([b1, b2, b3])


def simple_resblock(x, filters):
    """2x Conv3x3 ResBlock with conditional 1x1 shortcut projection."""
    res = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    res = layers.Conv2D(filters, (3, 3), padding='same')(res)
    if x.shape[-1] != filters:
        x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    out = layers.Add()([x, res])
    return layers.Activation('relu')(out)


def build_dual_branch_qcnn(input_shape=(PATCH_SIZE, PATCH_SIZE, N_BANDS), n_classes=N_CLASSES):
    """
    Build Dual-Branch Inception-ResNet CNN with Gated Attentive Fusion (112d).
    Architecture matches Kaggle v10 notebook exactly.
    """
    inputs = layers.Input(shape=input_shape)

    # Inception Branch: 32 filters x 3 branches = 96d after GAP
    b1 = simple_inception(inputs, 32)
    b1 = layers.GlobalAveragePooling2D()(b1)

    # ResNet Branch: 64 filters = 64d after GAP
    b2 = simple_resblock(inputs, 64)
    b2 = layers.GlobalAveragePooling2D()(b2)

    # Gated Attentive Fusion: concat 160d -> sigmoid gate -> multiply -> Dense 112
    concat = layers.Concatenate()([b1, b2])
    gate = layers.Dense(concat.shape[-1], activation='sigmoid')(concat)
    fused = layers.Multiply()([concat, gate])
    fused = layers.Dense(112, activation='relu', name='Attentive_Fusion')(fused)

    outputs = layers.Dense(n_classes, activation='softmax', name='cnn_output')(fused)

    model = tf.keras.Model(inputs, outputs, name='DualBranch_GatedFusion_CNN')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CNN_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
