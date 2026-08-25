from __future__ import annotations

import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    MaxPooling1D,
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2

from config import ModelConfig


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def kernel_size_for_dimension(dimension: int) -> int:
    if dimension <= 20:
        return 3
    if dimension <= 100:
        return 6
    return 9


def filter_schedule(sample_count: int) -> list[int]:
    return [16, 32, 64] if sample_count < 20_000 else [64, 128, 256]


def build_dynamic_cnn(
    feature_count: int,
    training_samples: int,
    class_count: int,
    config: ModelConfig,
) -> tf.keras.Model:
    if feature_count < 1:
        raise ValueError("A model requires at least one feature")
    kernel_size = kernel_size_for_dimension(feature_count)
    filters = filter_schedule(training_samples)
    model = Sequential(name="dynamic_1d_cnn")
    model.add(Input(shape=(feature_count, 1)))
    model.add(Conv1D(filters[0], kernel_size, padding="same", kernel_regularizer=l2(config.l2_penalty)))
    model.add(Activation("tanh"))
    model.add(Dropout(config.dropout))
    model.add(BatchNormalization())

    extra_blocks = 0 if feature_count <= 20 else (1 if feature_count <= 100 else 2)
    if extra_blocks >= 1:
        model.add(Conv1D(filters[1], kernel_size, padding="same", kernel_regularizer=l2(config.l2_penalty)))
        model.add(Activation("relu"))
        model.add(Dropout(config.dropout))
        model.add(BatchNormalization())
    if extra_blocks >= 2:
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Conv1D(filters[2], kernel_size, padding="same", kernel_regularizer=l2(config.l2_penalty)))
        model.add(LeakyReLU(negative_slope=0.1))
        model.add(Dropout(config.dropout))
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(max(32, filters[-1]), kernel_regularizer=l2(config.l2_penalty)))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(Dropout(config.dropout))
    output_units = 1 if class_count == 2 else class_count
    output_activation = "sigmoid" if class_count == 2 else "softmax"
    model.add(Dense(output_units, activation=output_activation, kernel_regularizer=l2(config.l2_penalty)))

    optimizer = (
        Adam(learning_rate=1e-3)
        if training_samples < 20_000
        else SGD(learning_rate=2e-3, momentum=0.9, nesterov=True)
    )
    loss = "binary_crossentropy" if class_count == 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def batch_size_for_samples(sample_count: int, config: ModelConfig) -> int:
    if sample_count < 2_000:
        return config.batch_size_small
    if sample_count < 20_000:
        return config.batch_size_medium
    return config.batch_size_large


def fit_with_internal_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_count: int,
    seed: int,
    config: ModelConfig,
) -> tf.keras.Model:
    set_random_seed(seed)
    model = build_dynamic_cnn(x_train.shape[1], len(x_train), class_count, config)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            min_delta=1e-3,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=config.reduce_lr_patience,
            factor=0.5,
            min_lr=1e-5,
            verbose=0,
        ),
    ]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size_for_samples(len(x_train), config),
        epochs=config.max_epochs,
        callbacks=callbacks,
        verbose=0,
    )
    return model
