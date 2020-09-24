"""Creates Convolutional Neural Network (CNN) model."""

import tensorflow as tf


def CNN(
    input_shape,
    output_size,
    loss,
    optimizer,
    conv_layers=[64, 128],
    kernel_sizes=[7, 5],
    pool_sizes=[2, 2],
    dense_layers=[],
    dense_dropout=0.0,
):
    """Convolutional Neural Network (CNN).

    Args:
        input_shape (tuple): Shape of the input data
        output_size (int): Number of neurons of the last layer.
        loss (tf.keras.Loss): Loss to be use for training.
        optimizer (tf.keras.Optimizer): Optimizer that implements theraining algorithm.
        conv_layers (list, optional): 
            Number of convolutional filters for each convolutional layer. 
            Defaults to [64, 128].
        kernel_sizes (list, optional): 
            Kernel size for each convolutional layer. 
            Defaults to [7, 5].
        pool_sizes (list, optional): 
            Pooling factor to be performed after each convolutional layer. 
            Defaults to [2, 2].
        dense_layers (list, optional): List with the number of hidden neurons for each 
            layer of the dense block before the output. 
            Defaults to [].
        dense_dropout (float between 0 and 1, optional): Fraction of the dense units to drop.
            Defaults to 0.0.

    Returns:
        tf.keras.Model: CNN model
    """
    assert len(conv_layers) == len(kernel_sizes)
    assert 0 <= dense_dropout <= 1
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # First conv block
    x = tf.keras.layers.Conv1D(
        conv_layers[0], kernel_sizes[0], activation="relu", padding="same"
    )(inputs)
    if pool_sizes[0] and x.shape[-2] // pool_sizes[0] > 1:
        x = tf.keras.layers.MaxPool1D(pool_size=pool_sizes[0])(x)
    # Rest of the conv blocks
    for chanels, kernel_size, pool_size in zip(
        conv_layers[1:], kernel_sizes[1:], pool_sizes[1:]
    ):
        x = tf.keras.layers.Conv1D(
            chanels, kernel_size, activation="relu", padding="same"
        )(x)
        if pool_size and x.shape[-2] // pool_size > 1:
            x = tf.keras.layers.MaxPool1D(pool_size=pool_size)(x)
    # Dense block
    x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            tf.keras.layers.Dropout(dense_dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)
    return model
