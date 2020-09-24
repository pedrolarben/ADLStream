"""Creates Multi Layer Perceptron (MLP) model."""
import tensorflow as tf


def MLP(
    input_shape, output_size, loss, optimizer, hidden_layers=[32, 16, 8], dropout=0.0,
):
    """Multi Layer Perceptron.

    Args:
        input_shape (tuple): Shape of the input data
        output_size (int): Number of neurons of the last layer.
        loss (tf.keras.Loss): Loss to be use for training.
        optimizer (tf.keras.Optimizer): Optimizer that implements theraining algorithm.
        hidden_layers (list, optional): List of neurons of the hidden layers. 
            Defaults to [32, 16, 8].
        dropout (float between 0 and 1, optional): Fraction of the dense units to drop.
            Defaults to 0.0.

    Returns:
        tf.keras.Model: MPL model.
    """
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    x = tf.keras.layers.Flatten()(inputs)  # Convert the 2d input in a 1d array
    for hidden_units in hidden_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model
