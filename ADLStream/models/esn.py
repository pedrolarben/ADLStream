"""Creates Echo State Network (ESN) model."""

import tensorflow as tf
import tensorflow_addons as tfa


def ESN(
    input_shape,
    output_size,
    loss,
    optimizer,
    recurrent_units=[64],
    return_sequences=False,
    dense_layers=[32],
    dense_dropout=0,
    out_activation="linear",
):
    """Echo State Network (ESN).

    Args:
        input_shape (tuple): Shape of the input data
        output_size (int): Number of neurons of the last layer.
        loss (tf.keras.Loss): Loss to be use for training.
        optimizer (tf.keras.Optimizer): Optimizer that implements theraining algorithm.
        recurrent_units (list, optional): Number of recurrent units for each ESN layer. 
            Defaults to [64].
        recurrent_dropout (int between 0 and 1, optional): Fraction of the input units to drop.
            Defaults to 0.
        return_sequences (bool, optional): Whether to return the last output in the output sequence, or the full sequence. 
            Defaults to False.
        dense_layers (list, optional): List with the number of hidden neurons for each 
            layer of the dense block before the output. 
            Defaults to [32].
        dense_dropout (float between 0 and 1, optional): Fraction of the dense units to drop.
            Defaults to 0.0.
        out_activation (tf activation function, optional): Activation of the output layer.
            Defaults to "linear".

    Returns:
        tf.keras.Model: ESN model
    """
    input_shape = input_shape[-len(input_shape) + 1 :]
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = inputs
    if len(input_shape) < 2:
        x = tf.keras.layers.Reshape((inputs.shape[1], 1))(x)

    # ESN layers
    for i, u in enumerate(recurrent_units):
        return_sequences_tmp = (
            return_sequences if i == len(recurrent_units) - 1 else True
        )
        x = tfa.layers.ESN(u, return_sequences=return_sequences_tmp, use_norm2=True)(x)

    # Dense layers
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            x = tf.keras.layers.Dropout(dense_dropout)(x)
    x = tf.keras.layers.Dense(output_size, activation=out_activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model
