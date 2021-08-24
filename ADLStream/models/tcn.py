"""Creates Temporal Convolutional Network (TCN) model."""

import tensorflow as tf
import tcn as keras_tcn


def TCN(
    input_shape,
    output_size,
    loss,
    optimizer,
    nb_filters=64,
    kernel_size=2,
    nb_stacks=1,
    dilations=[1, 2, 4, 8, 16],
    tcn_dropout=0.0,
    return_sequences=True,
    activation="linear",
    out_activation="linear",
    padding="causal",
    use_skip_connections=True,
    use_batch_norm=False,
    dense_layers=[],
    dense_dropout=0.0,
    dense_activation="linear",
):
    """Temporal Convolutional Network.

    Args:
        input_shape (tuple): Shape of the input data
        output_size (int): Number of neurons of the last layer.
        loss (tf.keras.Loss): Loss to be use for training.
        optimizer (tf.keras.Optimizer): Optimizer that implements theraining algorithm.
        nb_filters (int, optional): Number of convolutional filtersto use in the TCN blocks.
            Defaults to 64.
        kernel_size (int, optional): The size of the convolutional kernel.
            Defaults to 2.
        nb_stacks (int, optional): The number of stacks of residual blocks to use.
            Defaults to 1.
        dilations (list, optional): The list of the dilations.
            Defaults to [1, 2, 4, 8, 16].
        tcn_dropout (float between 0 and 1, optional): Fraction of the input units to drop.
            Defaults to 0.0.
        return_sequences (bool, optional):
            Whether to return the last output in the output sequence, or the full sequence.
            Defaults to True.
        activation (tf activation function, optional):
            The activation used in the residual blocks o = Activation(x + F(x)).
            Defaults to "linear".
        out_activation (tf activation function, optional): Activation of the output layer.
            Defaults to "linear".
        padding (str, optional): The padding to use in the convolutional layers,
            can be 'causal' or 'same'.
            Defaults to "causal".
        use_skip_connections (bool, optional):
            If we want to add skip connections from input to each residual block.
            Defaults to True.
        use_batch_norm (bool, optional):
            Whether to use batch normalization in the residual layers or not.
            Defaults to False.
        dense_layers (list, optional): List with the number of hidden neurons for each
            layer of the dense block before the output.
            Defaults to [].
        dense_dropout (float between 0 and 1, optional): Fraction of the dense units to drop.
            Defaults to 0.0.
        dense_activation (tf activation function, optional): Activation function of the dense
            layers after the convolutional block.
            Defaults to "linear".

    Returns:
        tf.keras.Model: TCN model
    """
    input_shape = input_shape[-len(input_shape) + 1 :]
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = inputs
    if len(input_shape) < 2:
        x = tf.keras.layers.Reshape((inputs.shape[1], 1))(x)

    x = keras_tcn.TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        use_skip_connections=use_skip_connections,
        dropout_rate=tcn_dropout,
        activation=activation,
        use_batch_norm=use_batch_norm,
        padding=padding,
    )(x)

    # Dense block
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units, activation=dense_activation)(x)
        if dense_dropout > 0:
            tf.keras.layers.Dropout(dense_dropout)(x)
    x = tf.keras.layers.Dense(output_size, activation=out_activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model
