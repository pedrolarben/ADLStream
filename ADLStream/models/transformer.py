import tensorflow as tf
import numpy as np


def positional_encoding(position, d_model):
    """Function that calculates the positional encoding.

    Args:
        position (int): Maximum position encoding (input or output).
        d_model (int): Dimension of the model.

    Returns:
        tf.Tensor: Positional encoding as a tensor with shape (1, position, d_model).
    """

    angle_rates = 1 / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
    )
    angle_rads = np.arange(position)[:, np.newaxis] * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    """Class that creates and computes the multi-head attention layer.

    Args:
        num_heads (int): Number of heads of the multihead attention layer.
        d_model (int): Dimension of the model.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def _scaled_dot_product_attention(self, q, k, v, mask):
        """Function that computes the scaled dot product: softmax(q*k^t/sqrt(dk))*v.

        Args:
            q (tf.Tensor): Query vector with shape (..., seq_len_q, depth).
            k (tf.Tensor): Keys vector with shape (..., seq_len_k, depth).
            v (tf.Tensor): Values vector with shape (..., seq_len_v, depth_v).
            mask (tf.Tensor): tf.Tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k).

        Returns:
            tf.Tensor: Final output tensor with shape (..., seq_len_q, depth_v).
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output

    def _split_heads(self, x, batch_size):
        """Function that splits the last dimension of a tensor and transposes it.

        Args:
            x (tf.Tensor): Input tensor with shape (batch_size,seq_len).
            batch_size (int): Size of the batches.

        Returns:
            tf.Tensor: Tensor with shape (batch_size,num_heads,seq_len,depth).
        """
        if x.shape[1] != 1:
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        else:
            x = tf.reshape(x, (batch_size, 1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
        Args:
            q (tf.Tensor): Query vector with shape (batch_size, seq_len, d_model).
            k (tf.Tensor): Keys vector with shape (batch_size, seq_len, d_model).
            v (tf.Tensor): Values vector with shape (batch_size, seq_len, d_model).
            mask (tf.Tensor): Tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k).

        Returns:
            tf.Tensor: Tensor with shape (batch_size, seq_len_q, d_model).
        """

        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self._split_heads(
            q, batch_size
        )  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split_heads(
            k, batch_size
        )  # (batch_size, num_heads, seq_len_k, depth)
        v = self._split_heads(
            v, batch_size
        )  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention = self._scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        if scaled_attention.shape[1] != 1:
            concat_attention = tf.reshape(
                scaled_attention, (batch_size, -1, self.d_model)
            )
        else:
            concat_attention = tf.reshape(
                scaled_attention, (batch_size, 1, self.d_model)
            )

        output = self.dense(concat_attention)

        return output


# Decoder
class DecoderLayer(tf.keras.layers.Layer):
    """Class that creates and computes the decoder layer.

    Args:
        d_model (int): Dimension of the model.
        num_heads (int): Number of heads of the multihead attention layer.
        dff (int): Number of neurons uses in the first layer of the
            point wise feed forward network.
        activation (tf.keras.Loss.Activation/String): Activation function for the
            point wise feed forward network. Defaults to relu.
        dropout_rate (float between 0 and 1): Fraction of the dense units to drop.
            Defaults to 0.1.
    """

    def __init__(self, d_model, num_heads, dff, activation="relu", dropout_rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        self.dense1 = tf.keras.layers.Dense(dff, activation=activation)
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, x, enc_output, training, look_ahead_mask):
        """
        Args:
            x (tf.Tensor): Input tf.Tensor
            enc_output (tf.Tensor): Output tf.Tensor of the encoder (positional encoding).
            training (bool): True if the network is in training mode, false if not.
            look_ahead_mask (tf.Tensor): tf.Tensor with shape broadcastable to
                (..., seq_len_q, seq_len_k) Look at MultiHeadAttention class for more info.

        Returns:
            tf.Tensor: Tensor with shape (batch_size, target_seq_len, d_model)
        """
        if len(x.shape) == 4:
            x = tf.squeeze(x, axis=[1])

        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2 = self.mha2(enc_output, enc_output, out1, None)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.dense1(out2)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class Decoder(tf.keras.layers.Layer):
    """Class that creates and computes the decoder.

    Args:
        num_layers (int): Number of decoder layers of the model.
        d_model (int): Dimension of the model.
        num_heads (int): Number of heads of the multihead attention layer.
        dff (int): Number of neurons uses in the first layer of the
            point wise feed forward network.
        maximum_position_encoding (int): Maximum position encoding.
        activation (tf.keras.Loss.Activation/String): Activation function for the
            point wise feed forward network. Defaults to relu.
        dropout_rate (float between 0 and 1): Fraction of the dense units to drop.
            Defaults to 0.1.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        maximum_position_encoding,
        activation="relu",
        dropout_rate=0.1,
    ):

        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.linear = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, activation, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask):
        """
        Args:
            x (tf.Tensor): Input tf.Tensor.
            enc_output (tf.Tensor): Output tf.Tensor of the encoder (positional encoding).
            training (bool): True if the network is in training mode, false if not.
            look_ahead_mask (tf.Tensor): tf.Tensor with shape broadcastable to
                (..., seq_len_q, seq_len_k) Look at MultiHeadAttention class for more info.

        Returns:
            tf.Tensor: Tensor with shape (batch_size, target_seq_len, d_model).
        """
        seq_len = x.shape[1]
        if len(tf.shape(x)) == 2:
            x = tf.expand_dims(x, -1)

        x = self.linear(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, look_ahead_mask)
        return x


# Transformer
class TransformerModel(tf.keras.Model):
    """Class that creates and computes the Transformer.

    Args:
        attribute (list): Ordered list of the indexes of the attributes that we want to predict, if the number of
            attributes of the input is different from the ones of the output.
            Defaults to None.
        num_layers (int): Number of decoder layers of the model.
        d_model (int): Dimension of the model.
        num_heads (int): Number of heads of the multihead attention layer.
        dff (int): Number of neurons uses in the first layer of the
            point wise feed forward network.
        input_size (int): Size of the input.
        target_size (int): Size of the output.
        target_shape (tuple): Shape of the output.
        pe_input (int): Maximum position encoding for the input.
        pe_target (int): Maximum position encoding for the target.
        activation (tf.keras.Loss.Activation/String): Activation function for the
            point wise feed forward network. Defaults to relu.
        dropout_rate (float between 0 and 1): Fraction of the dense units to drop.
            Defaults to 0.1.
    """

    def __init__(
        self,
        attribute,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_size,
        target_size,
        target_shape,
        pe_input,
        pe_target,
        activation="relu",
        dropout_rate=0.1,
    ):
        super().__init__()
        self.linear = tf.keras.layers.Dense(d_model)

        self.pos_encoding = positional_encoding(pe_input, d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, pe_target, activation, dropout_rate
        )

        self.final_layer = tf.keras.layers.Dense(target_shape[1])

        self.target_shape = target_shape

        self.attribute = attribute

    def _preprocess_tr_input(self, X, y):
        """Function that calculates the input of the decoder.

        Args:
            X (tf.Tensor): Input tf.Tensor.
            y (tf.Tensor): Target tf.Tensor.

        Returns:
            tf.Tensor: Tensor used as input for the decoder
                with the same shape as the target.
        """
        len_tar = y.shape[1]
        if len_tar == 1:
            return tf.gather(X, [X.shape[1] - 1], axis=1)

        tar_inp0 = tf.gather(y, [i for i in range(len_tar - 1)], axis=1)

        if len(X.shape) == 2 and len(y.shape) == 2:
            tar_inp1 = tf.gather(X, [X.shape[1] - 1], axis=1)

            tar_inp = tf.concat([tar_inp1, tar_inp0], axis=1)

        elif X.shape[-1] != y.shape[-1]:
            elements = []
            for at in self.attribute:
                elements.append(tf.gather(X, [at], axis=-1))

            tar_inp1 = tf.concat(elements, axis=-1)

            tar_inp1 = tf.gather(tar_inp1, [tar_inp1.shape[1] - 1], axis=1)
            tar_inp = tf.concat([tar_inp1, tar_inp0], axis=1)
        else:
            tar_inp1 = tf.gather(X, [X.shape[1] - 1], axis=1)
            tar_inp = tf.concat([tar_inp1, tar_inp0], axis=1)

        return tar_inp

    def _create_masks(self, inp, tar):
        """Function that creates the mask.

        Args:
            inp (tf.Tensor): Input tf.Tensor.
            tar (tf.Tensor): Target tf.Tensor.

        Returns:
            tf.Tensor: Tensor used as mask that indicates which entries should not be
                used with shape (seq_len,seq_len).
        """
        mask = 1 - tf.linalg.band_part(
            tf.ones((tf.shape(tar)[1], tf.shape(tar)[1])), -1, 0
        )
        return mask

    def call(self, inps, training):
        """

        Args:
            inps (tuple): Tuple of the input, the mask and the input of the decoder.
            training (bool): True if the network is in training mode, false if not.

        Returns:
            tf.Tensor: Output tensor.
        """
        x, combined_mask, tar_inp = inps
        seq_len = tf.shape(x)[1]

        if len(tf.shape(x)) == 2:
            x = tf.expand_dims(x, -1)

        x = self.linear(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        dec_output = self.decoder(tar_inp, x, training, combined_mask)
        final_output = self.final_layer(dec_output)
        return final_output

    def train_step(self, data):
        """Function that calculates a train step.

        Args:
            data (tuple): Tuple with the input and the target tf.Tensors.

        Returns:
            dict: Training step metrics.
        """
        inp, tar = data
        tar = tf.reshape(
            tar, (tf.shape(tar)[0], self.target_shape[0], self.target_shape[1])
        )
        tar_inp = self._preprocess_tr_input(inp, tar)
        combined_mask = self._create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions = self((inp, combined_mask, tar_inp), True)
            if len(predictions.shape) != len(tar.shape):
                predictions = predictions[:, :, 0]
            loss = self.compiled_loss(
                tar, predictions, regularization_losses=self.losses
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(tar, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Function that calculates a test step.

        Args:
            data (tuple): Tuple of the input and the target tf.Tensors.

        Returns:
            dict: Evaluation step metrics.
        """

        x, tar = data

        if len(tf.shape(x)) == 2:
            x = tf.expand_dims(x, -1)

        combined_mask = None
        tar_inp = tf.gather(x, [x.shape[1] - 1], axis=1)

        if self.attribute != None:
            tar_inp = tf.gather(x, [self.attribute], axis=-1)
            tar_inp = tf.gather(tar_inp, [tar_inp.shape[1] - 1], axis=1)

        for i in range(self.target_shape[0]):
            output = self((x, combined_mask, tar_inp), False)

            if i != self.target_shape[0] - 1:
                output = tf.gather(output, [output.shape[1] - 1], axis=1)
                tar_inp = tf.concat([tar_inp, output], axis=1)
        predictions = output

        self.compiled_loss(tar, predictions, regularization_losses=self.losses)
        self.compiled_metrics.update_state(tar, predictions)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        """Function that calculates a prediction step.

        Args:
            data (tf.Tensor): Input.

        Returns:
            tf.Tensor: Output tensor.
        """
        x = data

        combined_mask = None
        tar_inp = tf.gather(x, [x.shape[1] - 1], axis=1)

        if self.attribute != None:
            tar_inp = tf.gather(x, [self.attribute], axis=-1)
            tar_inp = tf.gather(tar_inp, [tar_inp.shape[1] - 1], axis=1)
            tar_inp = tf.squeeze(tar_inp, axis=[1])

        for i in range(self.target_shape[0]):
            output = self((x, combined_mask, tar_inp), False)

            if i != self.target_shape[0] - 1:
                output = tf.gather(output, [output.shape[1] - 1], axis=1)
                tar_inp = tf.concat([tar_inp, output], axis=1)
        output = tf.squeeze(output, axis=[2])
        return output


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def Transformer(
    input_shape,
    output_size,
    loss,
    optimizer,
    output_shape,
    attribute=None,
    num_heads=4,
    num_layers=2,
    d_model=16,
    dff=64,
    pe_input=1000,
    pe_target=1000,
    dropout_rate=0.1,
    activation="relu",
):

    """Transformer

    Args:
        input_shape (tuple): Shape of the input data.
        output_size (int): Number of neurons of the last layer.
        loss (tf.keras.Loss): Loss to be use for training.
        optimizer (tf.keras.Optimizer): Optimizer that implements the training algorithm.
          Use "custom" in order to use a customize optimizer for the transformer model.
        output_shape (tuple): Shape of the output data. Must be [forecasting_horizon,1].
        attribute (list): Ordered list of the indexes of the attributes that we want to predict, if the number of
            attributes of the input is different from the ones of the output.
            Defaults to None.
        num_heads (int): Number of heads of the attention layer.
            Defaults to 4.
        num_layers (int): Number of decoder and encoder layers. Defaults to 2.
        d_model (int): Number of neurons of the dense layer at the beginning
            of the encoder and decoder. Defaults to 16.
        dff (int): Number of neurons of the rest of dense layers in the model.
            Defaults to 64.
        pe_input (int): Maximum position encoding for the input.
            Defaults to 1000.
        pe_output (int): Maximum position encoding for the output.
            Defaults to 1000.
        dropout_rate (float between 0 and 1): Fraction of the dense units to drop.
            Defaults to 0.1.
        activation (tf.keras.Loss.Activation/String): Activation function for the
            point wise feed forward network. Defaults to "relu".

    Returns:
        tf.keras.Model: Transformer model
    """

    model = TransformerModel(
        attribute=attribute,
        input_size=input_shape[1],
        target_size=output_size,
        target_shape=output_shape,
        num_heads=num_heads,
        num_layers=num_layers,
        d_model=d_model,
        dff=dff,
        pe_input=pe_input,
        pe_target=pe_target,
        dropout_rate=dropout_rate,
        activation=activation,
    )
    if optimizer == "custom":
        learning_rate = CustomSchedule(16)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )
    att_inp = input_shape[-1]
    att_out = output_shape[-1]

    inp_len = input_shape[1]
    inp = np.arange(inp_len * att_inp).reshape((1, inp_len, att_inp))
    tar_inp = np.arange(output_size).reshape((1, output_shape[0], att_out))

    # First call to the model, in order to initialize the weights of the model, with arbitrary data.
    model.compile(optimizer=optimizer, loss=loss)
    model.call((inp, None, tar_inp), False)

    return model
