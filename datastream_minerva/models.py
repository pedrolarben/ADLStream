from tensorflow.keras import layers, Input, Model
from tcn import TCN


def create_cnn_model(num_features, num_classes, loss_func, optimizer="adam",  conv_layers=[64, 128],
                     kernel_sizes=[7, 5], dense_layers=[64, 32],  dropout=0.2):

    inp = Input(shape=(num_features, 1), name='input')
    c = layers.Conv1D(conv_layers[0], kernel_sizes[0], padding='same', activation='relu')(inp)
    c = layers.MaxPool1D(pool_size=2)(c)
    for conv_channels, kernel_size in zip(conv_layers[1:], kernel_sizes[1:]):
        c = layers.Conv1D(conv_channels, kernel_size, padding='same', activation='relu')(c)
        c = layers.MaxPool1D(pool_size=2)(c)
    c = layers.Flatten()(c)
    for hidden_units in dense_layers:
        c = layers.Dense(hidden_units)(c)
        if dropout:
            c = layers.Dropout(dropout)(c)
    c = layers.Dense(num_classes, activation="softmax", name="prediction")(c)
    model = Model(inp, c)

    model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
    return model


def create_mlp_model(num_features, num_classes, loss_func, optimizer='adam', dense_layers=[32, 64, 128], dropout=0.2):

    inp = Input(shape=(num_features, 1), name='input')
    c = layers.Flatten()(inp)  # Convert the 2d input in a 1d array
    for hidden_units in dense_layers:
        c = layers.Dense(hidden_units)(c)
        if dropout:
            c = layers.Dropout(dropout)(c)
    c = layers.Dense(num_classes, activation="softmax", name="prediction")(c)

    model = Model(inputs=inp, outputs=c)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

    return model


def create_lstm_model(num_features, num_classes, loss_func, optimizer='adam', lstm_layers=[64, 128],
                      dense_layers=[64, 32], dropout=0.2):

    inp = Input(shape=(num_features, 1), name='input')
    return_sequences = len(lstm_layers) > 1
    c = layers.LSTM(lstm_layers[0], return_sequences=return_sequences, dropout=dropout)(inp)
    for i in range(len(lstm_layers[1:])):
        return_sequences = i < len(lstm_layers[1:]) - 1
        c = layers.LSTM(lstm_layers[i], return_sequences=return_sequences, dropout=dropout)(c)
    for hidden_units in dense_layers:
        c = layers.Dense(hidden_units)(c)
        if dropout:
            c = layers.Dropout(dropout)(c)
    c = layers.Dense(num_classes)(c)

    model = Model(inputs=inp, outputs=c)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

    return model


def create_tcn_model(num_features, num_classes, loss_func, optimizer='adam', nb_filters=64, kernel_size=2,
                     nb_stacks=1, dilations=[1, 2, 4], dropout_rate=0.2, use_skip_connections=True,
                     use_batch_norm=False, activation='linear'):

    inp = Input(shape=(num_features, 1), name='input')

    c = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations,
            use_skip_connections=use_skip_connections, dropout_rate=dropout_rate, activation=activation,
            use_batch_norm=use_batch_norm)(inp)
    c = layers.Dense(num_classes)(c)

    model = Model(inputs=inp, outputs=c)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

    return model



MODELS = {
    'cnn': create_cnn_model,
    'lstm': create_lstm_model,
    'mlp': create_mlp_model,
    'tcn': create_tcn_model
}


def create_model(model_name, num_features, num_classes, loss_func):
    return MODELS[model_name](num_features, num_classes, loss_func)