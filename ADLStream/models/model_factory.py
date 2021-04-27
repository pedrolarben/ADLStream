"""Deep learning model factory."""

from ADLStream.models import MLP
from ADLStream.models import LSTM
from ADLStream.models import GRU
from ADLStream.models import ERNN
from ADLStream.models import ESN
from ADLStream.models import CNN
from ADLStream.models import TCN
from ADLStream.models import Transformer

MODEL_FACTORY = {
    "MLP": MLP,
    "LSTM": LSTM,
    "GRU": GRU,
    "ERNN": ERNN,
    "ESN": ESN,
    "CNN": CNN,
    "TCN": TCN,
    "TRANSFORMER": Transformer,
}


def create_model(model_architecture, input_shape, output_size, loss, optimizer, **args):
    """Creates a deep learning model.

    Args:
        model_architecture (str): Model architecture to implemet.
        input_shape (tuple): Shape of the input data
        output_size (int): Number of neurons of the last layer.
        loss (tf.keras.Loss): Loss to be use for training.
        optimizer (tf.keras.Optimizer): Optimizer that implements theraining algorithm.
        **args: specific model parameters.

    Returns:
        tf.keras.Model: keras model
    """
    assert model_architecture.upper() in MODEL_FACTORY, "Model {} not supported".format(
        model_architecture
    )
    return MODEL_FACTORY[model_architecture.upper()](
        input_shape, output_size, loss, optimizer, **args
    )
