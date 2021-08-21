"""Metrics factory."""
import numpy as np


def _confusion_matrix(y, o):
    y, o = _preprocess_predictions(y, o)
    labels = {l: idx for idx, l in enumerate(np.unique([y, o]).tolist())}
    matrix = np.zeros((len(labels), len(labels)))
    for yi, oi in zip(y, o):
        i, j = labels[yi], labels[oi]
        matrix[i][j] += 1
    return matrix


def _preprocess_predictions(y, o):
    if len(y.shape) > 1 and y.shape[1] > 1:
        # y and o are one-hot encoded
        y = np.argmax(y, axis=1)
        o = np.argmax(o, axis=1)
    else:
        # y and o are binary probabilities [0, 1]
        y = y.flatten()
        o = np.array(o.flatten() >= 0.5, dtype=y.dtype)
    return y, o


def accuracy(y, o):
    """Accuracy score.

    Args:
        y (np.array): Real values.
        o (np.array): Predictions.

    Returns:
        float: Mean accuracy.
    """
    y, o = _preprocess_predictions(y, o)
    return np.mean(y == o)


def mae(y, o):
    """Mean absolute error (MAE).

    Args:
        y (np.array): Real values.
        o (np.array): Predictions.

    Returns:
        float: MAE.
    """
    return np.mean(np.abs(o - y))


def kappa(y, o):
    """Cohen kappa score.

    Args:
        y (np.array): Real values.
        o (np.array): Predictions.

    Returns:
        float: Kappa.
    """
    confusion = _confusion_matrix(y, o)
    n_classes = confusion.shape[0]
    if n_classes < 2:
        return 1
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    w_mat = np.ones([n_classes, n_classes], dtype=np.int)
    w_mat.flat[:: n_classes + 1] = 0

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k


def auc(y, o):
    """Area under curve (AUC).

    TODO Implement AUC

    Args:
        y (np.array): Real values.
        o (np.array): Predictions.

    Returns:
        float: AUC.
    """
    raise NotImplementedError


def mape(y, o):
    """Mean Absolute Percentage Error (MAPE).

    TODO Implement MAPE

    Args:
        y (np.array): Real values.
        o (np.array): Predictions.

    Returns:
        float: MAPE.
    """
    raise NotImplementedError


def wape(y, o):
    """Weighted Absolute Percentage Error (WAPE).

    TODO Implement WAPE

    Args:
        y (np.array): Real values.
        o (np.array): Predictions.

    Returns:
        float: WAPE.
    """
    raise NotImplementedError


METRICS = {
    "ACCURACY": accuracy,
    "MAE": mae,
    "KAPPA": kappa,
    "AUC": auc,
    "MAPE": mape,
    "WAPE": wape,
}


def evaluate(metric, y, o, **kwargs):
    """Compute a specific loss function given expected output and predicted output.

    Args:
        metric (str): Loss function to use.
        y (list or np.array): Target data.
        o (list or np.array): Predictions.

    Returns:
        float: loss
    """
    assert metric.upper() in METRICS, "Metric {} not supported.".format(metric)
    o = np.asarray(o) if type(o) == type([]) else o
    y = np.asarray(y).reshape(o.shape) if type(y) == type([]) else y
    return METRICS[metric.upper()](y, o, **kwargs)
