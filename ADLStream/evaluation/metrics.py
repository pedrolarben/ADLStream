"""Metrics factory."""
import numpy as np


def accuracy(y, o):
    """Accuracy score.

    Args:
        y (np.array): Real values.
        o (np.array): Predictions.

    Returns:
        float: Mean accuracy.
    """
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
    
    TODO Implement KAPPA

    Args:
        y (np.array): Real values.
        o (np.array): Predictions.

    Returns:
        float: Kappa.
    """
    raise NotImplementedError


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
