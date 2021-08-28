"""Implements a min max scaler"""
from ADLStream.data.preprocessing import BasePreprocessor


class MinMaxScaler(BasePreprocessor):
    """Transform features by scaling each feature between zero and one.

    This estimator scales and translates each feature (column) individually
    such that it is in the the range (0, 1).

    The transformation is given by
        x_scaled = (x - min_x) / (max_x - min_x)

    where min_x is the minimun value seen until now for the feature x and
    max_x represents the maximun value seen until now for the feauter x.

    Arguments:
        share_params (bool): Whether to share scaler parameters among columns.
            Defaults to False.

    """

    def __init__(self, share_params=False):
        self.share_params = share_params
        self.data_min = None
        self.data_max = None

    def _minimum(self, a, b):
        assert len(a) == len(b)
        min_values = [min(a[i], b[i]) for i in range(len(a))]
        if self.share_params:
            min_values = [min(min_values) for _ in min_values]
        return min_values

    def _maximum(self, a, b):
        assert len(a) == len(b)
        max_values = [max(a[i], b[i]) for i in range(len(a))]
        if self.share_params:
            max_values = [max(max_values) for _ in max_values]
        return max_values

    def learn_one(self, x):
        """Updates `min` and `max` parameters for each feature

        Args:
            x (list): input data from stream generator.

        Returns:
            BasePreprocessor: self updated scaler.
        """
        if self.data_min is None:
            self.data_min = x
            self.data_max = x
        self.data_min = self._minimum(x, self.data_min)
        self.data_max = self._maximum(x, self.data_max)
        return self

    def _min_max(self, val, min_val, max_val):
        def _safe_div_zero(a, b):
            return 0 if b == 0 else a / b

        return _safe_div_zero((val - min_val), (max_val - min_val))

    def transform_one(self, x):
        """Scales one instance data

        Args:
            x (list): input data from stream generator.

        Returns:
            scaled_x (list): minmax scaled data.
        """
        assert (
            self.data_min is not None
        ), "Parameters not initialized - learn_one before must be called before transform_one."
        scaled_x = [
            self._min_max(v, m, M) for v, m, M in zip(x, self.data_min, self.data_max)
        ]
        return scaled_x
