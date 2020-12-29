"""Implements a min max scaler"""
from ADLStream.data.preprocessing import BasePreprocessor


class MinMaxScaler(BasePreprocessor):
    """Scales the data between 0 and 1.

    The transformation is given by 
        x_scaled = (x - min_x) / (max_x - min_x)

    where min_x is the minimun value seen until now for the feature x and 
    max_x represents the maximun value seen until now for the feauter x.

    """

    def __init__(self):
        self.data_min = None
        self.data_max = None

    def _minimum(self, a, b):
        assert len(a) == len(b)
        min_values = [min(a[i], b[i]) for i in range(len(a))]
        return min_values

    def _maximum(self, a, b):
        assert len(a) == len(b)
        max_values = [max(a[i], b[i]) for i in range(len(a))]
        return max_values

    def learn_one(self, x):
        if self.data_min is None:
            self.data_min = x
            self.data_max = x
        else:
            self.data_min = self._minimum(x, self.data_min)
            self.data_max = self._maximum(x, self.data_max)
        return self

    def _min_max(self, val, min_val, max_val):
        def _safe_div_zero(a, b):
            return 0 if b == 0 else a / b

        return _safe_div_zero((val - min_val), (max_val - min_val))

    def transform_one(self, x):
        assert self.data_min is not None
        scaled_x = [
            self._min_max(v, m, M) for v, m, M in zip(x, self.data_min, self.data_max)
        ]
        return scaled_x
