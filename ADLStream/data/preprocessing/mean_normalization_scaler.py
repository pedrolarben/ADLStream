"""Implements a mean normalization scaler"""
from ADLStream.data.preprocessing import BasePreprocessor


class MeanNormalizationScaler(BasePreprocessor):
    """The transformation is given by
        x_scaled = (x - avg_x) / (max_x - min_x)

    where min_x is the minimun value seen until now for the feature x,
    max_x represents the maximun value seen until now for the feature x
    and avg_x is the mean seen until now for the feature x.

    Arguments:
        share_params (bool): Whether to share scaler parameters among columns.
            Defaults to False.

    """

    def __init__(self, share_params=False):
        self.share_params = share_params
        self.data_min = None
        self.data_max = None
        self.data_sum = None
        self.data_count = 1
        self.data_avg = None

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

    def _mean(self, a):
        if self.share_params == False:
            assert len(a) == len(self.data_sum)
            self.data_sum = [self.data_sum[i] + a[i] for i in range(len(a))]
            mean = [(self.data_sum[i]) / self.data_count for i in range(len(a))]

        else:
            self.data_sum += sum(a)
            mean = [self.data_sum / (self.data_count * len(a))] * len(a)

        return mean

    def learn_one(self, x):
        """Updates `min` `max` `avg` and `count` parameters for each feature

        Args:
            x (list): input data from stream generator.

        Returns:
            BasePreprocessor: self updated scaler.
        """
        if self.data_min is None:
            self.data_min = x
            self.data_max = x
            self.data_avg = x
            self.data_sum = [0.0] * len(x)
            if self.share_params == True:
                self.data_sum = 0.0
        self.data_min = self._minimum(x, self.data_min)
        self.data_max = self._maximum(x, self.data_max)
        self.data_avg = self._mean(x)
        self.data_count += 1
        return self

    def _mean_normalization(self, val, min_val, max_val, avg_val):
        def _safe_div_zero(a, b):
            return 0 if b == 0 else a / b

        return _safe_div_zero((val - avg_val), (max_val - min_val))

    def transform_one(self, x):
        """Scales one instance data

        Args:
            x (list): input data from stream generator.

        Returns:
            scaled_x (list): minmax scaled data.
        """
        assert self.data_min is not None
        scaled_x = [
            self._mean_normalization(v, m, M, a)
            for v, m, M, a in zip(x, self.data_min, self.data_max, self.data_avg)
        ]
        return scaled_x
