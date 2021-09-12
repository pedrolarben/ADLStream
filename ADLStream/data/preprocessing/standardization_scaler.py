"""Implements a mean normalization scaler"""
from ADLStream.data.preprocessing import BasePreprocessor
import math


class StandardizationScaler(BasePreprocessor):
    """The transformation is given by
        x_scaled = (x - avg_x) / x_stdev

    where avg_x is the mean seen until now for the feature x and x_stdev
    is the standard deviation seen until now for teh feature x.

    Arguments:
        share_params (bool): Whether to share scaler parameters among columns.
            Defaults to False.

    """

    def __init__(self, share_params=False):
        self.share_params = share_params
        self.data = None
        self.data_sum = None
        self.data_stdev_sum = None
        self.data_avg = None
        self.data_stdev = None
        self.data_count = 1

    def _mean(self, a):

        if self.share_params == False and self.data_count != 1:
            assert len(a) == len(self.data_sum)
            self.data_sum = [self.data_sum[i] + a[i] for i in range(len(a))]
            mean = [(self.data_sum[i]) / self.data_count for i in range(len(a))]

        elif self.share_params == False and self.data_count == 1:
            assert len(a) == len(self.data_sum)
            mean = [(self.data_sum[i]) / self.data_count for i in range(len(a))]

        elif self.share_params == True and self.data_count != 1:
            self.data_sum += sum(a)
            mean = [self.data_sum / (self.data_count * len(a))] * len(a)

        elif self.share_params == True and self.data_count == 1:
            mean = [self.data_sum / (self.data_count * len(a))] * len(a)
        return mean

    def _standard_deviation(self, a):
        if self.share_params == False:
            assert len(a) == len(self.data_sum)
        mean = self.data_avg
        if self.share_params == False and self.data_count != 1:
            self.data.append(a)
            self.data_stdev_sum = [
                sum([(self.data[i][j] - mean[j]) ** 2 for i in range(len(self.data))])
                for j in range(len(mean))
            ]
            stdev = [
                math.sqrt(1 / self.data_count * (self.data_stdev_sum[i]))
                for i in range(len(self.data_stdev_sum))
            ]

        elif self.share_params == False and self.data_count == 1:
            self.data_stdev_sum = [(a[i] - mean[i]) ** 2 for i in range(len(a))]
            stdev = [
                math.sqrt(1 / self.data_count * (self.data_stdev_sum[i]))
                for i in range(len(self.data_stdev_sum))
            ]

        elif self.share_params == True and self.data_count != 1:
            self.data.append(a)
            self.data_stdev_sum = [
                sum(
                    [
                        (self.data[i][j] - mean[j]) ** 2
                        for i in range(len(self.data))
                        for j in range(len(mean))
                    ]
                )
            ] * len(a)
            stdev = [
                math.sqrt(1 / (self.data_count * len(a)) * (self.data_stdev_sum[i]))
                for i in range(len(self.data_stdev_sum))
            ]

        elif self.share_params == True and self.data_count == 1:
            self.data_stdev_sum = [
                sum([(a[i] - mean[i]) ** 2 for i in range(len(a))])
            ] * len(a)
            stdev = [
                math.sqrt(1 / (self.data_count * len(a)) * (self.data_stdev_sum[i]))
                for i in range(len(self.data_stdev_sum))
            ]
        return stdev

    def learn_one(self, x):
        """Updates `avg` and `count` parameters for each feature

        Args:
            x (list): input data from stream generator.

        Returns:
            BasePreprocessor: self updated scaler.
        """
        if self.data_sum is None:
            self.data = [x]
            self.data_avg = x
            self.data_mean = x
            self.data_sum = x
            self.data_stdev_sum = x
            if self.share_params == True:
                self.data_sum = sum(x)
        self.data_avg = self._mean(x)
        self.data_stdev = self._standard_deviation(x)

        self.data_count += 1
        return self

    def _standardization(self, val, avg_val, std_val):
        def _safe_div_zero(a, b):
            return 0 if b == 0 else a / b

        return _safe_div_zero((val - avg_val), std_val)

    def transform_one(self, x):
        """Scales one instance data

        Args:
            x (list): input data from stream generator.

        Returns:
            scaled_x (list): minmax scaled data.
        """
        assert self.data_sum is not None
        scaled_x = [
            self._standardization(v, a, s)
            for v, a, s in zip(x, self.data_avg, self.data_stdev)
        ]
        return scaled_x
