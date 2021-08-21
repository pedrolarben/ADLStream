"""Implements a moving window stream generator for time series forecasting."""
from ADLStream.data import BaseStreamGenerator


class MovingWindowStreamGenerator(BaseStreamGenerator):
    """Moving window stream generator.

    This class performs a moving-window preprocessing method for time series forecasting
    problems.

    Arguments:
        stream (inherits ADLStream.data.stream.BaseStream):
            Stream source to be feed to the ADLStream framework.
        past_history (int): The width (number of time steps) of the input window (`x`).
        forecasting_horizon (int):
            The width (number of time steps) of the label window (`y`).
        shift (int >=1, optional): The time offset between input and label windows.
            Defaults to 1.
        input_idx (int or list, optional): The index/indices of the input feature/s.
            If None, every feature is considered as input feature. Defaults to None.
        target_idx (int or list, optional): The index/indices of the target feature/s.
            If None, every feature is considered as target feature. Defaults to None.
    """

    def __init__(
        self,
        stream,
        past_history,
        forecasting_horizon,
        shift=1,
        input_idx=None,
        target_idx=None,
        **kwargs
    ):
        super().__init__(stream, **kwargs)
        self.past_history = past_history
        self.forecasting_horizon = forecasting_horizon
        self.shift = shift
        self.input_idx = input_idx
        self.target_idx = target_idx

        self.x_window = []
        self.y_window = []

    def _select_features(self, message, idx):
        res = None
        if isinstance(idx, int):
            res = [message[idx]]
        elif isinstance(idx, list):
            res = [message[i] for i in idx]
        else:
            res = message
        return res

    def _get_x(self, message):
        return self._select_features(message, self.input_idx)

    def _get_y(self, message):
        return self._select_features(message, self.target_idx)

    def preprocess(self, message):
        x, y = None, None
        self.x_window.append(self._get_x(message))
        if self.num_messages >= self.past_history + self.shift:
            self.y_window.append(self._get_y(message))

        if len(self.x_window) > self.past_history:
            self.x_window.pop(0)
            x = self.x_window

        if len(self.y_window) > self.forecasting_horizon:
            self.y_window.pop(0)
            y = self.y_window

        return x, y
