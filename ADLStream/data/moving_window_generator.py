"""Implements a moving window stream generator for time series forecasting."""
from typing import List, Optional, Tuple, Union
from ADLStream.data import BaseStreamGenerator


class MovingWindowStreamGenerator(BaseStreamGenerator):
    """Moving window stream generator.

    This class performs a moving-window preprocessing method for time series forecasting
    problems.

    The main logic can be found in the `prepocess(self, message)` function.

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
        stream: "BaseStream",
        past_history: int,
        forecasting_horizon: int,
        shift: int = 1,
        input_idx: Optional[Union[int, list]] = None,
        target_idx: Optional[Union[int, list]] = None,
        **kwargs
    ) -> None:
        super().__init__(stream, **kwargs)
        self.past_history = past_history
        self.forecasting_horizon = forecasting_horizon
        self.shift = shift
        self.input_idx = input_idx
        self.target_idx = target_idx

        self.x_window = []
        self.y_window = []

    def _select_features(
        self, message: List[float], idx: Optional[Union[int, List[int]]]
    ) -> List[float]:
        """Gets features indicated by `idx` from the `message`.

        Args:
            message (List[float]): stream message.
            idx (Optional[Union[int,List[int]]]): features indices.

        Returns:
            List[float]: selected features from the stream message.
        """
        res = None
        if isinstance(idx, int):
            res = [message[idx]]
        elif isinstance(idx, list):
            res = [message[i] for i in idx]
        else:
            res = message
        return res

    def _get_x(self, message: List[float]) -> List[float]:
        """Get input features from the message.

        Args:
            message (List[float]): stream message.

        Returns:
            List[float]: Input instance (`x`).
        """
        return self._select_features(message, self.input_idx)

    def _get_y(self, message: List[float]) -> List[float]:
        """Get output features from the message.

        Args:
            message (List[float]): stream message.

        Returns:
            List[float]: Expected output (`y`).
        """
        return self._select_features(message, self.target_idx)

    def preprocess(self, message: List[float]) -> Tuple[List[float], List[float]]:
        """Apply the moving window to the stream data for time series forecasting
        problems.

        Args:
            message (List[float]): stream message.

        Returns:
            Tuple[Optional[List[float]],Optional[List[float]]]: (`x`, `y`) input and output instances.
        """
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
