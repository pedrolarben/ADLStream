"""Implements a moving window stream generator for time series forecasting."""
from ADLStream.data import BaseStreamGenerator


class MovingWindowStreamGenerator(BaseStreamGenerator):
    """Moving window stream generator.

    This class performs a moving-window preprocessing method for time series forecasting
    problems.

    TODO: Implement variable selection for x and y.

    Arguments:
        stream (inherits ADLStream.data.stream.BaseStream): 
            Stream source to be feed to the ADLStream framework.
        past_history (int): The width (number of time steps) of the input window (`x`).
        forecasting_horizon (int): 
            The width (number of time steps) of the label window (`y`).
        shift (int >=1, optional): The time offset between input and label windows.
            Defaults to 1     
    """

    def __init__(self, stream, past_history, forecasting_horizon, shift=1, **kwargs):
        super().__init__(stream, **kwargs)
        self.past_history = past_history
        self.forecasting_horizon = forecasting_horizon
        self.shift = shift

        self.x_window = []
        self.y_window = []

    def preprocess(self, message):
        x, y = None, None
        self.x_window.append(message)
        if self.num_messages >= self.past_history + self.shift:
            self.y_window.append(message)

        if len(self.x_window) == self.past_history:
            x = self.x_window
            self.x_window.pop(0)

        if len(self.y_window) == self.forecasting_horizon:
            y = self.y_window
            self.y_window.pop(0)

        return x, y
