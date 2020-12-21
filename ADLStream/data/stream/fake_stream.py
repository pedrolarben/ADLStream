"""Implements a fake stream to be use for code testing."""

from ADLStream.data.stream import BaseStream
import time
import numpy as np


class FakeStream(BaseStream):
    """Fake Stream.
    This stream returns a sine wave in a specific shape and length.

    Arguments:
        num_features (int >=1, optional): Number of features for each message.
            Defaults to 1.
        stream_length (int, optional): Maximun number of messages to be returned.
            Defaults to 1000.
        **kwargs: BaseStream arguments.
    """

    def __init__(
        self,
        num_features=1,
        stream_length=1000,
        stream_period=0,
        timeout=30000,
        **kwargs
    ):
        super().__init__(stream_period=stream_period, timeout=timeout, **kwargs)
        self.num_features = num_features
        self.stream_length = stream_length

        self.messages = []

    def start(self):
        super().start()
        data = list(
            np.sin([[x / 100] * self.num_features for x in range(self.stream_length)])
        )
        self.messages = data

    def get_message(self):
        message = self.messages.pop(0)
        return message
