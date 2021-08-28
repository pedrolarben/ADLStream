"""Implements a stream fed from a list."""

from ADLStream.data.stream import BaseStream


class DataStream(BaseStream):
    """Data stream.

    This class implements a stream given a list of messages.


    Arguments:
        messages (iterable): List with the stream messages.
        stream_period (int >=0, optional): Stream time period in milliseconds.
            It is the minimun time between consecutive messages in the stream. If it
            is 0, when a message is required (`next`), it is sent as soon as possible.
            Defaults to 0.
        timeout (int >0, optional): Stream time out in milliseconds.
            It is the maximun time to wait for the next message. If it takes longer,
            `StopIteration` exception is raised.
            Defaults to 30000.
    """

    def __init__(self, messages, stream_period=0, timeout=30000, **kwargs):
        super().__init__(stream_period=stream_period, timeout=timeout, **kwargs)
        self.messages = messages.copy()
        self.iterator = None

    def start(self):
        self.iterator = iter(self.messages)
        return super().start()

    def get_message(self):
        return next(self.iterator)
