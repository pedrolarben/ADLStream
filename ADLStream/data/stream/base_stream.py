"""Implements an abstract object representing a Stream."""

from abc import ABC, abstractmethod
import time


class BaseStream(ABC):
    """Abstract Base Stream.

    This is the base class for implementing streams from custom sources.

    Every `Stream` must have the properties below and implement `get_message`, which
    should return a stream message as an array of any dimensions. Other methods such as
    `start` or `stop` may be overrided if additional logic is needed.

    Examples:
    ```python
        class RandomStream(BaseStream):
            def __init__(self, seed=1, **kwargs):
                self.seed = seed
                self.super().__init__(**kwargs)

            def start(self):
                random.seed(self.seed)
                self.super().start()

            def get_message(self):
                message = [random.random()]
                return message

        stream = RandomStream()
        stream.start()
        message = stream.next()

    ```

    Arguments:
        stream_period (int >=0, optional): Stream time period in milliseconds.
            It is the minimun time between consecutive messages in the stream. If it
            is 0, when a message is required (`next`), it is sent as soon as possible.
            Defaults to 0.
        timeout (int >0, optional): Stream time out in milliseconds.
            It is the maximun time to wait for the next message. If it takes longer,
            `StopIteration` exception is raised.
            Defaults to 30000.
    """

    def __init__(self, stream_period=0, timeout=30000):
        self.stream_period = stream_period
        self.timeout = timeout
        self.last_message_time = None

    def start(self):
        """Function to be called before asking any message."""
        self.last_message_time = time.time()

    def stop(self):
        """Function to be called when stream is finished."""
        pass

    @abstractmethod
    def get_message(self):
        """The function that contains the logic to generate a new message.
        It must return the message as an array.
        This function must be override by every custom stream.

        Raises:
            NotImplementedError: Abstract function has not been overrided.

        Returns:
            list: message
        """
        raise NotImplementedError("Abstract method")

    def next(self):
        """The function that returns the next stream message.

        Raises:
            StopIteration: The stream has finisshed

        Returns:
            list: message
        """
        if self.stream_period > 0:
            if time.time() - self.last_message_time < self.stream_period / 1000:
                time.sleep(
                    self.stream_period / 1000
                    - (
                        (time.time() - self.last_message_time)
                        % (self.stream_period / 1000)
                    )
                )
        starttime = time.time()

        try:
            message = self.get_message()
        except Exception:
            self.stop()
            raise StopIteration

        self.last_message_time = starttime

        return message
