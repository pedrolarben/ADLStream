"""Implements an abstract object representing a Stream Generator."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Type


class BaseStreamGenerator(ABC):
    """Abstract Base Stream Generator

    This is the base class for implementing stream generators with custom behavior.

    Every `StreamGenerator` must have the properties below and implement `preprocess` with
    the signature `(x, y) = preprocess(message)`.

    Examples:
    ```python
        class MinimalBaseGenerator(BaseStreamGenerator):

            def __init__(self, stream, class_index=-1, **kwargs):
                self.class_index = class_index
                super().__init__(stream, **kwargs)

            def preprocess(message):
                x = message
                y = x.pop(self.class_index)
                return x, y
    ```

    Arguments:
        stream (inherits [BaseStream]):
            Stream source to be feed to the ADLStream framework.
        preprocessing_steps (list of BasePreprocessor):
            List of operations to be perform sequentially over the input data (x).
            Defaults to [].
        max_instances (int > 0): Max number of instance to generate.  If None, it will
            continue generating new message until the stream stops.
            Defaults to None.
    """

    def __init__(
        self,
        stream: Type["BaseStream"],
        preprocessing_steps: List[Type["BasePreprocessor"]] = [],
        max_instances: int = None,
    ) -> None:
        self.stream = stream
        self.num_messages = 0
        self.preprocessing_steps = preprocessing_steps
        self.max_messages = max_instances

    @property
    def num_messages(self) -> int:
        """Return number of messages processed from the stream.

        Returns:
            int: number of messages.
        """
        return self._num_messages

    @num_messages.setter
    def num_messages(self, value: int) -> None:
        self._num_messages = value

    def _check_number_instances(self):
        """Stop when it reaches the maximum number of messages.

        Raises:
            StopIteration:
        """
        if self.max_messages is not None:
            if self.num_messages >= self.max_messages:
                self.stream.stop()
                raise StopIteration

    def next(self, context: "ADLStreamContext") -> List[float]:
        """Get the next message from the stream.

        Args:
            context (ADLStreamContext): ADLStream shared object. Used for logging.

        Returns:
            List[float]: new message.
        """
        message = None
        try:
            self._check_number_instances()
            message = self.stream.next()
            self.num_messages += 1
        except StopIteration:
            context.log("INFO", "GENERATOR-PROCESS - Stream has finished")
            context.set_time_out()
        except Exception as e:
            context.log(
                "ERROR",
                "GENERATOR-PROCESS - Error getting messages from stream {}".format(
                    str(e)
                ),
            )
            context.set_time_out()
        return message

    @abstractmethod
    def preprocess(self, message: List[float]) -> Tuple[List[float], List[float]]:
        """The function that contains the logic to transform a stream message into
        model imput and target data `(x ,y)`.

        Both output, `x` or `y`, can be `None` what means it should not be added to
        the context.

        The target data `y` can be delayed. Although we are sending `x` and `y` at
        the same time, it does not mean that `y` is the corresponding target value
        of `x`. However, input data and target data should be in order: `y_i` is the
        target value of `x_i`. So the first target data sent (`y_0`) corresponds with
        the first input sent (`x_0`).

        Args:
            message (list): message received from the stream

        Raises:
            NotImplementedError: This is an abstract method which should be implemented.

        Returns:
            x (list): instance of model's input data.
            y (list): instance of model's target data.
        """
        raise NotImplementedError("Abstract method")

    def _perform_preprocessing_steps(self, x: List[float]) -> List[float]:
        """Apply the different preprocessors to a message.

        Args:
            x (List[float]): message.

        Returns:
            List[float]: updated message.
        """
        for preprocessor in self.preprocessing_steps:
            x = preprocessor.learn_one(x).transform_one(x)
        return x

    def run(self, context: "ADLStreamContext"):
        """The function that sends data to ADLStream framework

        It gets messages from the stream, preprocesses them and sends to the specific
        ADLStream context.

        Args:
            context (ADLStream.ADLStreamContext): context where to send the stream data
        """
        self.stream.start()
        message = self.next(context)
        while message is not None:
            (x, y) = self.preprocess(message)
            if x is not None or y is not None:
                x = self._perform_preprocessing_steps(x)
                context.add(x, y)
            message = self.next(context)
