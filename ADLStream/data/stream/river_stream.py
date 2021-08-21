"""Implements a wraper for river datasets."""

from ADLStream.data.stream import BaseStream


class RiverStream(BaseStream):
    """River stream wrapper

    This class acts as a wrapper for river streams, allowing to use ADLStream
    logic with streams from river framework. This is extremely usefull in order
    to use the synthetic stream generators implemented in river.

    For more information about river framework, visit its [docs](https://github.com/online-ml/river).

    Examples:
    ```python
        import ADLStream
        rom river import synth

        dataset = synth.Agrawal(classification_function=0, seed=42)
        stream = ADLStream.data.RiverStream(dataset)
        generator = ADLStream.data.ClassificationStreamGenerator(stream)

    ```

    Arguments:
        dataset (inherits from river Dataset class): original stream from river
            framework.
        stream_period (int >=0, optional): Stream time period in milliseconds.
            It is the minimun time between consecutive messages in the stream. If it
            is 0, when a message is required (`next`), it is sent as soon as possible.
            Defaults to 0.
        timeout (int >0, optional): Stream time out in milliseconds.
            It is the maximun time to wait for the next message. If it takes longer,
            `StopIteration` exception is raised.
            Defaults to 30000.
    """

    def __init__(
        self, dataset, stream_period=0, timeout=30000, n_instances=1000, **kwargs
    ):
        super().__init__(stream_period=stream_period, timeout=timeout, **kwargs)
        self.dataset = dataset
        self.n_instances = n_instances
        self.data = None

    def start(self):
        self.data = self.dataset.take(self.n_instances)
        super().start()

    def get_message(self):
        # Get X and y from river stream
        X, y = next(self.data)
        # Convert X from dict format to list
        X = list(X.values())
        # Return X and y concatenated as a single list
        return X + [y]
