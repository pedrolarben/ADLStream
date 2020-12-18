"""Implements a wraper for scikit-multiflow streams."""

from ADLStream.data.stream import BaseStream


class SkmultiflowStream(BaseStream):
    """Scikit-multiflow stream wrapper

    This class acts as a wrapper for scikit-multiflow streams, allowing to use ADLStream
    logic with streams fron scikit-multiflow. This is extremely usefull in order to use 
    the stream generators implemented in scikit-multiflow.
    
    For more information about scikit-multiflow visit scikit-multiflow docs:
        https://scikit-a.readthedocs.io

    Examples:
    ```python
        import ADLStream
        from skmultiflow.data import AGRAWALGenerator

        stream = ADLStream.data.SkmultiflowStream(AGRAWALGenerator())
        generator = ADLStream.data.ClassificationStreamGenerator(stream)

    ```

    Arguments:
        stream (inherits skmultiflow.data.base_stream.Stream): original stream
            from scikit-multiflow.
        stream_period (int >=0, optional): Stream time period in milliseconds. 
            It is the minimun time between consecutive messages in the stream. If it 
            is 0, when a message is required (`next`), it is sent as soon as possible.
            Defaults to 0.
        timeout (int >0, optional): Stream time out in milliseconds.
            It is the maximun time to wait for the next message. If it takes longer,
            `StopIteration` exception is raised.
            Defaults to 30000.
    """

    def __init__(self, stream, stream_period=0, timeout=30000, **kwargs):
        super(stream_period=stream_period, timeout=timeout, **kwargs)
        self.stream = stream

    def start(self):
        super().start()
        if self.stream.is_restartable():
            self.stream.restart()

    def get_message(self):
        if not self.stream.has_more_samples():
            raise StopIteration
        # Get X and y from skmultiflow stream
        X, y = self.stream.next_sample(batch_size=1)
        # Convert numpy output into lists and remove batch dimension
        X = (X.tolist()[0],)
        y = y.tolist() if len(y.shape) < 2 else y.tolist()[0]
        # Return X and y concatenated as a single list
        return X + y
