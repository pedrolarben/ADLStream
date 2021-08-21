from abc import ABC, abstractmethod
from ADLStream.data.stream import BaseStream


class FileStream(BaseStream, ABC):
    """Abstract File Stream.

    This class implements the logic to read lines from a file.

    Every class inheriting `FileStream` must have the properties below and implement
    `decode` function with the signature `message = decode(line)`, where `line` is a
    string read from the file and `message` must be of type list.

    Examples:
    ```
        class CSVFileStream(FileStream):

            def __init__(self, filename, stream_period=0, timeout=30000, **kwargs):
                super().__init__(filename, stream_period, timeout, **kwargs)

            def decode(self, line):
                return line.split(",")
    ```

    Arguments:
        filename (str): Path of file to read.
        skip_first (int >=0, optional): Number of lines to skip at the begining.
            Defaults to 0.
        stream_period (int >=0, optional): Stream time period in milliseconds.
            It is the minimun time between consecutive messages in the stream. If it
            is 0, when a message is required (`next`), it is sent as soon as possible.
            Defaults to 100.
        timeout (int >0, optional): Stream time out in milliseconds.
            It is the maximun time to wait for the next message. If it takes longer,
            `StopIteration` exception is raised.
            Defaults to 30000.
    """

    def __init__(
        self, filename, skip_first=0, stream_period=100, timeout=30000, **kwargs
    ):
        super().__init__(stream_period=stream_period, timeout=timeout, **kwargs)
        self.filename = filename
        self.skip_first = skip_first

    def start(self):
        self.file = open(self.filename, "r")
        for _ in range(self.skip_first):
            next(self.file)
        super().start()

    def stop(self):
        self.file.close()
        super().stop()

    @abstractmethod
    def decode(self, line):
        """Transform file line into a message

        Args:
            line (str): Line read from the file.

        Returns:
            list: represents the data decoded from the file line.
        """
        raise NotImplementedError

    def get_message(self):
        line = next(self.file)
        line = line[:-1]  # Delete newlines character
        message = self.decode(line)
        return message
