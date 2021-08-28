from ADLStream.data.stream import FileStream


class CSVFileStream(FileStream):
    """CSV File Stream.

    This class creates a stream from a csv file.

    Arguments:
        filename (str): Path of file to read.
        sep (str): Delimiter to use.
            Defaults to ",".
        index_col (int >=0, optional): Number of columns to use as index.
            Defaults to 0.
        header (int >=0, optional): Number of rows to use as column names.
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
        self,
        filename,
        sep=",",
        index_col=0,
        header=0,
        stream_period=100,
        timeout=30000,
        **kwargs
    ):
        super().__init__(
            filename=filename,
            skip_first=header,
            stream_period=stream_period,
            timeout=timeout,
            **kwargs
        )
        self.sep = sep
        self.index_col = index_col

    def decode(self, line):
        message = line.strip().split(self.sep)[self.index_col :]
        message = [float(x.strip()) for x in message]
        return message
