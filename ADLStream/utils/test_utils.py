"""Utilities for testing ADLStream."""


from ADLStream.data import BaseStreamGenerator


class FakeContext:
    """Fake ADLStream context for testing purposes."""

    def __init__(self):
        self.X = []
        self.y = []

    def set_time_out(self):
        pass

    def log(self, l, s):
        pass

    def add(self, x, y):
        if x is not None:
            self.X.append(x)
        if y is not None:
            self.y.append(y)


class SimpleTestGenerator(BaseStreamGenerator):
    """Simple ADLStream generator for testing purposes."""

    def preprocess(self, message):
        x = message
        return x, x
