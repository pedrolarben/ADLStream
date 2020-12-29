"""Implements an abstract object representing a stream preprocessor."""

from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    def learn_one(self, x):
        return self

    @abstractmethod
    def transform_one(self, x):
        raise NotImplementedError
