"""Implements an abstract object representing a stream preprocessor."""

from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """Abstract Base Preprocessor

    This is the base class for implementing a preprocessor object.

    Every `Preprocessor` have two main function:
    `learn_one` and `transform_one`.

    Every `Preprocessor` must implement `transform_one` and, if needed, `learn_one` with
    the signatures `x' = transform_one(x)` and `updated_preprocessor = learn_one(x)`.

    Examples:
    ```python
        class Squared(BasePreprocessor):

            def transform_one(self, x):
                ans = [e*e for e in x]
                return ans
    ```
    """

    def learn_one(self, x):
        """Updates inner parameters if needed.

        Args:
            x (list): input data from stream generator.

        Returns:
            BasePreprocessor: self updated object.
        """
        return self

    @abstractmethod
    def transform_one(self, x):
        """Transforms one instance data

        Args:
            x (list): input data from stream generator.

        Returns:
            x' (list): transformed input data.

        Raises:
            NotImplementedError: Child class must implement this funtion.
        """
        raise NotImplementedError
