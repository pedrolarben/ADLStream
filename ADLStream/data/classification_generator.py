from typing import List, Optional, Tuple, Type, Union
from ADLStream.data import BaseStreamGenerator
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class ClassificationStreamGenerator(BaseStreamGenerator):
    """Classification stream generator.

    This class is used for generating streams for classification problems.

    Arguments:
        stream (inherits BaseStream):
            Stream source to be feed to the ADLStream framework.
        label_index (int or list, optional): The column index/indices of the target
            label.
            Defaults to -1.
        one_hot_labels (list or None, optional): Possible label values if one-hot
            encoding must be done. If None, the target value is not one-hot encoded.
            Defaults to None.
    """

    def __init__(
        self,
        stream: Type["BaseStream"],
        label_index: Optional[Union[List[int], int]] = [-1],
        one_hot_labels: Optional[List] = None,
        **kwargs
    ) -> None:
        super().__init__(stream, **kwargs)
        self.label_index = label_index if type(label_index) is list else [label_index]
        self.labels = one_hot_labels
        self.one_hot_encoder = None
        if self.labels:
            self.one_hot_encoder = OneHotEncoder()
            self.one_hot_encoder.fit(np.asarray(self.labels).reshape(-1, 1))

    def preprocess(self, message: List) -> Tuple[List, List]:
        """Divide the message in `X` and `y`.
        It uses the `self.labels_index` features as `y` and the rest as `x`.
        Additionally, if indicated, it performs a one hot encoding to the labels.

        Args:
            message (List): stream message.

        Returns:
            Tuple[List, List]: `(x, y)` input value and its class.
        """
        x = message
        y = [message.pop(i) for i in self.label_index]

        if self.labels:
            y = self.one_hot_encoder.transform([y]).toarray()
            y = list(y[0])

        return x, y
