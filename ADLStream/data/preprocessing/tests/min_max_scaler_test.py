"""Tests for MinMax scaler preprocessor."""

import pytest

from ADLStream.data.stream import DataStream
from ADLStream.data.preprocessing import MinMaxScaler
from ADLStream.utils import test_utils


def _test_min_max_scaler(scaler, data, expected):
    stream = DataStream(data)
    generator = test_utils.SimpleTestGenerator(stream, preprocessing_steps=[scaler])
    context = test_utils.FakeContext()

    generator.run(context)
    out = context.X

    assert out == expected


def test_one_variable():
    data = [[-100], [100], [0], [50], [-50], [25], [-75]]
    expected = [[0], [1.0], [0.5], [0.75], [0.25], [0.625], [0.125]]

    scaler = MinMaxScaler()

    _test_min_max_scaler(scaler, data, expected)


def test_multivariable_variable():
    data = [[-100, 20], [100, 10], [0, 20], [50, 15], [-50, 0], [25, 50], [-75, 25]]
    expected = [
        [0, 0],
        [1.0, 0.0],
        [0.5, 1.0],
        [0.75, 0.5],
        [0.25, 0.0],
        [0.625, 1.0],
        [0.125, 0.5],
    ]

    scaler = MinMaxScaler()

    _test_min_max_scaler(scaler, data, expected)


def test_share_params():
    data = [[-100, 20], [100, 10], [0, 20], [50, 15], [-50, 0], [25, 50], [-75, 25]]
    expected = [
        [0.0, 1.0],
        [1.0, 0.55],
        [0.5, 0.6],
        [0.75, 0.575],
        [0.25, 0.5],
        [0.625, 0.75],
        [0.125, 0.625],
    ]

    scaler = MinMaxScaler(share_params=True)

    _test_min_max_scaler(scaler, data, expected)


def test_div_zero():
    scaler = MinMaxScaler()
    scaler.learn_one([0])
    scaler.transform_one([0])


def test_assert_not_initialized():
    scaler = MinMaxScaler()
    with pytest.raises(AssertionError):
        scaler.transform_one([1])


def test_assert_inconsistent_data():
    scaler = MinMaxScaler()
    with pytest.raises(AssertionError):
        scaler.learn_one([1])
        scaler.learn_one([1, 2])
