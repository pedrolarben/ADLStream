"""Tests for MeanNormalization scaler preprocessor."""

import pytest

from ADLStream.data.stream import DataStream
from ADLStream.data.preprocessing import MeanNormalizationScaler
from ADLStream.utils import test_utils


def _test_mean_normalization_scaler(scaler, data, expected):
    stream = DataStream(data)
    generator = test_utils.SimpleTestGenerator(stream, preprocessing_steps=[scaler])
    context = test_utils.FakeContext()

    generator.run(context)
    out = context.X

    assert out == expected


def test_one_variable():
    data = [[-100], [100], [0], [50], [-20], [30], [10]]
    expected = [[0], [0.5], [0], [0.1875], [-0.13], [0.1], [0.0]]

    scaler = MeanNormalizationScaler()

    _test_mean_normalization_scaler(scaler, data, expected)


def test_multivariable_variable():
    data = [[-100, 20], [100, 10], [0, 30], [50, 15], [-20, 0], [30, 15], [10, 50]]
    expected = [
        [0, 0],
        [0.5, -0.5],
        [0, 0.5],
        [0.1875, -0.1875],
        [-0.13, -0.5],
        [0.1, 0.0],
        [0.0, 0.6],
    ]

    scaler = MeanNormalizationScaler()

    _test_mean_normalization_scaler(scaler, data, expected)


def test_share_params():
    data = [[-100, 20], [100, 10], [20, -20], [50, 15], [-50, 0], [25, 50], [-75, 25]]
    expected = [
        [-0.5, 0.5],
        [0.4625, 0.0125],
        [0.075, -0.125],
        [0.190625, 0.015625],
        [-0.2725, -0.0225],
        [0.075, 0.2],
        [-0.4, 0.1],
    ]

    scaler = MeanNormalizationScaler(share_params=True)

    _test_mean_normalization_scaler(scaler, data, expected)


def test_div_zero():
    scaler = MeanNormalizationScaler()
    scaler.learn_one([0])
    scaler.transform_one([0])


def test_assert_not_initialized():
    scaler = MeanNormalizationScaler()
    with pytest.raises(AssertionError):
        scaler.transform_one([1])


def test_assert_inconsistent_data():
    scaler = MeanNormalizationScaler()
    with pytest.raises(AssertionError):
        scaler.learn_one([1])
        scaler.learn_one([1, 2])
