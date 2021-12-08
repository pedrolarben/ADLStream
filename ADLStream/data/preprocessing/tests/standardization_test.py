"""Tests for mean standardization scaler preprocessor."""

import pytest

from ADLStream.data.stream import DataStream
from ADLStream.data.preprocessing import StandardizationScaler
from ADLStream.utils import test_utils


def _test_mean_normalization_scaler(scaler, data, expected):
    stream = DataStream(data)
    generator = test_utils.SimpleTestGenerator(stream, preprocessing_steps=[scaler])
    context = test_utils.FakeContext()

    generator.run(context)
    out = context.X

    assert out == expected


def test_one_variable():
    data = [[-100], [100], [0], [20], [-20], [15], [-15]]
    expected = [
        [0],
        [0.7071067811865475],
        [0.0],
        [0.18234920215644926],
        [-0.2773500981126146],
        [0.19293661761242642],
        [-0.2520504151250418],
    ]

    scaler = StandardizationScaler()

    _test_mean_normalization_scaler(scaler, data, expected)


def test_multivariable_variable():
    data = [[-100, 20], [100, 10], [0, 30], [20, 15], [-20, 0], [15, 15], [-15, 50]]
    expected = [
        [0, 0],
        [0.7071067811865475, -0.7071067811865475],
        [0.0, 1.0],
        [0.18234920215644926, -0.4391550328268399],
        [-0.2773500981126146, -1.3416407864998738],
        [0.19293661761242642, 0.0],
        [-0.2520504151250418, 1.8665130505147653],
    ]

    scaler = StandardizationScaler()

    _test_mean_normalization_scaler(scaler, data, expected)


def test_share_params():
    data = [[-100, 20], [100, 10], [20, -20], [50, 15], [-50, 0], [25, 50], [-75, 25]]
    expected = [
        [-0.7071067811865475, 0.7071067811865475],
        [1.1251798047845263, 0.030410264994176386],
        [0.23063280200722128, -0.38438800334536877],
        [0.6666083751413966, 0.0546400307492948],
        [-1.007753209141362, -0.08320898057130512],
        [0.29494868637148924, 0.7865298303239713],
        [-1.5298253763543845, 0.38245634408859613],
    ]

    scaler = StandardizationScaler(share_params=True)

    _test_mean_normalization_scaler(scaler, data, expected)


def test_div_zero():
    scaler = StandardizationScaler()
    scaler.learn_one([0])
    scaler.transform_one([0])


def test_assert_not_initialized():
    scaler = StandardizationScaler()
    with pytest.raises(AssertionError):
        scaler.transform_one([1])


def test_assert_inconsistent_data():
    scaler = StandardizationScaler()
    with pytest.raises(AssertionError):
        scaler.learn_one([1])
        scaler.learn_one([1, 2])
