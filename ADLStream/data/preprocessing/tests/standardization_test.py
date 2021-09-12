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
        [1.0],
        [0],
        [0.21055872190307892],
        [-0.31008683647302115],
        [0.21135147527014894],
        [-0.27224556389190907],
    ]

    scaler = StandardizationScaler()

    _test_mean_normalization_scaler(scaler, data, expected)


def test_multivariable_variable():
    data = [[-100, 20], [100, 10], [0, 30], [20, 15], [-20, 0], [15, 15], [-15, 50]]
    expected = [
        [0, 0],
        [1.0, -1.0],
        [0, 1.2247448713915892],
        [0.21055872190307892, -0.50709255283711],
        [-0.31008683647302115, -1.5],
        [0.21135147527014894, 0.0],
        [-0.27224556389190907, 2.0160645150967413],
    ]

    scaler = StandardizationScaler()

    _test_mean_normalization_scaler(scaler, data, expected)


def test_share_params():
    data = [[-100, 20], [100, 10], [20, -20], [50, 15], [-50, 0], [25, 50], [-75, 25]]
    expected = [
        [-1.0, 1.0],
        [1.2992457263581536, 0.03511474936103118],
        [0.2526455763199557, -0.42107596053325946],
        [0.7126343288380518, 0.05841264990475834],
        [-1.0622651534102405, -0.08770996679534096],
        [0.30806385570456685, 0.8215036152121782],
        [-1.5875748207668994, 0.39689370519172484],
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
