"""Tests for MinMax scaler preprocessor."""

import pytest

from ADLStream.data.stream import DataStream
from ADLStream.utils import test_utils


def _test_moving_window(
    data,
    expected_X,
    expected_y,
    past_history=4,
    forecasting_horizont=2,
    shift=1,
    input_idx=None,
    target_idx=None,
):
    stream = DataStream(data)
    generator = ADLStream.data.MovingWindowStreamGenerator(
        stream=stream,
        past_history=past_history,
        forecasting_horizon=forecasting_horizont,
        shift=shift,
        input_idx=0,
        target_idx=0,
    )
    context = test_utils.FakeContext()

    generator.run(context)

    X = context.X
    y = context.y

    assert X == expected_X and y == expected_y


def test_one_variable():
    data = [x for x in range(10)]
    expected_X = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]
    expected_y = [[5, 6], [6, 7], [7, 8], [8, 9]]

    _test_moving_window(data, expected_X, expected_y)


def test_multivariable_variable():
    data = [[x, x] for x in range(10)]
    expected_X = [
        [[1, 1], [2, 2], [3, 3], [4, 4]],
        [[2, 2], [3, 3], [4, 4], [5, 5]],
        [[3, 3], [4, 4], [5, 5], [6, 6]],
        [[4, 4], [5, 5], [6, 6], [7, 7]],
        [[5, 5], [6, 6], [7, 7], [8, 8]],
        [[6, 6], [7, 7], [8, 8], [9, 9]],
    ]
    expected_y = [
        [[5, 5], [6, 6]],
        [[6, 6], [7, 7]],
        [[7, 7], [8, 8]],
        [[8, 8], [9, 9]],
    ]

    _test_moving_window(data, expected_X, expected_y)


def test_multivariable_variable_one_input():
    data = [[x, x + 1] for x in range(10)]
    expected_X = [
        [[1], [2], [3], [4]],
        [[2], [3], [4], [5]],
        [[3], [4], [5], [6]],
        [[4], [5], [6], [7]],
        [[5], [6], [7], [8]],
        [[6], [7], [8], [9]],
    ]
    expected_y = [
        [[5, 6], [6, 7]],
        [[6, 7], [7, 8]],
        [[7, 8], [8, 9]],
        [[8, 9], [9, 10]],
    ]
    _test_moving_window(data, expected_X, expected_y, input_idx=0)


def test_multivariable_variable_one_output():
    data = [[x, x + 1] for x in range(10)]
    expected_X = [
        [[1, 2], [2, 3], [3, 4], [4, 5]],
        [[2, 3], [3, 4], [4, 5], [5, 6]],
        [[3, 4], [4, 5], [5, 6], [6, 7]],
        [[4, 5], [5, 6], [6, 7], [7, 8]],
        [[5, 6], [6, 7], [7, 8], [8, 9]],
        [[6, 7], [7, 8], [8, 9], [9, 10]],
    ]
    expected_y = [[[5], [6]], [[6], [7]], [[7], [8]], [[8], [9]]]
    _test_moving_window(data, expected_X, expected_y, target_idx=0)


def test_shift_one_variable():
    data = [[x, x] for x in range(10)]
    expected_X = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]
    expected_y = [[7, 8], [8, 9]]
    _test_moving_window(data, expected_X, expected_y, shift=3)


def test_shift_multivariable():
    data = [[x, x + 1] for x in range(10)]
    expected_X = [
        [[1, 2], [2, 3], [3, 4], [4, 5]],
        [[2, 3], [3, 4], [4, 5], [5, 6]],
        [[3, 4], [4, 5], [5, 6], [6, 7]],
        [[4, 5], [5, 6], [6, 7], [7, 8]],
        [[5, 6], [6, 7], [7, 8], [8, 9]],
        [[6, 7], [7, 8], [8, 9], [9, 10]],
    ]
    expected_y = [[[7, 8], [8, 9]], [[8, 9], [9, 10]]]
    _test_moving_window(data, expected_X, expected_y, shift=3)


def test_shift_multivariable_one_output():
    data = [[x, x + 1] for x in range(10)]
    expected_X = [
        [[1, 2], [2, 3], [3, 4], [4, 5]],
        [[2, 3], [3, 4], [4, 5], [5, 6]],
        [[3, 4], [4, 5], [5, 6], [6, 7]],
        [[4, 5], [5, 6], [6, 7], [7, 8]],
        [[5, 6], [6, 7], [7, 8], [8, 9]],
        [[6, 7], [7, 8], [8, 9], [9, 10]],
    ]
    expected_y = [[[7], [8]], [[8], [9]]]
    _test_moving_window(data, expected_X, expected_y, shift=3, target_idx=0)
