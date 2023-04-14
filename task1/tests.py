import typing as tp

import pytest

from main import multiplicate


@pytest.mark.parametrize(
    'A, A_transformed', [
        ([8], [8]),
        ([0], [0]),
        ([-2, 3], [3, -2]),
        ([0, 0], [0, 0]),
        ([1, 2, 3], [6, 3, 2]),
        ([1, 0, 1, 2, 3], [0, 6, 0, 0, 0]),
        ([1, 0, 1, 2, 0], [0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0]),
        ([1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1]),
        ([1, 2, 3, 4, 5], [120, 60, 40, 30, 24]),
    ],
)
def test_correctness(A: tp.List[int], A_transformed: tp.List):
    assert A_transformed == multiplicate(A)
