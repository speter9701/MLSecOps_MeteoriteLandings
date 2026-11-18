import pytest

from meteorite.registry import should_promote


@pytest.mark.parametrize(
    "new,old,margin,expected",
    [
        (1.0, None, 0.0, True),
        (0.9, 1.0, 0.0, True),
        (1.0, 1.0, 0.0, True),
        (1.1, 1.0, 0.0, False),
        (0.99, 1.0, 0.05, False),
        (0.90, 1.0, 0.05, True),
    ],
)
def test_should_promote(new: float, old: float | None, margin: float, expected: bool) -> None:
    assert should_promote(new, old, margin) is expected
