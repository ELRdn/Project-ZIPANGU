import pytest

from zipangu.budget import check_budget


def test_pass() -> None:
    assert check_budget(0.5, 160, 10, 3000).allowed


def test_block() -> None:
    assert not check_budget(1, 160, 20, 3000).allowed


def test_invalid_values() -> None:
    with pytest.raises(ValueError, match="hourly_usd"):
        check_budget(0, 160, 1, 3000)
