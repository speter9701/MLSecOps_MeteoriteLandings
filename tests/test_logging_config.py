import logging
from collections.abc import Iterator

import pytest

from meteorite.logging_config import HANDLER_NAME, configure_logging


@pytest.fixture(autouse=True)
def restore_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    yield
    configure_logging("INFO")


def test_default_level_is_info() -> None:
    assert configure_logging().level == logging.INFO


def test_level_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert configure_logging().level == logging.DEBUG


def test_explicit_level_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert configure_logging("warning").level == logging.WARNING


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="LOG_LEVEL 'LOUD'"):
        configure_logging("loud")


def test_reconfiguring_keeps_a_single_handler() -> None:
    configure_logging()
    logger = configure_logging()
    # pytest attaches its own capture handlers to non-propagating loggers, so count only ours
    assert [handler.get_name() for handler in logger.handlers].count(HANDLER_NAME) == 1
    assert logger.propagate is False


def test_lines_are_key_value_formatted(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    logging.getLogger("meteorite.api.routes").info("predict request_id=%s", "abc")
    logging.getLogger("meteorite.api.routes").debug("not shown at INFO")
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("level=INFO logger=meteorite.api.routes predict request_id=abc")
