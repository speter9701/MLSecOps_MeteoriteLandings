"""Logging setup of the API process."""
import logging
import os
from typing import Final

LOG_FORMAT: Final[str] = "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"
HANDLER_NAME: Final[str] = "meteorite"


def configure_logging(level: str | None = None) -> logging.Logger:
    """Send the ``meteorite`` loggers to stderr in a key=value format, at ``LOG_LEVEL`` (default INFO).

    Calling it again replaces the handler instead of adding a second one.
    """
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = logging.getLevelNamesMapping().get(level_name)
    if numeric_level is None:
        raise ValueError(f"Unknown LOG_LEVEL {level_name!r}, use DEBUG, INFO, WARNING, ERROR or CRITICAL")
    handler = logging.StreamHandler()
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger = logging.getLogger("meteorite")
    for old in [h for h in logger.handlers if h.get_name() == HANDLER_NAME]:
        logger.removeHandler(old)
    logger.addHandler(handler)
    logger.setLevel(numeric_level)
    # no duplicate lines if the host process also puts a handler on the root logger
    logger.propagate = False
    return logger
