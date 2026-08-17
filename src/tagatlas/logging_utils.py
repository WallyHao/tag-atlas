"""Logging configuration helpers for TagAtlas applications."""

from __future__ import annotations

import logging

LOGGER_NAME = "tagatlas"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(
    level: int = logging.INFO,
    *,
    handler: logging.Handler | None = None,
) -> logging.Logger:
    """Configure and return the package logger.

    TagAtlas does not install output handlers automatically. Applications can
    call this helper once, or configure the ``tagatlas`` logger themselves.
    """

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if handler is None:
        handler = next(
            (
                current
                for current in logger.handlers
                if not isinstance(current, logging.NullHandler)
            ),
            None,
        )
        if handler is None:
            handler = logging.StreamHandler()
    if handler.formatter is None:
        handler.setFormatter(logging.Formatter(_FORMAT))
    if handler not in logger.handlers:
        logger.addHandler(handler)
    return logger
