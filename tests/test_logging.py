import io
import logging

from tagatlas import configure_logging


def test_configure_logging_formats_messages_and_reuses_handler() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)

    logger = configure_logging(logging.DEBUG, handler=handler)
    logger.info("test event")
    handler.flush()

    assert "INFO tagatlas: test event" in stream.getvalue()
    assert configure_logging(logging.WARNING, handler=handler) is logger
    assert logger.level == logging.WARNING

    logger.removeHandler(handler)
    logger.propagate = True


def test_configure_logging_creates_default_handler() -> None:
    logger = logging.getLogger("tagatlas")
    existing = set(logger.handlers)

    configure_logging()

    added = [handler for handler in logger.handlers if handler not in existing]
    assert len(added) == 1
    logger.removeHandler(added[0])
    logger.propagate = True
