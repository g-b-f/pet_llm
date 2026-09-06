import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class _NoopFileHandler(logging.Handler):
    """Stand-in for ``RotatingFileHandler`` that never opens a file.

    Exposes ``baseFilename`` so tests that inspect the configured log path
    keep working, while performing no I/O.
    """

    def __init__(self, filename, *args, **kwargs):
        super().__init__()
        self.baseFilename = os.path.abspath(str(filename))
        self.stream = None

    def emit(self, record):
        pass


def _strip_file_handlers() -> None:
    """Remove and close every file-based (or no-op) handler on any logger."""
    loggers = [logging.root, *logging.root.manager.loggerDict.values()]
    for logger in loggers:
        if isinstance(logger, logging.Logger):
            for handler in list(logger.handlers):
                if isinstance(handler, (logging.FileHandler, _NoopFileHandler)):
                    logger.removeHandler(handler)
                    handler.close()


@pytest.fixture(autouse=True)
def no_log_file(caplog):
    """Ensure tests never write to the log file; capture records via ``caplog``.

    ``get_logger`` attaches a ``RotatingFileHandler`` (opened at import time and
    on every call) that would otherwise append to the project-root log file.
    Strip existing file handlers and disable file-handler creation for the
    duration of each test so ``caplog`` is the only sink.
    """
    _strip_file_handlers()
    with patch("lib.utils.RotatingFileHandler", _NoopFileHandler):
        yield caplog
    _strip_file_handlers()
