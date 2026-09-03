import logging
from pathlib import Path

import pytest

from lib.utils import get_logger, namer


class TestNamer:
    def test_default_rotation_name(self):
        result = namer("log.txt.1")
        assert result == str(Path("log_1.txt"))

    def test_preserves_directory(self):
        result = namer("/some/dir/log.txt.2")
        assert result == str(Path("/some/dir/log_2.txt"))


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_default_level_is_info(self):
        logger = get_logger("test_default_level")
        assert logger.level == logging.INFO

    def test_explicit_level(self):
        logger = get_logger("test_debug_level", "DEBUG")
        assert logger.level == logging.DEBUG

    def test_case_insensitive_level(self):
        logger = get_logger("test_case_level", "debug")
        assert logger.level == logging.DEBUG

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            get_logger("test_bad_level", "NOTALEVEL")

    def test_handlers_cleared_on_recall(self):
        logger1 = get_logger("test_recall")
        handler_count = len(logger1.handlers)
        logger2 = get_logger("test_recall")
        assert len(logger2.handlers) == handler_count

    def test_child_loggers_use_parent_file(self, tmp_path: Path):
        tmp_log1 = tmp_path/ "log1.txt"
        tmp_log2 = tmp_path/ "log2.txt"
        tmp_log3 = tmp_path/ "log3.txt"

        logger_child1 = get_logger("lib.some_name1", log_file=tmp_log1)
        logger_parent = get_logger("__main__", log_file = tmp_log2)
        logger_child2 = get_logger("lib.some_name2", log_file = tmp_log3)
        
        # TODO: complete this
        raise

