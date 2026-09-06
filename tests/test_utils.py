import logging
from pathlib import Path

import pytest

from lib.utils import get_logger, loss_function, namer
from lib.types.config import LossFunctionWeights
from lib.types.report import BrainReport


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

    @pytest.mark.skip
    def test_handlers_cleared_on_recall(self):
        logger1 = get_logger("test_recall")
        handler_count = len(logger1.handlers)
        logger2 = get_logger("test_recall")
        assert len(logger2.handlers) == handler_count

    @pytest.mark.skip
    def test_child_loggers_use_parent_file(self, tmp_path: Path):
        tmp_log1 = tmp_path/ "log1.txt"
        tmp_log2 = tmp_path/ "log2.txt"
        tmp_log3 = tmp_path/ "log3.txt"

        logger_child1 = get_logger("lib.some_name1", log_file=tmp_log1)
        logger_parent = get_logger("__main__", log_file = tmp_log2)
        logger_child2 = get_logger("lib.some_name2", log_file = tmp_log3)
        
        # TODO: complete this
        raise


class TestLossFunction:
    @pytest.fixture
    def weights(self) -> LossFunctionWeights:
        return LossFunctionWeights(
            thought_loop=10.0,
            empty_thought=10.0,
            out_of_bounds=10.0,
            malformed_json=100.0,
            invalid_chars=10.0,
            similar_messages=10.0,
        )

    def test_no_iterations_raises(self, weights: LossFunctionWeights):
        report = BrainReport(iterations=0)
        with pytest.raises(RuntimeError, match="no iterations"):
            loss_function(report, weights)

    def test_zero_errors_zero_loss(self, weights: LossFunctionWeights):
        report = BrainReport(iterations=10)
        assert loss_function(report, weights) == 0.0

    def test_similar_messages_penalized(self, weights: LossFunctionWeights):
        base = BrainReport(iterations=10)
        with_similar = BrainReport(iterations=10, similar_messages=2)
        assert loss_function(with_similar, weights) > loss_function(base, weights)

    def test_similar_messages_weighted_correctly(self, weights: LossFunctionWeights):
        # 2 similar messages * weight 10 = 20 weighted score
        # error_rate = 20 / 10 = 2.0, absolute = 20 / 100 = 0.2 -> 2.2
        report = BrainReport(iterations=10, similar_messages=2)
        assert loss_function(report, weights) == pytest.approx(2.2)

