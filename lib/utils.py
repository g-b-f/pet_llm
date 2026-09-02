import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from lib.types.config import LossFunctionWeights
    from lib.types.report import BrainReport

DEFAULT_LOG_LEVEL = "INFO"
MAX_LOG_SIZE_BYTES = 1024 * 1024  # 1 MB
LOG_DIR = Path(__file__).parent.parent


def namer(default_name: str) -> str:
    """By default, `RotatingFileHandler` creates logs of the form `log.txt.1`.
    This custom namer instead makes them of the form `log_1.txt`"""

    default_path = Path(default_name)
    index = default_path.suffix.strip(".")
    base_file = Path(default_path.stem)
    new_name = f"{base_file.stem}_{index}{base_file.suffix}"
    return str(default_path.parent / new_name)


def get_logger(
    name: str, level=DEFAULT_LOG_LEVEL, log_file="log.txt"
) -> logging.Logger:
    if level.upper() not in logging._nameToLevel:
        raise ValueError(f"Invalid log level: {level}")
    level_int = logging._nameToLevel[level.upper()]

    handler = RotatingFileHandler(
        LOG_DIR / log_file, maxBytes=MAX_LOG_SIZE_BYTES, backupCount=2, encoding="utf-8"
    )
    handler.setLevel(level_int)
    handler.namer = namer
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level_int)
    logger.addHandler(handler)

    return logger


def frange(
    start: float, stop: float, step: float, multiplier: int = 100
) -> Iterator[float]:
    """A floating-point range generator."""

    current = int(start * multiplier)
    stop_int = int(stop * multiplier)
    step_int = int(step * multiplier)

    while current < stop_int:
        yield current / multiplier
        current += step_int


def loss_function(report: "BrainReport", weights: "LossFunctionWeights") -> float:
    """Calculates a normalized scalar loss penalizing degenerate LLM behaviors.

    Args:
        report: Execution report emitted by the simulation run.
        thought_loop_weight: Multiplier for repeated looping states.
        empty_thought_weight: Multiplier for uninformative or empty outputs.
        out_of_bounds_weight: Multiplier for safety and constraint violations.
        malformed_json_weight: Multiplier for unparseable LLM outputs.
        inactivity_penalty: Penalty returned if no iterations were executed.

    Returns:
        The total loss scalar to be minimized by Optuna.
    """
    if report.iterations <= 0:
        raise RuntimeError("no iterations")

    weighted_error_score = (
        (report.thought_loops * weights.thought_loop)
        + (report.empty_thoughts * weights.empty_thought)
        + (report.out_of_bounds_attempts * weights.out_of_bounds)
        + (report.non_alphanumeric * weights.invalid_chars)
        + (report.malformed_json * weights.malformed_json)
    )

    error_rate = weighted_error_score / float(report.iterations)
    absolute_error_term = weighted_error_score / 100.0

    return float(error_rate + absolute_error_term)
