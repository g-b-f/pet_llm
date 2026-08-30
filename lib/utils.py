import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator
from lib.extra_types import BrainReport

DEFAULT_LOG_LEVEL = "INFO"
MAX_LOG_SIZE_BYTES = 1024 * 1024 # 1 MB
LOG_DIR = Path(__file__).parent.parent

def namer(default_name: str) -> str:
    """By default, `RotatingFileHandler` creates logs of the form `log.txt.1`.
    This custom namer instead makes them of the form `log_1.txt`"""

    default_path = Path(default_name)
    index = default_path.suffix.strip(".")
    base_file = Path(default_path.stem)
    new_name = f"{base_file.stem}_{index}{base_file.suffix}"
    return str(default_path.parent / new_name)

def get_logger(name: str, level=DEFAULT_LOG_LEVEL) -> logging.Logger:
    if level.upper() not in logging._nameToLevel:
        raise ValueError(f"Invalid log level: {level}")
    level_int = logging._nameToLevel[level.upper()]

    handler = RotatingFileHandler(LOG_DIR / "log.txt", maxBytes=MAX_LOG_SIZE_BYTES, backupCount=2)
    handler.setLevel(level_int)
    handler.namer = namer
    formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level_int)
    logger.addHandler(handler)

    return logger

def frange(start:float, stop:float, step:float, multiplier:int=100) -> Iterator[float]:
    """A floating-point range generator."""

    current = int(start * multiplier)
    stop_int = int(stop * multiplier)
    step_int = int(step * multiplier)

    while current < stop_int:
        yield current / multiplier
        current += step_int

# def loss_function(report: BrainReport) -> float:
#     total = report.iterations
#     empty = report.empty_thoughts
#     loop = report.thought_loops
#     oob = report.out_of_bounds_attempts

#     return (empty*


def loss_function(
    report: BrainReport,
    thought_loop_weight: float = 10.0,
    empty_thought_weight: float = 5.0,
    out_of_bounds_weight: float = 20.0,
    inactivity_penalty: float = 1000.0,
) -> float:
    """Calculates a normalized scalar loss penalizing degenerate LLM behaviors.

    Args:
        report: Execution report emitted by the simulation run.
        thought_loop_weight: Multiplier for repeated looping states.
        empty_thought_weight: Multiplier for uninformative or empty outputs.
        out_of_bounds_weight: Multiplier for safety and constraint violations.
        inactivity_penalty: Penalty returned if no iterations were executed.

    Returns:
        The total loss scalar to be minimized by Optuna.
    """
    if report.iterations <= 0:
        return inactivity_penalty

    weighted_error_score = (
        (report.thought_loops * thought_loop_weight)
        + (report.empty_thoughts * empty_thought_weight)
        + (report.out_of_bounds_attempts * out_of_bounds_weight)
    )

    error_rate = weighted_error_score / float(report.iterations)
    absolute_error_term = weighted_error_score / 100.0

    return float(error_rate + absolute_error_term)