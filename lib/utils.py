import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator

from lib.types.config import SimulationConfig

DEFAULT_LOG_LEVEL = "INFO"
MAX_LOG_SIZE_BYTES = 1024 * 1024 * 10 # 10 MB
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
    name: str, level=DEFAULT_LOG_LEVEL, log_file: str | Path = "log.txt"
) -> logging.Logger:
    if level.upper() not in logging._nameToLevel:
        raise ValueError(f"Invalid log level: {level}")
    level_int = logging._nameToLevel[level.upper()]

    if isinstance(log_file, str):
        log_file = LOG_DIR / log_file

    handler = RotatingFileHandler(
        log_file, maxBytes=MAX_LOG_SIZE_BYTES, backupCount=2, encoding="utf-8"
    )
    handler.setLevel(level_int)
    handler.namer = namer
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    if name == "__main__":
        for logger_name, logger_obj in logging.root.manager.loggerDict.items():
            if logger_name.startswith("lib") and isinstance(logger_obj, logging.Logger):
                logger_obj.handlers = [handler]

    logger = logging.getLogger(name)
    # Clear existing handlers so repeated calls don't stack them
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    logger.setLevel(level_int)
    logger.addHandler(handler)

    return logger


def frange(start: float, stop: float, step: float, multiplier=100) -> Iterator[float]:
    """Floating-point range generator"""
    current = start * multiplier
    while current < stop * multiplier:
        yield current / multiplier
        current += step * multiplier

def values_from_trial(
    trial_id: int,
    config=SimulationConfig.model_construct(),
    fpath=Path(__file__).parents[1] / "study_backend.jsonl"
) -> SimulationConfig:
    with open(fpath) as f:
        for line in f.readlines():
            d = json.loads(line)

            if d.get("trial_id") == trial_id:
                if d.get("param_name") == "temperature":
                    config.brain.params.temperature = d["param_value_internal"]
                if d.get("param_name") == "frequency_penalty":
                    config.brain.params.frequency_penalty = d["param_value_internal"]
                if d.get("param_name") == "presence_penalty":
                    config.brain.params.presence_penalty = d["param_value_internal"]
                if d.get("param_name") == "repeat_penalty":
                    config.brain.params.repeat_penalty = d["param_value_internal"]

    print(config.brain.params.model_dump_json(indent=2))
    return config
