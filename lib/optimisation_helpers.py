import json
from pathlib import Path

import optuna
from optuna.storages.journal import JournalFileBackend, JournalStorage
from optuna.storages.journal._file import BaseJournalFileLock
from lib.extra_types import BrainReport, ParamsConfig, StudyReport, TunerConfig
from lib.utils import get_logger

logger = get_logger(__file__)

class DummyLock(BaseJournalFileLock):
    """It's a single threaded process why tf are you making me use a lock"""
    def acquire(self):
        return True
    def release(self):
        pass

storage_backend = Path(__file__).parent /"study_backend.jsonl"
storage = JournalStorage(
        JournalFileBackend(
            str(storage_backend.resolve()),
            lock_obj=DummyLock()
            )
        )

def suggest_vals(trial: optuna.Trial, tuner_config: TunerConfig, params_config = ParamsConfig.model_construct()) -> ParamsConfig:
    params_config.temperature = trial.suggest_float("temperature", *tuner_config.temperature)
    params_config.frequency_penalty = trial.suggest_float("frequency_penalty", *tuner_config.frequency_penalty)
    params_config.presence_penalty = trial.suggest_float("presence_penalty", *tuner_config.presence_penalty)
    params_config.repeat_penalty = trial.suggest_float("repeat_penalty", *tuner_config.repeat_penalty)

    return params_config  


def append_report(report_path: Path, report: BrainReport):
    with open(report_path, "r") as f:
        study_report = StudyReport(**json.load(f))

    study_report.reports.append(report)
    report_path.write_text(study_report.model_dump_json(indent=2))