from pathlib import Path

import optuna
from optuna.storages.journal import JournalFileBackend, JournalStorage, JournalFileOpenLock
from optuna.storages.journal._file import BaseJournalFileLock

from lib.types.config import ParamsConfig, TunerConfig
from lib.types.report import BrainReport, StudyReport, Trial
from lib.utils import get_logger

logger = get_logger(__file__)


class DummyLock(BaseJournalFileLock):
    """It's a single threaded process why tf are you making me use a lock"""

    def acquire(self):
        return True

    def release(self):
        pass

storage_backend = Path(__file__).parent.parent /"study_backend.jsonl"
def get_storage(n_jobs:int) -> JournalStorage:
    string_backend = str(storage_backend.resolve())
    if n_jobs == 1:
        lock_obj: BaseJournalFileLock = DummyLock()
    elif n_jobs > 1:
        lock_obj: BaseJournalFileLock = JournalFileOpenLock(string_backend)
    else:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
    
    return JournalStorage(
        JournalFileBackend(
            string_backend,
            lock_obj=lock_obj
            )
        )

def suggest_vals(trial: optuna.Trial, tuner_config: TunerConfig, params_config = ParamsConfig.model_construct()) -> ParamsConfig:
    params_config.temperature = trial.suggest_float("temperature", *tuner_config.temperature)
    params_config.frequency_penalty = trial.suggest_float("frequency_penalty", *tuner_config.frequency_penalty)
    params_config.presence_penalty = trial.suggest_float("presence_penalty", *tuner_config.presence_penalty)
    params_config.repeat_penalty = trial.suggest_float("repeat_penalty", *tuner_config.repeat_penalty)

    return params_config


def append_report(report_path: Path, report: BrainReport, params: ParamsConfig):
    study_report = StudyReport.model_validate_json(report_path.read_text())
    study_report.trials.append(Trial(params=params, report=report))
    report_path.write_text(study_report.model_dump_json(indent=2))
