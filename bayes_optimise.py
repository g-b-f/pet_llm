from pathlib import Path

import optuna
from optuna.storages.journal import JournalFileBackend, JournalStorage
from optuna.storages.journal._file import BaseJournalFileLock
from lib.brain import Brain
from lib.extra_types import SimulationConfig
from lib.tank import Tank
from lib.utils import get_logger, loss_function
from models.download import Model, get_model

model_path = get_model(Model.smollm2)
logger = get_logger(__name__, "debug", log_file="log_bayes.txt")

RUNTIME = 250
N_TRIALS = 50
N_SEEDS = 3

storage_backend = Path(__file__).parent /"study_backend.jsonl"

class DummyLock(BaseJournalFileLock):
    """It's a single threaded process why tf are you making me use a lock"""
    def acquire(self):
        return True
    def release(self):
        pass

def evaluate_simulation(trial: optuna.Trial) -> float:
    """Evaluates simulation loss for a set of sampled brain parameters.

    Args:
        trial: An Optuna trial instance used to sample hyperparameters.

    Returns:
        The scalar loss output calculated from the simulation result.
    """
    temperature = trial.suggest_float("temperature", 1.5, 2.5)
    frequency_penalty = trial.suggest_float("frequency_penalty", 1.5, 2.5)
    presence_penalty = trial.suggest_float("presence_penalty", 1.5, 2.5)
    repeat_penalty = trial.suggest_float("repeat_penalty", 1.5, 2.5)


    config = SimulationConfig.model_construct()
    config.tank.runtime = RUNTIME

    config.brain.params.temperature = temperature
    config.brain.params.frequency_penalty = frequency_penalty
    config.brain.params.presence_penalty = presence_penalty
    config.brain.params.repeat_penalty = repeat_penalty

    losses:list[float] = []

    for seed in range(N_SEEDS):
        brain = Brain(model_path, config.brain)
        tank = Tank(brain, config.tank)

        # make headless:
        # tank._render_scene = lambda: None  # type: ignore[attr-defined, method-assign]

        logger.info(
            f"{seed=}, {temperature=}, {frequency_penalty=}, "
            f"{presence_penalty=}, {repeat_penalty=}"
        )
        result = tank.run()
        loss = loss_function(result.report)
        logger.info(f"{loss=}")
        losses.append(loss)

    return sum(losses) / len(losses)


if __name__ == "__main__":
    logger.info(f"starting optimisation")
    logger.info(f"{RUNTIME=}, {N_TRIALS=}, {N_SEEDS=}")
    eta = RUNTIME * N_TRIALS * N_SEEDS
    logger.info(f"eta: {eta/(60*60):.2f} hours")

    storage = JournalStorage(
        JournalFileBackend(
            str(storage_backend.resolve()),
            lock_obj=DummyLock()
            )
        )

    optimization_study = optuna.create_study(
        study_name=f"pet_llm_{model_path.stem}",
        storage=storage,
        direction="minimize",
        load_if_exists=True
        )
    optimization_study.optimize(evaluate_simulation, n_trials=N_TRIALS)

    logger.info(f"Best parameters: {optimization_study.best_params}")
    logger.info(f"Best loss: {optimization_study.best_value}")