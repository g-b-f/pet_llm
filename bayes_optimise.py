import optuna
from lib.brain import Brain
from lib.extra_types import SimulationConfig
from lib.tank import Tank
from lib.utils import get_logger, loss_function
from models.download import Model, get_model

model_path = get_model(Model.smollm2)
logger = get_logger(__name__)

RUNTIME = 300


def evaluate_simulation(trial: optuna.Trial, seeds=1) -> float:
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

    for seed in range(seeds):
        brain = Brain(model_path, config.brain)
        tank = Tank(brain, config.tank)
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
    optimization_study = optuna.create_study(direction="minimize")
    optimization_study.optimize(evaluate_simulation, n_trials=20)

    logger.info(f"Best parameters: {optimization_study.best_params}")
    logger.info(f"Best loss: {optimization_study.best_value}")