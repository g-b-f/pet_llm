from pathlib import Path

import optuna

from lib.brain import Brain
from lib.extra_types import (
    LossFunctionWeights,
    SimulationConfig,
    StudyReport,
    TunerConfig,
)
from lib.optimisation_helpers import append_report, storage, suggest_vals
from lib.tank import Tank
from lib.utils import get_logger, loss_function
from models.download import Model, get_model

RUNTIME = 300
N_TRIALS = 25
N_SEEDS = 3
VERSION = 3

logger = get_logger(__name__, "debug", log_file="log_bayes.txt")
model_path = get_model(Model.smollm2)

study_name = f"v{VERSION}_pet_llm_{model_path.stem}"
report_path = Path(__file__).parent / f"reports/{study_name}.json"

loss_function_weights = LossFunctionWeights(
    thought_loop=10.0,
    empty_thought=10.0,
    out_of_bounds=10.0,
    malformed_json=100.0,
    invalid_chars=10.0,
    inactivity_penalty=1000.0,
)

tuner_config = TunerConfig(
    temperature=(1.5, 2.5),
    frequency_penalty=(1.5, 2.5),
    presence_penalty=(1.5, 2.5),
    repeat_penalty=(1.5, 2.5),
)


# TODO: move into `optimisation_helpers`
def evaluate_simulation(trial: optuna.Trial) -> float:
    """Evaluates simulation loss for a set of sampled brain parameters.

    Args:
        trial: An Optuna trial instance used to sample hyperparameters.

    Returns:
        The scalar loss output calculated from the simulation result.
    """
    config = SimulationConfig.model_construct()
    config.tank.runtime = RUNTIME
    config.brain.params = suggest_vals(trial, tuner_config, config.brain.params)

    losses: list[float] = []

    if not report_path.exists():
        study_report = StudyReport(
            tuner_config=tuner_config,
            loss_function_weights=loss_function_weights,
            simulation_config=config,
            reports=[],
        )
        report_path.write_text(study_report.model_dump_json(indent=2))

    for seed in range(N_SEEDS):
        brain = Brain(model_path, config.brain)
        tank = Tank(brain, config.tank)

        # make headless:
        # tank._render_scene = lambda: None  # type: ignore[attr-defined, method-assign]

        vals = [f"{k}={v}" for k, v in config.brain.params]
        logger.info(f"{seed=}, {', '.join(vals)}")
        result = tank.run()
        loss = loss_function(result.report, loss_function_weights)
        logger.info(f"{loss=}")
        losses.append(loss)
        append_report(report_path, result.report)

    return sum(losses) / len(losses)


if __name__ == "__main__":
    logger.info("starting optimisation")
    logger.info(f"{RUNTIME=}, {N_TRIALS=}, {N_SEEDS=}")
    eta = RUNTIME * N_TRIALS * N_SEEDS

    hours, minutes = divmod(eta, 60 * 60)
    logger.info(f"eta: {hours} hours, {minutes} minutes")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(evaluate_simulation, n_trials=N_TRIALS)

    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best loss: {study.best_value}")
