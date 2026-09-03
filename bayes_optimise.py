from pathlib import Path

import optuna
import humanize

from lib.brain import Brain
from lib.types.config import (
    LossFunctionWeights,
    SimulationConfig,
    TunerConfig,
)
from lib.types.report import StudyReport
from lib.optimisation_helpers import append_report, storage, suggest_vals
from lib.tank import Tank
from lib.utils import get_logger, loss_function
from models.download import Model, get_model

RUNTIME = 200
N_TRIALS = 20
N_SEEDS = 3

logger = get_logger(__name__, "debug", log_file="log.txt")

class Optimiser:

    version = 6
    COMMENTS = "Improved message upon trying to leave tank"

    loss_function_weights = LossFunctionWeights(
    thought_loop=10.0,
    empty_thought=10.0,
    out_of_bounds=10.0,
    malformed_json=100.0,
    invalid_chars=10.0,
    )

    tuner_config = TunerConfig(
        temperature=(0.8, 2.5),
        frequency_penalty=(0.2, 2.5),
        presence_penalty=(0.2, 2.5),
        repeat_penalty=(0.2, 2.5),
    )

    config = SimulationConfig.model_construct()

    def __init__(self, model: Model) -> None:
        self.model = model
        self.model_path = get_model(self.model)
        self.study_name = f"v{self.version}_pet_llm_{self.model_path.stem}"
        self.report_path = Path(__file__).parent / f"reports/{self.study_name}.json"

    def evaluate_simulation(self, trial: optuna.Trial) -> float:
        config = self.config
        config.tank.runtime = RUNTIME
        config.brain.params = suggest_vals(trial, self.tuner_config, config.brain.params)

        losses: list[float] = []

        if not self.report_path.exists():
            study_report = StudyReport(
                comments=self.COMMENTS,
                tuner_config=self.tuner_config,
                loss_function_weights=self.loss_function_weights,
                simulation_config=config,
                trials=[],
            )
            self.report_path.write_text(study_report.model_dump_json(indent=2))

        for seed in range(N_SEEDS):
            config.brain.params.seed = seed
            brain = Brain(self.model_path, config.brain)
            tank = Tank(brain, config.tank)

            # make headless:
            # tank._render_scene = lambda: None  # type: ignore[attr-defined, method-assign]

            vals = ""
            for k, v in config.brain.params:
                if isinstance(v, float):
                    v = round(v,2)
                if vals:
                    vals += ", "
                vals += f"{k}={v}"
            logger.info(vals)

            result = tank.run()
            loss = loss_function(result.report, self.loss_function_weights)
            logger.info(f"{loss=}")
            losses.append(loss)
            append_report(self.report_path, result.report, config.brain.params)

        return sum(losses) / len(losses)

    def run(self):
        logger.info("starting optimisation")
        logger.info(f"{RUNTIME=}, {N_TRIALS=}, {N_SEEDS=}")
        eta = RUNTIME * N_TRIALS * N_SEEDS
        logger.info(f"eta: {humanize.naturaltime(eta, future=True)}")

        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(self.evaluate_simulation, n_trials=N_TRIALS, show_progress_bar=True)

        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best loss: {study.best_value}")


if __name__ == "__main__":
    model = Model.smollm2
    for ver, oob in enumerate([
        "You can't leave the tank! Try a coordinate inside ({}, {}).",
        "You can't leave the tank! Ensure x coordinate is between 0 and {}, and y coordinate is between 0 and {}.",
        "You can't leave the tank!",
        ]):

        logger.info(f"starting for {model.value}")
        opt = Optimiser(model)
        opt.version += ver
        opt.config.brain.thoughts.out_of_bounds_message = oob
        opt.run()