import json
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

    COMMENTS = "Testing out different models"

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


    def __init__(self, model: Model, version:int, config: SimulationConfig) -> None:
        self.config = config
        self.model = model
        self.model_path = get_model(self.model)
        self.study_name = f"v{version}_pet_llm_{self.model_path.stem}"
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

            vals = [f"{k}={v}" for k, v in config.brain.params]
            logger.info(f"{seed=}, {', '.join(vals)}")

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

        num_trials = 0
        total_trials = N_TRIALS*N_SEEDS

        if self.report_path.exists():
            data = json.loads(self.report_path.read_text())
            num_trials = len(StudyReport(**data).trials)
            logger.info(f"{num_trials=}, {total_trials=}, {N_TRIALS - round(num_trials // N_SEEDS)=}")
            if len(StudyReport(**data).trials) >= N_TRIALS*N_SEEDS:
                logger.info(f"enough trials for {self.report_path.stem}, exiting")
                return
            del data

        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True
        )
        study.optimize(
            self.evaluate_simulation,
            n_trials=N_TRIALS - num_trials // N_SEEDS,
            show_progress_bar=True,
            catch=(RuntimeError))

        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best loss: {study.best_value}")


if __name__ == "__main__":
    original_version = 11

    options = [Model.smollm3, Model.gemma, Model.granite, Model.deepseek]

    eta = RUNTIME * N_TRIALS * N_SEEDS * len(options)
    print(f"eta: {humanize.naturaltime(eta, future=True)}")

    for ver, model in enumerate(options):
        ver = 0 # keep same version for now

        config = SimulationConfig.model_construct()
        opt = Optimiser(model, original_version+ver, config)

        logger.info(f"starting for {model.value}")
        opt.run()