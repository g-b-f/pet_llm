import json
from pathlib import Path

import humanize
import optuna

from lib.brain import Brain
from lib.drivers import DummyDriver, PyGameDriver
from lib.optimisation_helpers import append_report, get_storage, suggest_vals
from lib.tank import Tank
from lib.types.config import LossFunctionWeights, SimulationConfig, TunerConfig
from lib.types.report import StudyReport
from lib.utils import get_logger, loss_function
from models.download import Model, get_model

RUNTIME = 200
N_TRIALS = 20
N_SEEDS = 3

logger = get_logger(__name__, "debug", log_file="log.txt")


class Optimiser:
    comments = "Testing out different models"
    n_jobs = 2

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

    def __init__(
        self, model: Model, version: int, config: SimulationConfig, comments: str, *, visual: bool
    ):
        self.config = config
        self.model = model
        self.comments = comments
        self.visual = visual

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
                comments=self.comments,
                tuner_config=self.tuner_config,
                loss_function_weights=self.loss_function_weights,
                simulation_config=config,
                trials=[],
            )
            self.report_path.write_text(study_report.model_dump_json(indent=2))

        for seed in range(N_SEEDS):
            runtime = config.tank.runtime
            visual_bounds = (config.tank.screen_width, config.tank.screen_height)
            driver = PyGameDriver(runtime, visual_bounds) if self.visual else DummyDriver(runtime)

            config.brain.params.seed = seed
            brain = Brain(self.model_path, config.brain)
            tank = Tank(brain, config.tank, driver)

            params = [f"{k}={v}" for k, v in config.brain.params]
            logger.info(f"{seed=}, {', '.join(params)}")

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
        total_trials = N_TRIALS * N_SEEDS

        if self.report_path.exists():
            data = json.loads(self.report_path.read_text())
            num_trials = len(StudyReport(**data).trials)
            logger.info(
                f"{num_trials=}, {total_trials=}, {N_TRIALS - round(num_trials // N_SEEDS)=}"
            )
            if len(StudyReport(**data).trials) >= N_TRIALS * N_SEEDS:
                logger.info(f"enough trials for {self.report_path.stem}, exiting")
                return
            del data

        storage = get_storage(self.n_jobs)
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
            catch=(RuntimeError),
            n_jobs=self.n_jobs
        )

        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best loss: {study.best_value}")


if __name__ == "__main__":
    original_version = 12

    options = [Model.smollm3, Model.llama, Model.granite, Model.deepseek, Model.smollm3, Model.gemma]

    eta = RUNTIME * N_TRIALS * N_SEEDS * len(options)
    print(f"eta: {humanize.naturaltime(eta, future=True)}")

    for version_increment, model in enumerate(options):
        version_increment = 0  # keep same version for now

        opt = Optimiser(
            model,
            original_version + version_increment,
            SimulationConfig.model_construct(),
            "Testing out different models, with dummy driver",
            visual=False,
        )

        logger.info(f"starting for {model.value}")
        opt.run()
