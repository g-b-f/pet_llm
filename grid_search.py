from pathlib import Path

from lib.brain import Brain
from lib.drivers import DummyDriver
from lib.inference import LlamaCpp
from lib.optimisation_helpers import append_report
from lib.tank import Tank
from lib.types.config import LossFunctionWeights, TunerConfig
from lib.types.report import StudyReport
from lib.utils import get_logger, values_from_trial
from models.download import Model, get_model

model_path = get_model(Model.llama)
logger = get_logger(__name__)

RUNTIME = 300
version = 13
comments = "thought guiding using schema: thought regex is '^[a-zA-Z .!?,']{10,250}$'"

report_path = Path(__file__).parent / f"reports/v{version}_pet_llm_{model_path.stem}.json"

if __name__ == "__main__":
    config = values_from_trial(187)
    config.tank.runtime = RUNTIME

    # for temperature in frange(1.5, 2.5, 0.3):
    if 1:

        if not report_path.exists():
            study_report = StudyReport(
                comments=comments,
                tuner_config=TunerConfig.model_construct(),
                loss_function_weights=LossFunctionWeights.model_construct(),
                simulation_config=config,
                trials=[],
            )
            report_path.write_text(study_report.model_dump_json(indent=2))
        
        for seed in range(1, 6):
            logger.info(f"{seed=}")

            bounds = Tank.get_bounds(config.tank)
            inference = LlamaCpp(config.brain.params, model_path)
            brain = Brain(config.brain, bounds, inference)
            simulation = Tank(brain, config.tank, DummyDriver(RUNTIME))
            result = simulation.run()

            append_report(report_path, result.report, config.brain.params)
