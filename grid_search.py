from lib.brain import Brain
from lib.extra_types import SimulationConfig
from lib.tank import Tank
from lib.utils import frange, get_logger
from models.download import Model, get_model

model_path = get_model(Model.smollm2)
logger = get_logger(__name__)

RUNTIME = 300

if __name__ == "__main__":
    config = SimulationConfig.model_construct()
    config.tank.runtime = RUNTIME

    for temperature in frange(1.5, 2.5, 0.3):
        for seed in range(1,6):
            logger.info(f"{temperature=}, {seed=}")

            config.brain.params.temperature = temperature

            brain = Brain(model_path, config.brain)
            simulation = Tank(brain, config.tank)
            simulation.run()
