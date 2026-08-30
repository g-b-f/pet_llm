from lib.brain import Brain
from lib.tank import Tank
from lib.extra_types import SimulationConfig
from lib.utils import get_logger
from models.download import Model, get_model

model_path = get_model(Model.smollm2)
logger = get_logger(__name__)

RUNTIME = 300

if __name__ == "__main__":
    config = SimulationConfig.model_construct()
    config.tank.runtime = RUNTIME

    for enable_learning in [False, True]:
        for temp in range(10, 26, 3):
            for seed in range(1, 7):
                logger.info(f"{temp=}, {seed=}, {enable_learning=}")
                config.brain.params.temperature = temp/10
                config.brain.params.seed = seed
                config.brain.learning.enabled = enable_learning

                brain = Brain(model_path, config.brain)
                simulation = Tank(brain, config.tank)
                simulation.run()