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

    for pres in range(8, 20, 3):
        for freq in range(8, 20, 3):
            for seed in range(1, 5):
                presence_penalty = pres/10
                frequency_penalty = freq/10
                logger.info(f"{presence_penalty=}, {frequency_penalty=}, {seed=}")

                config.brain.params.seed = seed
                config.brain.params.presence_penalty = presence_penalty
                config.brain.params.frequency_penalty = frequency_penalty

                brain = Brain(model_path, config.brain)
                simulation = Tank(brain, config.tank)
                simulation.run()