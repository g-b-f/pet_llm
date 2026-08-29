from lib.brain import Brain
from lib.tank import Tank
from lib.extra_types import SimulationConfig
from models.download import Model, get_model

model_path = get_model(Model.smollm)

RUNTIME = 600

if __name__ == "__main__":
    config = SimulationConfig.model_construct()
    config.tank.runtime = RUNTIME

    brain = Brain(model_path, config.brain)
    simulation = Tank(brain, config.tank)
    simulation.run()