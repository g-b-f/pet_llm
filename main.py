from lib.brain import Brain
from lib.tank import Tank
from lib.extra_types import BrainConfig
from models.download import Model, get_model

model_path = get_model(Model.smollm)

if __name__ == "__main__":
    brain_config = BrainConfig.model_construct()
    brain = Brain(model_path, brain_config)
    simulation = Tank(brain)
    simulation.run()