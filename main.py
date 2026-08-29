from lib.brain import Brain
from lib.tank import Tank
from lib.extra_types import BrainConfig
from models.download import Model, get_model

model_path = get_model(Model.smollm)

if __name__ == "__main__":
    brain = Brain(model_path, BrainConfig.model_construct())
    simulation = Tank(brain)
    simulation.run()