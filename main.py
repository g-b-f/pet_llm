from lib.brain import Brain
from lib.tank import Tank
from models.download import Model, get_model

model_path = get_model(Model.smollm)

if __name__ == "__main__":
    brain = Brain(model_path)
    simulation = Tank(brain)
    simulation.run()