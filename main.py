from lib.brain import Brain
from lib.tank import Tank
from models.download import get_model, Model

model_path = get_model(Model.smollm)

if __name__ == "__main__":
    brain = Brain(model_path)
    simulation = Tank(brain)
    simulation.run()