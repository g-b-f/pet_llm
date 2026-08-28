from lib.brain import Brain
from lib.simulation import PetTankSimulation
from models.download import get_model

model_path = get_model("qwen2.5-1.5b-instruct-q4_k_m")
model_path = get_model("smollm2-1.7b-q8_0")

# Layout (pixels)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TEXT_BOX_HEIGHT = 100
TANK_PADDING_X = 50

bounds = SCREEN_WIDTH, SCREEN_HEIGHT
bounds_offset = TANK_PADDING_X, TEXT_BOX_HEIGHT // 2

if __name__ == "__main__":
    brain = Brain(model_path)
    simulation = PetTankSimulation(brain, bounds, bounds_offset)
    simulation.run()