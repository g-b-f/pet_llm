from pathlib import Path

from lib.brain import Brain
from lib.simulation import PetTankSimulation

# MODEL_FILE_PATH = (Path().parent/"models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
MODEL_FILE_PATH = (Path().parent/"models/smollm2-1.7b-q8_0.gguf")

# Layout (pixels)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TEXT_BOX_HEIGHT = 100
TANK_PADDING_X = 50

bounds = SCREEN_WIDTH, SCREEN_HEIGHT
bounds_offset = TANK_PADDING_X, TEXT_BOX_HEIGHT // 2

if __name__ == "__main__":
    brain = Brain(MODEL_FILE_PATH)
    simulation = PetTankSimulation(brain, bounds, bounds_offset)
    simulation.run()