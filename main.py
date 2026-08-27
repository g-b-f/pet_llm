from pathlib import Path

from lib.brain import Brain
from lib.simulation import PetTankSimulation

MODEL_FILE_PATH = (Path().parent/"models/qwen2.5-1.5b-instruct-q4_k_m.gguf")


# Layout (pixels)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TEXT_BOX_HEIGHT = 100
TEXT_BOX_MARGIN = 10
TANK_PADDING_X = 50
STATUS_LOC = (20, 10)
THOUGHT_LOC = (20, 30)

bounds = SCREEN_WIDTH, SCREEN_HEIGHT
bounds_offset = TANK_PADDING_X, TEXT_BOX_HEIGHT // 2


if __name__ == "__main__":
    brain = Brain(MODEL_FILE_PATH, bounds, bounds_offset)
    simulation = PetTankSimulation(brain, bounds, bounds_offset)
    simulation.run()