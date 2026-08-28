from lib.brain import Brain
from lib.tank import Tank
from models.download import get_model

# model_path = get_model("qwen2.5-1.5b-instruct-q4_k_m")
model_path = get_model("smollm2-1.7b-q8_0")

if __name__ == "__main__":
    brain = Brain(model_path)
    simulation = Tank(brain)
    simulation.run()