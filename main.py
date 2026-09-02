import json
from pathlib import Path

from lib.brain import Brain
from lib.types.config import SimulationConfig
from lib.tank import Tank
from lib.utils import get_logger
from models.download import Model, get_model

model_path = get_model(Model.smollm2)
logger = get_logger(__name__)

def values_from_trial(trial_id:int, config = SimulationConfig.model_construct(), fpath=Path(__file__).parent/"study_backend.jsonl"):
    with open(fpath) as f:
        for line in f.readlines():
            d = json.loads(line)

            if d.get("trial_id") == trial_id:
                if d.get("param_name") == "temperature":
                    config.brain.params.temperature = d["param_value_internal"]
                if d.get("param_name") == "frequency_penalty":
                    config.brain.params.frequency_penalty = d["param_value_internal"]
                if d.get("param_name") == "presence_penalty":
                    config.brain.params.presence_penalty = d["param_value_internal"]
                if d.get("param_name") == "repeat_penalty":
                    config.brain.params.repeat_penalty = d["param_value_internal"]

    print(config.brain.params.model_dump_json(indent=2))
    return config

if __name__ == "__main__":
    config = values_from_trial(98)
    brain = Brain(model_path, config.brain)
    simulation = Tank(brain, config.tank)
    simulation.run()