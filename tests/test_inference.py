from pathlib import Path
from unittest.mock import MagicMock
import pytest

from lib.brain import Brain
from lib.inference.base import InferenceBase
from lib.types.config import BrainConfig, ParamsConfig
from lib.types.other import EnvironmentalInfo, PetAction, RoleContent


class DummyInferenceAdapter(InferenceBase):
    def __init__(self, config: ParamsConfig, action_to_return: PetAction):
        super().__init__(config)
        self.action_to_return = action_to_return
        self.calls_made = 0

    def create_chat_completion(self, messages: list[RoleContent]) -> RoleContent:
        self.calls_made += 1
        return RoleContent.assistant(self.action_to_return.model_dump_json())


def test_brain_accepts_custom_inference_adapter():
    config = BrainConfig.model_construct()
    expected_action = PetAction(thought="exploring", target_x=75, target_y=80)
    adapter = DummyInferenceAdapter(config.params, expected_action)

    brain = Brain(Path("fake.gguf"), config, inference=adapter)
    assert brain.inference is adapter

    brain.wake_up((100, 100))

    # Trigger generation directly
    brain._generate_decision(50, 50)
    assert adapter.calls_made == 1
    assert not brain.result_queue.empty()
    action = brain.result_queue.get()
    assert action.thought == "exploring"
    assert action.target_x == 75
    assert action.target_y == 80


class ScriptedDriver:
    def __init__(self, actions_to_emit: list[tuple[int, int]]):
        self.running = True
        self.actions_to_emit = actions_to_emit
        self.calls = 0

    def loop(self, render_info):
        self.calls += 1
        if self.calls > 5:
            self.running = False


def test_tank_simulation_with_decoupled_inference():
    from lib.tank import Tank
    from lib.types.config import TankConfig

    config = BrainConfig.model_construct()
    expected_action = PetAction(thought="swimming", target_x=50, target_y=50)
    adapter = DummyInferenceAdapter(config.params, expected_action)

    brain = Brain(Path("fake.gguf"), config, inference=adapter)
    tank_config = TankConfig.model_construct(screen_width=100, screen_height=100, runtime=0.1)
    driver = ScriptedDriver([])

    tank = Tank(brain, tank_config, driver)
    tank.run()
    assert adapter.calls_made >= 1

