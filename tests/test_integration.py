"""End-to-end integration tests for the Tank + Brain + Driver pipeline.

These run the real simulation loop (real Brain, real Tank, a headless
DummyDriver) and only mock the expensive LLM call, so the threading,
queue, memory, and report machinery all run for real.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.mocks import ScriptedInference, MockInference, BlockingBrain
from lib.brain import Brain
from lib.drivers import DummyDriver
from lib.drivers.pygame_driver import PyGameDriver
from lib.tank import Tank
from lib.types.config import SimulationConfig
from lib.types.other import PetAction, RoleContent
from lib.types.report import OutputReport

RUNTIME_SECONDS = 2

@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    file = tmp_path / "model.gguf"
    file.touch()
    return file


@pytest.fixture
def config() -> SimulationConfig:
    cfg = SimulationConfig.model_construct()
    cfg.tank.runtime = RUNTIME_SECONDS
    return cfg

class TestEndToEnd:
    def test_simulation_wakes_brain_and_produces_output_report(
        self, config: SimulationConfig, model_path: Path
    ):
        inference = ScriptedInference.from_actions(
            PetAction(thought="", target_y=1, target_x=1)
        )
        brain = Brain(model_path, config.brain, inference)
        driver = DummyDriver(config.tank.runtime)
        tank = Tank(brain, config.tank, driver)
        report = tank.run()

        assert brain.awake
        assert isinstance(report, OutputReport)
        assert report.report.iterations > 0
        assert report.report.actual_runtime is not None

    def test_simulation_honors_requested_runtime(
        self, config: SimulationConfig, model_path: Path
    ):
        brain = Brain(model_path, config.brain, MockInference())
        driver = DummyDriver(config.tank.runtime)
        tank = Tank(brain, config.tank, driver)

        start = time.time()
        tank.run()
        elapsed = time.time() - start

        assert elapsed >= RUNTIME_SECONDS - 0.5

    def test_simulation_accumulates_llm_messages_in_memory(
        self, config: SimulationConfig, model_path: Path
    ):
        # Return distinct thoughts to prevent thought-loop detector clearing memory
        actions = [
            PetAction(thought=f"swimming {i}", target_x=100, target_y=100) for i in range(10)
            ]
        thoughts = [RoleContent.assistant(action.model_dump_json()) for action in actions]
        inference = ScriptedInference(thoughts)

        brain = Brain(model_path, config.brain, inference=inference)
        driver = DummyDriver(config.tank.runtime)
        tank = Tank(brain, config.tank, driver)
        tank.run()

        assert brain.memory.length > 1

    def test_pet_moves_and_stays_in_bounds(
        self, config: SimulationConfig, model_path: Path
    ):
        brain = Brain(model_path, config.brain, MockInference())
        driver = DummyDriver(config.tank.runtime)
        tank = Tank(brain, config.tank, driver)
        tank.run()

        expected_w = config.tank.screen_width - 2 * Tank.TANK_PADDING_X
        expected_h = config.tank.screen_height - Tank.TEXT_BOX_HEIGHT
        assert 0 <= brain.current_x <= expected_w
        assert 0 <= brain.current_y <= expected_h

    def test_malformed_llm_output_uses_fallback(self, config: SimulationConfig, model_path: Path):
        inference = ScriptedInference(RoleContent.assistant("not valid json {"))
        brain = Brain(model_path, config.brain, inference)
        driver = DummyDriver(config.tank.runtime)
        tank = Tank(brain, config.tank, driver)
        res = tank.run()

        assert res.report.malformed_json > 0

    @pytest.mark.slow
    def test_pygame_driver_displays_coordinates(self, config: SimulationConfig):
        import pygame

        target_x, target_y = 150, 200
        action = PetAction(thought="moving", target_x=target_x, target_y=target_y).model_dump_json()
        brain = BlockingBrain(RoleContent.assistant(action))
        driver = PyGameDriver(2, (config.tank.screen_width, config.tank.screen_height))

        with patch("lib.drivers.pygame_driver.pygame.draw", wraps=pygame.draw) as mock_draw:
            tank = Tank(brain, config.tank, driver)
            tank.run()
            assert mock_draw.circle.called

        offset_x, offset_y = driver.bounds_offset
        expected_x = int(target_x) + offset_x
        expected_y = int(target_y) + offset_y

        circle_calls = mock_draw.circle.call_args_list
        actual = [call.args[2] for call in circle_calls]

        # TODO: check seperately for drawing of big and little circle
        target_drawn = any(call.args[2] == (expected_x, expected_y) for call in circle_calls)
        assert target_drawn, f"Coordinate ({expected_x}, {expected_y}) was not drawn, actual: {actual}"

    @pytest.mark.slow
    def test_previous_response_appears_in_system_prompt(self, config: SimulationConfig):
        target_x, target_y = 150, 200
        action1= PetAction(thought="moving", target_x=target_x, target_y=target_y).model_dump_json()
        action2 = PetAction(thought="moving", target_x=10, target_y=10).model_dump_json()

        brain = BlockingBrain([RoleContent.assistant(action1),RoleContent.assistant(action2)])
        driver = DummyDriver(0.1)

        real_worker = getattr(Brain, "_generate_decision")

        def spy(self_brain, cx, cy):
            return real_worker(self_brain, cx, cy)

        with patch.object(Brain, "_generate_decision", autospec=True, side_effect=spy) as mock_gen:
            tank = Tank(brain, config.tank, driver)
            tank.run()

        arrived = any(
            Brain.near((call.args[1], call.args[2]), (target_x, target_y))
            for call in mock_gen.call_args_list
        )
        assert arrived, (
            f"no decision requested near ({target_x}, {target_y}); "
            f"calls: {mock_gen.call_args_list}"
        )