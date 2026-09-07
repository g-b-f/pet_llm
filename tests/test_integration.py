"""End-to-end integration tests for the Tank + Brain + Driver pipeline.

These run the real simulation loop (real Brain, real Tank, a headless
DummyDriver) and only mock the expensive LLM call, so the threading,
queue, memory, and report machinery all run for real.
"""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.mocks import ScriptedInference, MockInference
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
        brain = Brain(model_path, config.brain, MockInference())
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

    @pytest.mark.skip("not ready yet")
    def test_pygame_driver_displays_coordinates(
        self, config: SimulationConfig, model_path: Path
        ):
        target_x, target_y = 150, 200
        action = PetAction(thought="moving", target_x=target_x, target_y=target_y).model_dump_json()
        inference = ScriptedInference([RoleContent.assistant(action)])


        # actions = [
        #     PetAction(thought=f"swimming {i}", target_x=100+1, target_y=101+1) for i in range(100)
        # ]
        # thoughts = [RoleContent.assistant(action.model_dump_json()) for action in actions]
        # inference = ScriptedInference(thoughts)

        brain = Brain(model_path, config.brain, inference=inference)

        driver = PyGameDriver(10, (config.tank.screen_width, config.tank.screen_height))

        offset_x, offset_y = driver.bounds_offset
        expected_x = int(target_x) + offset_x
        expected_y = int(target_y) + offset_y

        with patch("pygame.draw") as mock_pg:
            tank = Tank(brain, config.tank, driver)
            tank.run()
        
            draw_circle_calls = mock_pg.draw.circle.call_args_list
            target_drawn = False
            actual:list[tuple[int,int]] = []

            for call_args in draw_circle_calls:
                args, _kwargs = call_args
                if len(args) >= 3 and args[2] == (expected_x, expected_y):
                    target_drawn = True
                    break
                actual.append( args[2] )

                
        assert target_drawn, f"Coordinate ({expected_x}, {expected_y}) was not drawn, actual: {actual}"

