from unittest.mock import MagicMock, patch

import pytest

from lib.types.other import EnvironmentalInfo
from lib.types.config import BrainConfig, TankConfig
from lib.tank import Tank


@pytest.fixture
def mock_brain() -> MagicMock:
    brain = MagicMock()
    brain.current_x = 50.0
    brain.current_y = 50.0
    brain.target_x = 60.0
    brain.target_y = 60.0
    brain.current_thought = "test thought"
    brain.is_thinking = False
    brain.config = BrainConfig.model_construct()
    brain.debug_info = {"current": (50, 50), "target": (60, 60), "iteration": 0}
    return brain


@pytest.fixture
def tank_config() -> TankConfig:
    return TankConfig.model_construct()




@pytest.fixture
def tank(mock_brain: MagicMock, tank_config: TankConfig, mocker: MagicMock):
    mocker.running = False
    t = Tank(mock_brain, tank_config, mocker)
    yield t


class TestTankInit:
    def test_brain_stored(self, tank: Tank, mock_brain: MagicMock):
        assert tank.brain is mock_brain

    def test_bounds_from_config(self, tank: Tank, tank_config: TankConfig):
        assert tank.bounds == (tank_config.screen_width, tank_config.screen_height)

    def test_bounds_offset(self, tank: Tank):
        assert tank.bounds_offset == (Tank.TANK_PADDING_X, Tank.TEXT_BOX_HEIGHT // 2)


class TestRunWakeUp:
    def test_wake_up_called_with_tank_bounds(
        self, tank: Tank, mock_brain: MagicMock, tank_config: TankConfig
    ):
        tank.config.runtime = 0  # end the loop immediately
        # tank._mock_pygame.time.get_ticks.return_value = 0
        # tank._mock_pygame.event.get.return_value = []
        with patch.object(Tank, "get_report"):
            tank.run()
        expected_w = tank_config.screen_width - 2 * Tank.TANK_PADDING_X
        expected_h = tank_config.screen_height - Tank.TEXT_BOX_HEIGHT
        mock_brain.wake_up.assert_called_once_with((expected_w, expected_h))


@pytest.mark.skip("not needed for now")
class TestGetInfo:
    def test_returns_environmental_info(self, tank: Tank):
        tank._mock_pygame.mouse.get_pos.return_value = (100, 200)
        info = tank.get_info()
        assert isinstance(info, EnvironmentalInfo)
        assert info.mouse == (100, 200)

