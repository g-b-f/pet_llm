from unittest.mock import MagicMock, patch
import pytest

from lib.tank import Tank
from tests.mocks import BlockingBrain
from lib.drivers import PyGameDriver, DummyDriver
from lib.types.config import TankConfig
from lib.types.other import RenderInfo

BOUNDS = (100, 100)

@pytest.fixture
def render_info():
    return RenderInfo(
        current_x=20,
        current_y=20,
        target_x=30,
        target_y=30,
        current_thought="hello",
        is_thinking=True,
        has_started_thinking=True,
        debug_info={}
    )


@pytest.fixture
def driver():
    with patch("lib.drivers.pygame_driver.pygame") as mock_pg:
        mock_surface = MagicMock()
        mock_surface.get_size.return_value = (800, 600)
        mock_pg.display.set_mode.return_value = mock_surface
        mock_pg.time.Clock.return_value = MagicMock()
        mock_font = MagicMock()
        mock_font.size.side_effect = lambda s: (len(s) * 8, 15)
        mock_pg.font.SysFont.return_value = mock_font
        driver = PyGameDriver(None, BOUNDS)
        driver._mock_pygame = mock_pg
        yield driver


def test_drivers_give_same_results():
    config = TankConfig.model_construct()

    py_driver = PyGameDriver(10, (500,500))
    py_brain = BlockingBrain.generate_valid_thoughts(15)
    py_brain.memory.total_recall = True
    py_tank = Tank(py_brain, config, py_driver)

    dummy_driver = DummyDriver(5)
    dummy_brain = BlockingBrain.generate_valid_thoughts(15)
    dummy_brain.memory.total_recall = True
    dummy_tank = Tank(dummy_brain, config, dummy_driver)

    py_tank.run()
    dummy_tank.run()

    py_thoughts = list(py_tank.brain.memory.total_memory)
    dummy_thoughts = list(dummy_tank.brain.memory.total_memory)

    # dummy driver doesn't have "Start exploring"?
    assert py_thoughts == dummy_thoughts


class TestPyGameDriver:
    def test_render_calls_correct_functions(self, driver: PyGameDriver, render_info: RenderInfo):
        driver.render(render_info)
        driver.screen.fill.assert_called_once_with(PyGameDriver.BACKGROUND_COLOR)
        driver._mock_pygame.display.flip.assert_called_once()
        assert driver._mock_pygame.draw.circle.call_count >= 2

    def test_status_shown_when_started_thinking(
        self, driver: PyGameDriver, render_info: RenderInfo
    ):
        render_info.has_started_thinking = True
        driver.render(render_info)
        render_calls = driver.font.render.call_args_list
        status_calls = [call for call in render_calls if "Status:" in str(call)]
        assert len(status_calls) > 0

    def test_status_hidden_when_not_started_thinking(
        self, driver: PyGameDriver, render_info: RenderInfo
    ):
        render_info.has_started_thinking = False
        driver.render(render_info)
        render_calls = driver.font.render.call_args_list
        status_calls = [c for c in render_calls if "Status:" in str(c)]
        assert len(status_calls) == 0

    def test_blits_text_wraps_long_text(self, driver: PyGameDriver):
        surface = MagicMock()
        surface.get_size.return_value = (100, 600)
        font = MagicMock()
        font.size.side_effect = lambda s: (len(s) * 8, 15)
        long_text = "word " * 50
        driver._blit_text(surface, long_text, (0, 0), font, (255, 255, 255))
        assert surface.blit.call_count > 1
