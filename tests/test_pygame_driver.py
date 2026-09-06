from unittest.mock import MagicMock, patch

import pytest

from lib.drivers.pygame_driver import PyGameDriver
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


class TestTankInit:
    def test_bounds_offset(self, driver: PyGameDriver):
        assert driver.bounds_offset == (
            PyGameDriver.TANK_PADDING_X,
            PyGameDriver.TEXT_BOX_HEIGHT // 2
        )

    def test_pygame_init_called(self, driver: PyGameDriver):
        driver._mock_pygame.init.assert_called_once()

    def test_display_mode_set(self, driver: PyGameDriver):
        driver._mock_pygame.display.set_mode.assert_called_once_with(BOUNDS)

    def test_caption_set(self, driver: PyGameDriver):
        driver._mock_pygame.display.set_caption.assert_called_once_with("Pet LLM")


class TestRenderScene:
    def test_screen_filled(self, driver: PyGameDriver, render_info: RenderInfo):
        driver.render(render_info)
        driver.screen.fill.assert_called_once_with(PyGameDriver.BACKGROUND_COLOR)

    def test_display_flipped(self, driver: PyGameDriver, render_info: RenderInfo):
        driver.render(render_info)
        driver._mock_pygame.display.flip.assert_called_once()

    def test_pet_drawn(self, driver: PyGameDriver, render_info: RenderInfo):
        driver.render(render_info)
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


class TestBlitText:
    def test_blits_text(self, driver: PyGameDriver):
        surface = MagicMock()
        surface.get_size.return_value = (800, 600)
        font = MagicMock()
        font.size.side_effect = lambda s: (len(s) * 8, 15)
        driver._blit_text(surface, "hello world", (0, 0), font, (255, 255, 255))
        assert surface.blit.call_count > 0

    def test_wraps_long_text(self, driver: PyGameDriver):
        surface = MagicMock()
        surface.get_size.return_value = (100, 600)
        font = MagicMock()
        font.size.side_effect = lambda s: (len(s) * 8, 15)
        long_text = "word " * 50
        driver._blit_text(surface, long_text, (0, 0), font, (255, 255, 255))
        assert surface.blit.call_count > 1
