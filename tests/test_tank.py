from unittest.mock import MagicMock, patch

import pytest

from lib.extra_types import BrainConfig, EnvironmentalInfo, TankConfig
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
def tank(mock_brain: MagicMock, tank_config: TankConfig):
    with patch("lib.tank.pygame") as mock_pg:
        mock_surface = MagicMock()
        mock_surface.get_size.return_value = (800, 600)
        mock_pg.display.set_mode.return_value = mock_surface
        mock_pg.time.Clock.return_value = MagicMock()
        mock_font = MagicMock()
        mock_font.size.side_effect = lambda s: (len(s) * 8, 15)
        mock_pg.font.SysFont.return_value = mock_font
        t = Tank(mock_brain, tank_config)
        t._mock_pygame = mock_pg
        yield t


class TestTankInit:
    def test_brain_stored(self, tank: Tank, mock_brain: MagicMock):
        assert tank.brain is mock_brain

    def test_bounds_from_config(self, tank: Tank, tank_config: TankConfig):
        assert tank.bounds == (tank_config.screen_width, tank_config.screen_height)

    def test_bounds_offset(self, tank: Tank):
        assert tank.bounds_offset == (Tank.TANK_PADDING_X, Tank.TEXT_BOX_HEIGHT // 2)

    def test_pygame_init_called(self, tank: Tank):
        tank._mock_pygame.init.assert_called_once()

    def test_display_mode_set(self, tank: Tank, tank_config: TankConfig):
        tank._mock_pygame.display.set_mode.assert_called_once_with(
            (tank_config.screen_width, tank_config.screen_height)
        )

    def test_caption_set(self, tank: Tank):
        tank._mock_pygame.display.set_caption.assert_called_once_with("Pet LLM")


class TestRunWakeUp:
    def test_wake_up_called_with_tank_bounds(
        self, tank: Tank, mock_brain: MagicMock, tank_config: TankConfig
    ):
        tank.config.runtime = 0  # end the loop immediately
        tank._mock_pygame.time.get_ticks.return_value = 0
        tank._mock_pygame.event.get.return_value = []
        with patch.object(Tank, "generate_report"):
            tank.run()
        expected_w = tank_config.screen_width - 2 * Tank.TANK_PADDING_X
        expected_h = tank_config.screen_height - Tank.TEXT_BOX_HEIGHT
        mock_brain.wake_up.assert_called_once_with((expected_w, expected_h))


class TestGetInfo:
    def test_returns_environmental_info(self, tank: Tank):
        tank._mock_pygame.mouse.get_pos.return_value = (100, 200)
        info = tank.get_info()
        assert isinstance(info, EnvironmentalInfo)
        assert info.mouse == (100, 200)


class TestRenderScene:
    def test_screen_filled(self, tank: Tank):
        tank._render_scene()
        tank.screen.fill.assert_called_once_with(Tank.BACKGROUND_COLOR)

    def test_display_flipped(self, tank: Tank):
        tank._render_scene()
        tank._mock_pygame.display.flip.assert_called_once()

    def test_pet_drawn(self, tank: Tank):
        tank._render_scene()
        assert tank._mock_pygame.draw.circle.call_count >= 2

    def test_status_shown_when_not_initial_thought(self, tank: Tank):
        tank._render_scene()
        render_calls = tank.font.render.call_args_list
        status_calls = [c for c in render_calls if "Status:" in str(c)]
        assert len(status_calls) > 0

    def test_status_hidden_when_initial_thought(
        self, tank: Tank, mock_brain: MagicMock
    ):
        mock_brain.current_thought = mock_brain.config.thoughts.initial_thought
        tank._render_scene()
        render_calls = tank.font.render.call_args_list
        status_calls = [c for c in render_calls if "Status:" in str(c)]
        assert len(status_calls) == 0


class TestBlitText:
    def test_blits_text(self, tank: Tank):
        surface = MagicMock()
        surface.get_size.return_value = (800, 600)
        font = MagicMock()
        font.size.side_effect = lambda s: (len(s) * 8, 15)
        tank._blit_text(surface, "hello world", (0, 0), font, (255, 255, 255))
        assert surface.blit.call_count > 0

    def test_wraps_long_text(self, tank: Tank):
        surface = MagicMock()
        surface.get_size.return_value = (100, 600)
        font = MagicMock()
        font.size.side_effect = lambda s: (len(s) * 8, 15)
        long_text = "word " * 50
        tank._blit_text(surface, long_text, (0, 0), font, (255, 255, 255))
        assert surface.blit.call_count > 1
