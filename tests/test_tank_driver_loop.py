"""Integration-style tests for the Tank.run() main loop.

These verify that information still flows correctly between Tank, Brain and
DriverBase after the rendering code was refactored out of Tank into drivers:
- Tank wakes the brain with tank-local (screen-offset-free) bounds.
- Each frame the driver receives a RenderInfo snapshot of the brain.
- The driver cannot steer the brain: it only receives a snapshot for drawing,
  so mutations to it never reach the brain.
- EnvironmentalInfo from Tank.get_info() reaches Brain.update().
- The loop stops when driver.running becomes False and a report is produced.
"""

from unittest.mock import MagicMock

import pytest

from lib.drivers.base import DriverBase
from lib.tank import Tank
from lib.types.config import BrainConfig, TankConfig
from lib.types.other import EnvironmentalInfo, RenderInfo
from lib.types.report import BrainReport

N_FRAMES = 3


class ScriptedDriver(DriverBase):
    """Minimal driver double: runs n_frames iterations and records the
    RenderInfo snapshots it receives."""

    def __init__(self, n_frames: int):
        super().__init__(None)
        self.n_frames = n_frames
        self.received: list[RenderInfo] = []
        self.frames = 0

    def render(self, info: RenderInfo) -> None:  # pragma: no cover - not used
        pass

    def loop(self, info: RenderInfo) -> None:
        self.received.append(info)
        self.frames += 1
        if self.frames >= self.n_frames:
            self.running = False


class MutatingDriver(ScriptedDriver):
    """Driver double that corrupts every RenderInfo it is handed, to prove
    driver mutations cannot leak back into the brain.

    Records a pristine deep copy of each snapshot before corrupting the one
    Tank handed over.
    """

    def loop(self, info: RenderInfo) -> None:
        self.received.append(info.model_copy(deep=True))
        info.target_x = -999
        info.target_y = -999
        info.current_thought = "corrupted"
        info.debug_info.clear()
        self.frames += 1
        if self.frames >= self.n_frames:
            self.running = False


@pytest.fixture
def mock_brain() -> MagicMock:
    """A brain double that mimics the post-wake_up state of a real Brain."""
    brain = MagicMock()
    brain.current_x = 350.0
    brain.current_y = 250.0
    brain.target_x = 360.0
    brain.target_y = 260.0
    brain.current_thought = "just a test thought"
    brain.is_thinking = False
    brain.config = BrainConfig.model_construct()
    brain.report = BrainReport.model_construct()
    brain.debug_info = {"current": (350, 250), "target": (360, 260), "iteration": 0}
    return brain


@pytest.fixture
def tank_config() -> TankConfig:
    return TankConfig.model_construct()


class TestWakeUpBounds:
    """The brain must be woken with tank-local bounds, not screen bounds."""

    def test_wake_up_uses_tank_local_bounds(
        self, mock_brain: MagicMock, tank_config: TankConfig
    ):
        driver = ScriptedDriver(n_frames=1)
        tank = Tank(mock_brain, tank_config, driver)
        tank.run()

        expected_w = tank_config.screen_width - 2 * Tank.TANK_PADDING_X
        expected_h = tank_config.screen_height - Tank.TEXT_BOX_HEIGHT
        mock_brain.wake_up.assert_called_once_with((expected_w, expected_h))


class TestRenderInfoFlow:
    def test_driver_receives_snapshot_each_frame(
        self, mock_brain: MagicMock, tank_config: TankConfig
    ):
        driver = ScriptedDriver(n_frames=N_FRAMES)
        tank = Tank(mock_brain, tank_config, driver)
        tank.run()

        assert len(driver.received) == N_FRAMES
        for info in driver.received:
            assert isinstance(info, RenderInfo)
            # snapshot taken from the brain's state that frame
            assert info.current_x == mock_brain.current_x
            assert info.current_y == mock_brain.current_y
            assert info.current_thought == mock_brain.current_thought
            assert info.is_thinking == mock_brain.is_thinking
            assert info.debug_info == mock_brain.debug_info

    @pytest.mark.parametrize(
        ("thought", "started"),
        [(None, False), ("a real thought", True)],
        ids=["initial_thought", "new_thought"],
    )
    def test_has_started_thinking_flag(
        self,
        mock_brain: MagicMock,
        tank_config: TankConfig,
        thought: str | None,
        *,
        started: bool,
    ):
        """has_started_thinking is False only while the thought equals the
        configured initial thought."""
        if thought is None:
            thought = mock_brain.config.thoughts.initial_thought
        mock_brain.current_thought = thought
        driver = ScriptedDriver(n_frames=1)
        Tank(mock_brain, tank_config, driver).run()
        assert driver.received[0].has_started_thinking is started

    def test_loop_runs_until_driver_stops(
        self, mock_brain: MagicMock, tank_config: TankConfig
    ):
        driver = ScriptedDriver(n_frames=N_FRAMES)
        Tank(mock_brain, tank_config, driver).run()

        assert mock_brain.update.call_count == N_FRAMES
        assert driver.running is False


class TestDriverCannotSteerBrain:
    """The driver only receives a RenderInfo snapshot for drawing; mutating it
    must have no effect on the brain's state."""

    def test_driver_mutation_does_not_change_brain(
        self, mock_brain: MagicMock, tank_config: TankConfig
    ):
        driver = MutatingDriver(n_frames=1)
        Tank(mock_brain, tank_config, driver).run()

        assert mock_brain.target_x == 360.0
        assert mock_brain.target_y == 260.0
        assert mock_brain.current_thought == "just a test thought"

    def test_driver_mutation_not_visible_next_frame(
        self, mock_brain: MagicMock, tank_config: TankConfig
    ):
        driver = MutatingDriver(n_frames=N_FRAMES)
        Tank(mock_brain, tank_config, driver).run()

        # each frame's snapshot is built fresh from the brain
        for info in driver.received:
            assert info.target_x == mock_brain.target_x
            assert info.target_y == mock_brain.target_y


class TestEnvironmentalInfoFlow:
    def test_brain_update_receives_environmental_info(
        self, mock_brain: MagicMock, tank_config: TankConfig
    ):
        driver = ScriptedDriver(n_frames=N_FRAMES)
        Tank(mock_brain, tank_config, driver).run()

        for call in mock_brain.update.call_args_list:
            (env_info,), _ = call
            assert isinstance(env_info, EnvironmentalInfo)
            assert env_info.mouse == (0, 0)


class TestReport:
    def test_report_wraps_brain_report_and_configs(
        self, mock_brain: MagicMock, tank_config: TankConfig
    ):
        driver = ScriptedDriver(n_frames=1)
        tank = Tank(mock_brain, tank_config, driver)
        output = tank.run()

        assert output.report is mock_brain.report
        assert output.config.tank is tank_config
        assert output.config.brain is mock_brain.config
        assert isinstance(output.report.actual_runtime, float)
