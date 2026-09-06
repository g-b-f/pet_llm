import time
import tomllib
from pathlib import Path

from lib.brain import Brain
from lib.drivers.base import DriverBase
from lib.types.config import SimulationConfig, TankConfig
from lib.types.other import EnvironmentalInfo, RenderInfo
from lib.types.report import OutputReport

DEBUG = True

tom = (Path(__file__).parent.parent / "pyproject.toml").read_text()
version = tomllib.loads(tom)["project"]["version"]

report_path = Path(__file__).parent.parent / "report.json"


class Tank:
    """Handles Pygame window rendering and interface displays.

    The pet's position, target, and thought live in the Brain in tank-local
    coordinates; this class only adds `bounds_offset` when drawing them on screen.
    """

    # TODO: decouple/ remove these
    TEXT_BOX_HEIGHT = 100
    TEXT_BOX_MARGIN = 10
    TANK_PADDING_X = 50

    def __init__(self, brain: Brain, config: TankConfig, driver: DriverBase):
        self.brain = brain
        self.config = config
        self.driver = driver

        self.bounds = (self.config.screen_width, self.config.screen_height)
        self.bounds_offset = self.TANK_PADDING_X, self.TEXT_BOX_HEIGHT // 2

    def get_info(self) -> EnvironmentalInfo:
        """Gets information from the outside world to pass to the brain"""
        # ret = EnvironmentalInfo(mouse=pygame.mouse.get_pos())
        ret = EnvironmentalInfo(mouse=(0, 0))

        return ret

    def get_report(self) -> OutputReport:
        brain_report = self.brain.report
        brain_report.actual_runtime = round(time.time() - self.start_time, 1)

        return OutputReport(
            config=SimulationConfig(tank=self.config, brain=self.brain.config),
            report=brain_report
            )

    def run(self) -> OutputReport:
        """Runs the main rendering loop"""

        tank_bounds = (
            self.config.screen_width - 2 * self.TANK_PADDING_X,
            self.config.screen_height - self.TEXT_BOX_HEIGHT
        )
        self.brain.wake_up(tank_bounds)
        self.start_time = time.time()

        while self.driver.running:
            render_info = RenderInfo.from_brain(self.brain)
            self.brain.update(self.get_info())
            self.driver.loop(render_info)

        return self.get_report()
