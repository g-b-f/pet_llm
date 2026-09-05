import json
import time
import tomllib
from pathlib import Path

from lib.brain import Brain
from lib.drivers.base import DriverBase
from lib.types.other import RenderInfo, EnvironmentalInfo
from lib.types.config import SimulationConfig, TankConfig
from lib.types.report import OutputReport

DEBUG = True

tom = (Path(__file__).parent.parent/ "pyproject.toml").read_text()
version = tomllib.loads(tom)["project"]["version"]

report_path = Path(__file__).parent.parent / "report.json"


class Tank:
    """Handles Pygame window rendering and interface displays.

    The pet's position, target, and thought live in the Brain in tank-local
    coordinates; this class only adds `bounds_offset` when drawing them on screen.
    """

    # Layout (pixels)
    TEXT_BOX_HEIGHT = 100
    TEXT_BOX_MARGIN = 10
    TANK_PADDING_X = 50
    STATUS_LOC = (20, 10)
    THOUGHT_LOC = (20, 30)
    DEBUG_LOC = (20, 110)

    # Colors (R, G, B)
    BACKGROUND_COLOR = (18, 26, 38)
    TEXT_BOX_COLOR = (10, 15, 25)
    PET_COLOR = (255, 140, 0)
    TARGET_COLOR = (255, 200, 100)
    THOUGHT_TEXT_COLOR = (220, 220, 220)
    THINKING_STATUS_COLOR = (200, 200, 130)
    SWIMMING_STATUS_COLOR = (130, 200, 130)

    # Pet rendering
    PET_RADIUS = 16
    TARGET_RADIUS = 4
    TARGET_OUTLINE_WIDTH = 1

    # UI text and timing
    FONT_NAME = "monospace"
    FONT_SIZE = 15
    FPS = 60

    def __init__(self, brain:Brain, config: TankConfig, driver: DriverBase):
        self.brain = brain
        self.config = config
        self.driver = driver

        self.bounds = (self.config.screen_width, self.config.screen_height)
        self.bounds_offset = self.TANK_PADDING_X, self.TEXT_BOX_HEIGHT // 2        

    def get_info(self) -> EnvironmentalInfo:
        """Gets information from the outside world to pass to the brain"""
        # TODO: make this a string that gets appended
        # ret = EnvironmentalInfo(mouse=pygame.mouse.get_pos())
        ret = EnvironmentalInfo(mouse=(0,0))

        return ret

    def get_report(self) -> OutputReport:
        brain_report = self.brain.report
        brain_report.actual_runtime = time.time() - self.start_time

        return OutputReport(
            config=SimulationConfig(tank=self.config, brain=self.brain.config),
            report=brain_report
            )
    
    def generate_report(self) -> OutputReport:
        brain_report = self.brain.report
        # brain_report.actual_runtime = pygame.time.get_ticks() / 1000
        brain_report.actual_runtime = round(time.time() - self.start_time, 1)

        with open(report_path, "r") as f:
            try:
                report: list[dict] = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print(f"error decoding: {e}")
                report = []
            except FileNotFoundError as e:
                print(f"file not found: {e}")
                report = []

        compiled_report = self.get_report()
        with open(report_path, "w") as f:
            report.append(compiled_report.model_dump())
            json.dump(report, f, indent=4)

        return compiled_report

    def run(self) -> OutputReport:
        """Runs the main game loop"""

        tank_bounds = (
            self.config.screen_width - 2 * self.TANK_PADDING_X, 
            self.config.screen_height - self.TEXT_BOX_HEIGHT
        )
        self.brain.wake_up(tank_bounds)
        self.start_time = time.time()

        while self.driver.running == True:
            brain_info = RenderInfo.from_brain(self.brain)
            self.brain.update(EnvironmentalInfo(mouse=(0,0)))
            self.driver.loop(brain_info)

        return self.get_report()