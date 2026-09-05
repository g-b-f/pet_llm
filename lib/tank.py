import json
import time
import tomllib
from pathlib import Path
from textwrap import wrap

import pygame

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

        pygame.init()
        self.screen = pygame.display.set_mode(self.bounds)
        pygame.display.set_caption("Pet LLM")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(self.FONT_NAME, self.FONT_SIZE)

        

    def get_info(self) -> EnvironmentalInfo:
        """Gets information from the outside world to pass to the brain"""
        # TODO: make this a string that gets appended
        ret = EnvironmentalInfo(mouse=pygame.mouse.get_pos())
        return ret

    def get_report(self) -> OutputReport:
        brain_report = self.brain.report
        brain_report.actual_runtime = pygame.time.get_ticks() / 1000

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

    def run(self,) -> OutputReport:
        """Runs the main game loop"""

        tank_bounds = (
            self.config.screen_width - 2 * self.TANK_PADDING_X, 
            self.config.screen_height - self.TEXT_BOX_HEIGHT
        )
        self.brain.wake_up(tank_bounds)
        self.start_time = time.time()

        while self.driver.running == True:
            brain_info = RenderInfo.from_brain(self.brain)
            self.driver.loop(self.config.runtime, brain_info)

        return self.get_report()

    def _blit_text(
            self,
            surface:pygame.Surface,
            text:str,
            pos:tuple[int,int],
            font:pygame.font.Font,
            color: tuple[int,int,int] | pygame.Color
            ):
        space_width = font.size(' ')[0]
        max_width, max_height = surface.get_size()
        x, y = pos
        max_chars = (max_width - x) // space_width
        lines = wrap(text, max_chars)
        for line in lines:
            line_width, line_height = font.size(line)
            if x + line_width > max_width:
                break
            surface.blit(font.render(line, True, color), (x, y))
            y += line_height


    def _render_scene(self):
        self.screen.fill(self.BACKGROUND_COLOR)

        offset_x, offset_y = self.bounds_offset
        pet_screen_x = int(self.brain.current_x) + offset_x
        pet_screen_y = int(self.brain.current_y) + offset_y
        target_screen_x = int(self.brain.target_x) + offset_x
        target_screen_y = int(self.brain.target_y) + offset_y

        pygame.draw.circle(self.screen, self.PET_COLOR, (pet_screen_x, pet_screen_y), self.PET_RADIUS)
        pygame.draw.circle(self.screen, self.TARGET_COLOR, (target_screen_x, target_screen_y), self.TARGET_RADIUS, self.TARGET_OUTLINE_WIDTH)

        text_box_rect = pygame.Rect(
            self.TEXT_BOX_MARGIN,
            self.TEXT_BOX_MARGIN,
            self.bounds[0] - 2 * self.TEXT_BOX_MARGIN,
            self.TEXT_BOX_HEIGHT - 2 * self.TEXT_BOX_MARGIN,
        )
        pygame.draw.rect(self.screen, self.TEXT_BOX_COLOR, text_box_rect)

        if self.brain.current_thought != self.brain.config.thoughts.initial_thought:
            status_label = "Status: Thinking..." if self.brain.is_thinking else "Status: Swimming"
            status_color = self.THINKING_STATUS_COLOR if self.brain.is_thinking else self.SWIMMING_STATUS_COLOR
            status_surface = self.font.render(status_label, True, status_color)
            self.screen.blit(status_surface, self.STATUS_LOC)

        self._blit_text(self.screen, "Thought: " + self.brain.current_thought, self.THOUGHT_LOC, self.font, self.THOUGHT_TEXT_COLOR)

        if DEBUG:
            brain_debug = ""
            for k, v in self.brain.debug_info.items():
                if isinstance(v, float):
                    v = round(v,2)
                if brain_debug:
                    brain_debug += "  "
                brain_debug += f"{k}:{v}"

            coords = brain_debug + f"  v{version}"
            debug_surface = self.font.render(coords, True, pygame.Color("white"))
            self.screen.blit(debug_surface, self.DEBUG_LOC)

        pygame.display.flip()
