from textwrap import wrap
from typing import Any
import pygame
from lib.drivers.base import DriverBase


DEBUG = True

class PyGameDriver(DriverBase):

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

    

    def __init__(self, end_time:int|None, bounds: tuple[int, int]):

        self.end_time = end_time
        self.bounds = bounds
        self.bounds_offset = self.TANK_PADDING_X, self.TEXT_BOX_HEIGHT // 2

        pygame.init()
        self.screen = pygame.display.set_mode(self.bounds)
        pygame.display.set_caption("Pet LLM")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(self.FONT_NAME, self.FONT_SIZE)

        super().__init__

    
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


    def render(self, info:Any):
        self.screen.fill(self.BACKGROUND_COLOR)

        offset_x, offset_y = self.bounds_offset
        pet_screen_x = int(info.current_x) + offset_x
        pet_screen_y = int(info.current_y) + offset_y
        target_screen_x = int(info.target_x) + offset_x
        target_screen_y = int(info.target_y) + offset_y

        pygame.draw.circle(self.screen, self.PET_COLOR, (pet_screen_x, pet_screen_y), self.PET_RADIUS)
        pygame.draw.circle(self.screen, self.TARGET_COLOR, (target_screen_x, target_screen_y), self.TARGET_RADIUS, self.TARGET_OUTLINE_WIDTH)

        text_box_rect = pygame.Rect(
            self.TEXT_BOX_MARGIN,
            self.TEXT_BOX_MARGIN,
            self.bounds[0] - 2 * self.TEXT_BOX_MARGIN,
            self.TEXT_BOX_HEIGHT - 2 * self.TEXT_BOX_MARGIN,
        )
        pygame.draw.rect(self.screen, self.TEXT_BOX_COLOR, text_box_rect)

        if info.current_thought != info.config.thoughts.initial_thought:
            status_label = "Status: Thinking..." if info.is_thinking else "Status: Swimming"
            status_color = self.THINKING_STATUS_COLOR if info.is_thinking else self.SWIMMING_STATUS_COLOR
            status_surface = self.font.render(status_label, True, status_color)
            self.screen.blit(status_surface, self.STATUS_LOC)

        self._blit_text(self.screen, "Thought: " + info.current_thought, self.THOUGHT_LOC, self.font, self.THOUGHT_TEXT_COLOR)

        if DEBUG:
            brain_debug = ""
            for k, v in info.debug_info.items():
                if isinstance(v, float):
                    v = round(v,2)
                if brain_debug:
                    brain_debug += "  "
                brain_debug += f"{k}:{v}"

            coords = brain_debug + f"  v{info.version}"
            debug_surface = self.font.render(coords, True, pygame.Color("white"))
            self.screen.blit(debug_surface, self.DEBUG_LOC)

        pygame.display.flip()

    def loop(self, end_time: None|int, brain_info:Any, ):
        if end_time is not None and pygame.time.get_ticks() < end_time:
            self.running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        if self.running == False:
            pygame.quit()


        # info = self.get_info()
        # self.brain.update(info)
        self.render(brain_info)
        self.clock.tick(self.FPS)