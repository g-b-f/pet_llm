import pygame
from textwrap import wrap

from lib.brain import Brain
from lib.extra_types import EnvironmentalInfo

DEBUG = True


class PetTankSimulation:
    """Handles Pygame window rendering and interface displays.

    The pet's position, target, and thought live in the Brain in tank-local
    coordinates; this class only adds `bounds_offset` when drawing them on screen.
    """

    # Layout (pixels)
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
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

    def __init__(self, brain:Brain, bounds:tuple[int,int], bounds_offset:tuple[int,int]) -> None:
        self.brain = brain
        self.bounds = bounds
        self.bounds_offset = bounds_offset

        tank_bounds = (self.SCREEN_WIDTH - 2 * self.TANK_PADDING_X, self.SCREEN_HEIGHT - self.TEXT_BOX_HEIGHT)


        pygame.init()
        self.screen = pygame.display.set_mode(self.bounds)
        pygame.display.set_caption("Pet LLM")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(self.FONT_NAME, self.FONT_SIZE)

        self.brain.wake_up(tank_bounds)

    def get_info(self) -> EnvironmentalInfo:
        """Gets information from the outside world to pass to the brain"""
        # TODO: make this a string that gets appended
        mouse = pygame.mouse.get_pos()
        ret = EnvironmentalInfo(mouse=mouse)

        return ret

    def run(self) -> None:
        """Executes the main Pygame loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            info = self.get_info()

            self.brain.update(info)
            self._render_scene()
            self.clock.tick(self.FPS)

        pygame.quit()

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

        if self.brain.current_thought != self.brain.INITIAL_THOUGHT:
            status_label = "Status: Thinking..." if self.brain.is_thinking else "Status: Swimming"
            status_color = self.THINKING_STATUS_COLOR if self.brain.is_thinking else self.SWIMMING_STATUS_COLOR
            status_surface = self.font.render(status_label, True, status_color)
            self.screen.blit(status_surface, self.STATUS_LOC)

        self._blit_text(self.screen, "Thought: " + self.brain.current_thought, self.THOUGHT_LOC, self.font, self.THOUGHT_TEXT_COLOR)

        if DEBUG:
            self.brain.target_x
            coords = f"current: ({self.brain.current_x:.1f}, {self.brain.current_y:.1f}) "\
            f"target: ({self.brain.target_x:.1f}, {self.brain.target_y:.1f}) "\
            f"memory length: {len(self.brain.memory)} "\
            f"iteration: {self.brain.iterations}"
            debug_surface = self.font.render(coords, True, pygame.Color("white"))
            self.screen.blit(debug_surface, self.DEBUG_LOC)

        pygame.display.flip()
