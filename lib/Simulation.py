import pygame
from textwrap import wrap

from lib.brain import Brain

class PetTankSimulation:
    """Handles Pygame window rendering, movement interpolation, and interface displays."""

    # Layout (pixels)
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    TEXT_BOX_HEIGHT = 100
    TEXT_BOX_MARGIN = 10
    TANK_PADDING_X = 50
    STATUS_LOC = (20, 10)
    THOUGHT_LOC = (20, 30)

    # Colors (R, G, B)
    BACKGROUND_COLOR = (18, 26, 38)
    TEXT_BOX_COLOR = (10, 15, 25)
    PET_COLOR = (255, 140, 0)
    TARGET_COLOR = (255, 200, 100)
    THOUGHT_TEXT_COLOR = (220, 220, 220)
    THINKING_STATUS_COLOR = (200, 200, 130)
    SWIMMING_STATUS_COLOR = (130, 200, 130)

    # Pet rendering and movement
    PET_RADIUS = 16
    PET_SPEED = 2.5
    TARGET_RADIUS = 4
    TARGET_OUTLINE_WIDTH = 1
    ARRIVAL_THRESHOLD = 3.0

    # UI text and timing
    FONT_NAME = "monospace"
    FONT_SIZE = 15
    FPS = 60
    INITIAL_THOUGHT = "Waking up in the tank..."

    def __init__(self, brain:Brain, bounds:tuple[int,int], bounds_offset:tuple[int,int]) -> None:
        self.brain = brain
        self.bounds = bounds
        self.bounds_offset = bounds_offset

        pygame.init()
        self.screen = pygame.display.set_mode(self.bounds)
        pygame.display.set_caption("Pet LLM")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(self.FONT_NAME, self.FONT_SIZE)

        bounds_x, bounds_y = self.bounds
        offset_x, offset_y = self.bounds_offset
        tank_bounds = (bounds_x - offset_x, bounds_y - offset_y)

        self.pet_x = float(tank_bounds[0] // 2)
        self.pet_y = float(tank_bounds[1] // 2)

        self.target_x = float(self.pet_x)
        self.target_y = float(self.pet_y)
        self.current_thought = self.INITIAL_THOUGHT

    def run(self) -> None:
        """Executes the main Pygame loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self._update_pet_logic()
            self._render_scene()
            self.clock.tick(self.FPS)

        pygame.quit()

    def _update_pet_logic(self) -> None:
        if not self.brain.result_queue.empty():
            decision = self.brain.result_queue.get()
            self.current_thought = decision.get_thought()
            self.target_x = decision.get_target_x(self.pet_x)
            self.target_y = decision.get_target_y(self.pet_y)

        delta_x = self.target_x - self.pet_x
        delta_y = self.target_y - self.pet_y
        distance = (delta_x**2 + delta_y**2) ** 0.5

        if distance > self.ARRIVAL_THRESHOLD:
            self.pet_x += (delta_x / distance) * self.PET_SPEED
            self.pet_y += (delta_y / distance) * self.PET_SPEED
        else:
            self.brain.request_decision_async(
                int(self.pet_x),
                int(self.pet_y)
            )

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


    def _get_text_box_rect(self) -> pygame.Rect:
        """Builds the background rectangle behind the status and thought text."""
        return pygame.Rect(
            self.TEXT_BOX_MARGIN,
            self.TEXT_BOX_MARGIN,
            self.bounds[0] - 2 * self.TEXT_BOX_MARGIN,
            self.TEXT_BOX_HEIGHT - 2 * self.TEXT_BOX_MARGIN,
        )

    def _render_scene(self):
        self.screen.fill(self.BACKGROUND_COLOR)

        pygame.draw.circle(self.screen, self.PET_COLOR, (int(self.pet_x), int(self.pet_y)), self.PET_RADIUS)
        pygame.draw.circle(self.screen, self.TARGET_COLOR, (int(self.target_x), int(self.target_y)), self.TARGET_RADIUS, self.TARGET_OUTLINE_WIDTH)

        text_box_rect = self._get_text_box_rect()
        pygame.draw.rect(self.screen, self.TEXT_BOX_COLOR, text_box_rect)

        if self.current_thought != self.INITIAL_THOUGHT:
            status_label = "Status: Thinking..." if self.brain.is_thinking else "Status: Swimming"
            status_color = self.THINKING_STATUS_COLOR if self.brain.is_thinking else self.SWIMMING_STATUS_COLOR
            status_surface = self.font.render(status_label, True, status_color)
            self.screen.blit(status_surface, self.STATUS_LOC)

        self._blit_text(self.screen, "Thought: " + self.current_thought, self.THOUGHT_LOC, self.font, self.THOUGHT_TEXT_COLOR)

        pygame.display.flip()
