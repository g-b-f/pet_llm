import json
import random
import threading
import queue
from pathlib import Path
import pygame
from pydantic import BaseModel, Field
from llama_cpp import Llama
from llama_cpp.llama_types import CreateChatCompletionResponse
from textwrap import wrap
from typing import Iterator


MODEL_FILE_PATH = (Path().parent/"models/qwen2.5-1.5b-instruct-q4_k_m.gguf")


class PetAction(BaseModel):
    thought: str = Field(description="The thought process of the pet.")
    action: str = Field(description="The action to take.", json_schema_extra={"enum": ["move_to", "idle", "swim_fast"]})
    target_x: int = Field(description="Target X coordinate.")
    target_y: int = Field(description="Target Y coordinate.")

    def get_thought(self) -> str:
        return self.thought if self.thought else ""

    def get_action(self) -> str:
        return self.action if self.action else "idle"

    def get_target_x(self, default=0.0) -> float:
        ret = self.target_x if self.target_x is not None else default
        return float(ret)

    def get_target_y(self, default=0.0) -> float:
        ret = self.target_y if self.target_y is not None else default
        return float(ret)

class Brain:
    """Manages background inference with llama-cpp-python without blocking the rendering loop."""

    def __init__(self, model_path: str) -> None:
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False
        )
        self.is_thinking = False
        self.result_queue: queue.Queue[PetAction] = queue.Queue()

    def request_decision_async(self, current_x: int, current_y: int, screen_width: int, screen_height: int) -> None:
        """Triggers a background thread to generate the next pet thought and action.

        Args:
            current_x: Current horizontal position of the pet.
            current_y: Current vertical position of the pet.
            screen_width: Width boundary of the tank.
            screen_height: Height boundary of the tank.
        """
        if self.is_thinking:
            return

        self.is_thinking = True
        worker_thread = threading.Thread(
            target=self._generate_decision,
            args=(current_x, current_y, screen_width, screen_height),
            daemon=True
        )
        worker_thread.start()

    def _generate_decision(self, current_x: int, current_y: int, screen_width: int, screen_height: int) -> None:
        system_prompt = (
            "You are a small pet living in a glass tank window. "
            "Formulate a brief thought and pick coordinates inside the tank bounds to move toward. "
            "Adhere strictly to the requested JSON schema."
        )
        user_prompt = f"Tank bounds: ({screen_width}, {screen_height}). Your position: ({current_x}, {current_y})."

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_object",
                    "schema": PetAction.model_json_schema()
                },
                temperature=0.7
            )
            assert not isinstance(response, Iterator)
            content = response["choices"][0]["message"]["content"]
            assert content is not None

            dict_decision = json.loads(content)
            parsed_decision = PetAction(**dict_decision)
            self.result_queue.put(parsed_decision)

        except Exception as e:
            print(f"Error during decision generation: {e}")

            fallback_decision = PetAction(
                thought= "Mind empty... drifting randomly.",
                action= "move_to",
                target_x= random.randint(50, screen_width - 50),
                target_y= random.randint(50, screen_height - 50)
            )
            self.result_queue.put(fallback_decision)
        finally:
            self.is_thinking = False


class PetTankSimulation:
    """Handles Pygame window rendering, movement interpolation, and interface displays."""

    THOUGHT_LOC = (20, 30)
    STATUS_LOC = (20, 10)
    TEXT_BOX_HEIGHT = THOUGHT_LOC[1] + STATUS_LOC[1] + 20

    def __init__(self, model_path: str|Path, screen_width: int = 800, screen_height: int = 600) -> None:
        if isinstance(model_path, Path):
            model_path = str(model_path.resolve())
        
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Pet LLM MVP")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 15)

        self.brain = Brain(model_path)
        
        self.pet_x = float(screen_width // 2)
        self.pet_y = float(screen_height // 2)
        self.target_x = float(self.pet_x)
        self.target_y = float(self.pet_y)
        self.pet_speed = 2.5
        self.current_thought = "Waking up in the tank..."

    def run(self) -> None:
        """Executes the main Pygame loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self._update_pet_logic()
            self._render_scene()
            self.clock.tick(60)

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

        if distance > 3.0:
            self.pet_x += (delta_x / distance) * self.pet_speed
            self.pet_y += (delta_y / distance) * self.pet_speed
        else:
            self.brain.request_decision_async(
                int(self.pet_x),
                int(self.pet_y),
                self.screen_width,
                self.screen_height
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


    def _render_scene(self) -> None:
        self.screen.fill((18, 26, 38))

        pygame.draw.circle(self.screen, (255, 140, 0), (int(self.pet_x), int(self.pet_y)), 16)
        pygame.draw.circle(self.screen, (255, 200, 100), (int(self.target_x), int(self.target_y)), 4, 1)
        pygame.draw.rect(self.screen, (10, 15, 25), (10, 10, self.screen_width - 20, self.TEXT_BOX_HEIGHT))

        if not self.current_thought == "Waking up in the tank...":
            status_label = "Status: Thinking..." if self.brain.is_thinking else "Status: Swimming"
            status_color = (200, 200, 130) if self.brain.is_thinking else (130, 200, 130)
            status_surface = self.font.render(status_label, True, status_color)
            self.screen.blit(status_surface, self.STATUS_LOC)

        self._blit_text(self.screen, "Thought: " + self.current_thought, self.THOUGHT_LOC, self.font, (220, 220, 220))

        pygame.display.flip()


if __name__ == "__main__":
    simulation = PetTankSimulation(model_path=MODEL_FILE_PATH)
    simulation.run()