import json
import random
import threading
import queue
from pathlib import Path
from typing import Iterator
from llama_cpp import Llama

from lib.extra_types import PetAction

class Brain:
    """Manages pet state and background inference with llama-cpp-python without blocking the rendering loop.

    All coordinates are tank-local: (0, 0) is the top-left corner of the swimmable
    tank area and (x_bounds, y_bounds) is the bottom-right corner. Applying any
    on-screen offset is the simulation's responsibility.
    """

    CONTEXT_SIZE = 2048
    GPU_LAYERS_ALL = -1  # -1 offloads every layer to the GPU
    TEMPERATURE = 0.7
    FALLBACK_THOUGHT = "Mind empty... drifting randomly."
    INITIAL_THOUGHT = "Waking up in the tank..."

    # Movement
    PET_SPEED = 2.5
    ARRIVAL_THRESHOLD = 3.0

    def __init__(self, model_path: str|Path) -> None:
        if isinstance(model_path, Path):
            model_path = model_path.resolve()
        self.model_path = str(model_path)
        self.awake = False

    def wake_up(self, bounds:tuple[int,int]):
        self.x_bounds, self.y_bounds = bounds

        self.current_x = float(self.x_bounds // 2)
        self.current_y = float(self.y_bounds // 2)
        self.target_x = self.current_x
        self.target_y = self.current_y
        self.current_thought = self.INITIAL_THOUGHT

        self.is_thinking = False
        self.result_queue: queue.Queue[PetAction] = queue.Queue()

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.CONTEXT_SIZE,
            n_gpu_layers=self.GPU_LAYERS_ALL,
            verbose=False
        )
        self.awake = True

    def update(self) -> None:
        """Applies queued LLM decisions and advances the pet toward its target.

        Call once per frame from the rendering loop.
        """
        if not self.result_queue.empty():
            decision = self.result_queue.get()
            self.current_thought = decision.get_thought()
            self.target_x = decision.get_target_x(self.current_x)
            self.target_y = decision.get_target_y(self.current_y)

        delta_x = self.target_x - self.current_x
        delta_y = self.target_y - self.current_y
        distance = (delta_x**2 + delta_y**2) ** 0.5

        if distance > self.ARRIVAL_THRESHOLD:
            self.current_x += (delta_x / distance) * self.PET_SPEED
            self.current_y += (delta_y / distance) * self.PET_SPEED
        else:
            self.request_decision_async(
                int(self.current_x),
                int(self.current_y)
            )

    def request_decision_async(self, current_x: int, current_y: int) -> None:
        """Triggers a background thread to generate the next pet thought and action.

        Args:
            current_x: Current horizontal position of the pet.
            current_y: Current vertical position of the pet.
        """
        if self.is_thinking:
            return

        self.is_thinking = True
        worker_thread = threading.Thread(
            target=self._generate_decision,
            args=(current_x, current_y),
            daemon=True
        )
        worker_thread.start()

    def _generate_decision(self, current_x: int, current_y: int) -> None:
        system_prompt = (
            "You are a small pet living in a glass tank window. "
            "Formulate a brief thought and pick coordinates inside the tank bounds to move toward. "
            "Adhere strictly to the requested JSON schema."
        )
        user_prompt = f"Tank bounds: ({self.x_bounds}, {self.y_bounds}). Your position: ({current_x}, {current_y})."

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
                temperature=self.TEMPERATURE
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
                thought=self.FALLBACK_THOUGHT,
                action="move_to",
                target_x=random.randint(0, self.x_bounds),
                target_y=random.randint(0, self.y_bounds)
            )
            self.result_queue.put(fallback_decision)
        finally:
            self.is_thinking = False
