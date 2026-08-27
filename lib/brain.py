import json
import random
import threading
import queue
from pathlib import Path
from typing import Iterator
from llama_cpp import Llama

from lib.extra_types import PetAction

class Brain:
    """Manages background inference with llama-cpp-python without blocking the rendering loop."""

    CONTEXT_SIZE = 2048
    GPU_LAYERS_ALL = -1  # -1 offloads every layer to the GPU
    TEMPERATURE = 0.7
    FALLBACK_THOUGHT = "Mind empty... drifting randomly."

    def __init__(self, model_path: str|Path, bounds:tuple[int,int], bounds_offset:tuple[int,int]) -> None:
        if isinstance(model_path, Path):
            model_path = str(model_path.resolve())

        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.CONTEXT_SIZE,
            n_gpu_layers=self.GPU_LAYERS_ALL,
            verbose=False
        )
        self.is_thinking = False
        self.result_queue: queue.Queue[PetAction] = queue.Queue()

        self.x_bounds, self.y_bounds = bounds
        self.x_offset, self.y_offset = bounds_offset

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
                target_x=random.randint(self.x_offset, self.x_bounds - self.x_offset),
                target_y=random.randint(self.y_offset, self.y_bounds - self.y_offset)
            )
            self.result_queue.put(fallback_decision)
        finally:
            self.is_thinking = False
