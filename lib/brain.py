import json
import random
import threading
import queue
from pathlib import Path
from typing import Iterator, cast
from collections import deque
from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage
from time import time
from hashlib import md5


from lib.extra_types import PetAction, EnvironmentalInfo
from lib.utils import get_logger

logger = get_logger(__name__, "info")

class Brain:
    """Manages pet state and background inference with llama-cpp-python without blocking the rendering loop.

    All coordinates are tank-local: (0, 0) is the top-left corner of the swimmable
    tank area and (x_bounds, y_bounds) is the bottom-right corner. Applying any
    on-screen offset is the simulation's responsibility.
    """

    CONTEXT_SIZE = 2048
    TEMPERATURE = 2
    FALLBACK_THOUGHT = "Mind empty... drifting randomly."
    INITIAL_THOUGHT = "Waking up..."

    MEMORY_LENGTH = 3

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

        self.memory: deque[ChatCompletionRequestMessage] = deque(maxlen=self.MEMORY_LENGTH)
        self.memory.clear()
        self.INITIAL_PROMPT = "Start exploring!"
        self.memory.append({"role": "user", "content": self.INITIAL_PROMPT})
        self.iterations = 0

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.CONTEXT_SIZE,
            n_gpu_layers=-1,
            verbose=False
        )
        self.awake = True

    def update(self, environment_info: EnvironmentalInfo) -> None:
        """Applies queued LLM decisions and advances the pet toward its target.

        Call once per frame from the rendering loop.
        """
        self.environment_info = environment_info

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
        assert self.awake, "still asleep!"
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
            "Formulate a thought then pick coordinates inside the tank bounds to move toward. "
            "Try to keep moving and not stay in the same place."
            "Adhere strictly to the requested JSON schema.\n"
            f"Tank bounds: ({self.x_bounds}, {self.y_bounds}). Your position: ({current_x}, {current_y}).\n"
            # f"Your owner's finger is at {self.environment_info.mouse}"
        )
        if self.current_thought == self.INITIAL_THOUGHT:
            prompt_hash = md5(system_prompt.encode("utf-8")).hexdigest()
            logger.info(f"system prompt hash: {prompt_hash}")

        try:
            start_time = time()
            messages=[cast(ChatCompletionRequestMessage, {"role": "system", "content": system_prompt})] + list(self.memory)
            response = self.llm.create_chat_completion(
                messages,
                temperature=self.TEMPERATURE,
                presence_penalty=0.6,
                frequency_penalty=0.8,
                response_format={
                    "type": "json_object",
                    "schema": PetAction.model_json_schema()
                },
            )
            assert not isinstance(response, Iterator)
            message = response["choices"][0]["message"]
            content = message["content"]
            assert content is not None

            self.memory.append(message)
            dict_decision = json.loads(content)
            parsed_decision = PetAction(**dict_decision)
            self.result_queue.put(parsed_decision)
            self.iterations += 1

            try:
                first_thought = json.loads(self.memory[0]["content"]) # type: ignore[reportTypedDictNotRequiredAccess, arg-type]
                last_thought = json.loads(self.memory[-1]["content"]) # type: ignore[arg-type]
                if len(self.memory) == self.memory.maxlen and first_thought == last_thought: 
                    logger.info(f"thought loop detected after {self.iterations} iterations, clearing memory")
                    # self.memory[1] = {"role": "system", "content": "you'd like to do something else now"}
                    self.memory.clear()
            except (KeyError, json.JSONDecodeError):
                pass

            end_time = time()
            logger.debug(f"thought for {end_time - start_time:.1f} seconds")

        except Exception as e:
            print(f"Error during decision generation: {e}")
            logger.exception(e)

            fallback_decision = PetAction(
                thought=self.FALLBACK_THOUGHT,
                action="move_to",
                target_x=random.randint(0, self.x_bounds),
                target_y=random.randint(0, self.y_bounds)
            )
            self.result_queue.put(fallback_decision)
        finally:
            self.is_thinking = False
