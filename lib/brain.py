import json
import queue
import random
import string
import threading
from hashlib import md5
from pathlib import Path
from typing import Iterator

from llama_cpp import Llama

from lib import memory
from lib.types.other import (
    Action,
    ChatCompletionResponse,
    EnvironmentalInfo,
    PetAction,
    RoleContent,
)
from lib.types.config import BrainConfig
from lib.types.report import BrainReport
from lib.utils import get_logger

logger = get_logger(__name__, "debug") 

class Brain:
    """Manages pet state and background inference using llama_cpp.

    All coordinates are tank-local: (0, 0) is the top-left corner of the swimmable
    tank area and (x_bounds, y_bounds) is the bottom-right corner. Applying any
    on-screen offset is the simulation's responsibility.
    """

    PET_SPEED = 2.5
    ARRIVAL_THRESHOLD = 3.0
    MAX_OOB_COUNT = 3

    def __init__(self, model_path: Path, config: BrainConfig) -> None:
        self.awake = False
        self.model_path = str(model_path.resolve())
        self.config = config
        self.initial_memory = RoleContent.user(self.config.thoughts.initial_prompt)

    def wake_up(self, bounds:tuple[int,int]):
        self.x_bounds, self.y_bounds = bounds

        self.current_x = float(self.x_bounds // 2)
        self.current_y = float(self.y_bounds // 2)
        self.target_x = self.current_x
        self.target_y = self.current_y
        self.current_thought = self.config.thoughts.initial_thought

        self.is_thinking = False
        self.result_queue: queue.Queue[PetAction] = queue.Queue()

        self.memory = memory.Memory(self.config.memory)
        self.memory += self.initial_memory
        self.iterations = 0
        self.current_oob_count = 0

        self.report = BrainReport.model_construct()

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.config.params.context_size,
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
            self.target_x = decision.target_x
            self.target_y = decision.target_y

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

        self.debug_info = {
            # "current": (round(self.current_x,1), round(self.current_y,1)),
            # "target": (round(self.target_x,1), round(self.target_y,1)),
            "iteration": self.iterations,
            "temperature": self.config.params.temperature,
            "seed": self.config.params.seed
            }

    def _fallback(self):
        fallback_decision = PetAction(
            thought=self.config.thoughts.fallback_thought,
            # action=Action.move_to,
            target_x=random.randint(0, self.x_bounds),
            target_y=random.randint(0, self.y_bounds)
        )
        self.result_queue.put(fallback_decision)

    def is_valid_chars(self, thought:str) -> bool:
        thought_chars = set(thought)
        valid_chars = string.ascii_letters + " ,.?!'"
        valid_chars_set = set(valid_chars)
        quoted = thought.startswith("'") or thought.endswith("'")
        return thought_chars.issubset(valid_chars_set) and not quoted

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

    def target_out_of_bounds(self, action:PetAction) -> bool:
        target_x = action.target_x
        target_y = action.target_y
        if target_x > self.x_bounds or target_x < 0:
            return True
        return bool(target_y > self.y_bounds or target_y < 0)

    def _generate_decision(self, current_x: int, current_y: int) -> None:
        system_prompt = (
            "You are a small pet living in a glass tank. "
            "Formulate a thought then pick coordinates inside the tank bounds to move toward. "
            "Keep moving and don't stay in the same place."
            # "Do not attempt to leave the bounds of the tank."
            "Adhere strictly to the requested JSON schema.\n"
            f"Tank bounds: ({self.x_bounds}, {self.y_bounds}). "
            f"Your position: ({current_x}, {current_y}).\n"
            # f"Your owner's finger is at {self.environment_info.mouse}"
        )
        if self.current_thought == self.config.thoughts.initial_thought:
            prompt_hash = md5(system_prompt.encode("utf-8")).hexdigest()
            logger.debug(f"system prompt hash: {prompt_hash}")

        messages = self.memory.get_messages(system_prompt)
        response = self.llm.create_chat_completion(
            messages,
            temperature=self.config.params.temperature,
            presence_penalty=self.config.params.presence_penalty,
            frequency_penalty=self.config.params.frequency_penalty,
            repeat_penalty=self.config.params.repeat_penalty,
            min_p=self.config.params.min_p,
            seed=self.config.params.seed,
            response_format={
                "type": "json_object",
                "schema": PetAction.model_json_schema(), #type:ignore
            },
        )
        assert not isinstance(response, Iterator)
        parsed_response = ChatCompletionResponse(**response) #type:ignore[arg-type]
        message = parsed_response.get_message()

        try:
            action = parsed_response.get_action()
        except (json.JSONDecodeError, ValueError):
            self.report.malformed_json += 1
            logger.warning("malformed LLM JSON; using fallback decision and resetting is_thinking")
            self._fallback()
            self.is_thinking = False
            return

        thought = action.get_thought()
        logger.info(f"thought '{thought}'")
        self.memory += message

        if not self.is_valid_chars(thought):
            self.report.non_alphanumeric +=1
        if not thought.strip():
            self.report.empty_thoughts +=1
        
        if self.target_out_of_bounds(action):
            logger.info(f"tried to go to {action.target_x, action.target_y}")
            self.memory += RoleContent.system(
                "You can't leave the tank! "
                f"Try a coordinate inside {(self.x_bounds, self.y_bounds)}."
                )

            self.current_oob_count +=1
            self.report.out_of_bounds_attempts +=1

            if self.current_oob_count >= self.MAX_OOB_COUNT:
                logger.info("attempted out-of-bounds too much, clearing memory")
                self._fallback()
                self.memory.clear()
                self.current_oob_count = 0
        else:
            self.result_queue.put(action)
            self.current_oob_count = 0

        self.iterations += 1
        self.report.iterations +=1

        try:
            self.memory.supervise()
        except memory.ThoughtLoopError as e:
            self.report.thought_loops +=1
            logger.info(
                f"thought loop detected after {self.iterations} iterations, clearing memory"
                )
            logger.info(f"thought was: '{e.last_thought}'")
            # self.memory.append(RoleContent.system("you'd like to do something else now"))
            self.memory.clear()
            self._fallback()
        finally:
            self.is_thinking = False

 
