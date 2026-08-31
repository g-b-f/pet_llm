import json
from collections import deque

from llama_cpp.llama_types import ChatCompletionRequestMessage

from lib.extra_types import PetAction, RoleContent, MemoryConfig
from lib.utils import get_logger

logger = get_logger(__name__, "info", log_file="log_bayes.txt") 

class MemoryHandlerError(Exception):
    pass

class ThoughtLoopError(MemoryHandlerError):
    def __init__(self, last_thought, *args) -> None:
        self.last_thought = last_thought
        super().__init__(*args)

class Memory:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._memory_queue: deque[RoleContent] = deque(maxlen=config.max_length)
        self.thought_loops = 0

    def get_messages(self, system_prompt:str) -> list[ChatCompletionRequestMessage]:
        sys_prompt = RoleContent.system(system_prompt)
        pydantic_messages = [sys_prompt] + list(self._memory_queue)
        messages = [msg.model_dump() for msg in pydantic_messages]
        return messages # type:ignore[report-return-type]

    def get_action(self, index:int) -> PetAction|None:
        memory = self._memory_queue[index].content
        try:
            return PetAction(**json.loads(memory))
        except:
            return None

    def supervise(self):
        if not self.is_full:
            logger.debug("memory not full, returning")
            return
        
        first_action = self.get_action(0)
        last_action = self.get_action(-1)

        if first_action is None or last_action is None:
            return

        if first_action.thought == last_action.thought:
            self.thought_loops += 1
            raise ThoughtLoopError(last_action.thought)

    @property
    def length(self) -> int:
        return len(self._memory_queue)

    @property
    def is_empty(self) -> bool:
        return self.length == 0

    @property
    def is_full(self) -> bool:
        return self.length == self.config.max_length
    
    def clear(self):
        self._memory_queue.clear()

    def __add__(self, message:RoleContent) -> "Memory":
        self._memory_queue.append(message)
        return self

    def __len__(self) -> int:
        return len(self._memory_queue)