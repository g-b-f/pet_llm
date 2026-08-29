from collections import deque
from collections.abc import MutableSequence
import json
from lib.extra_types import PetAction, EnvironmentalInfo, ChatCompletionResponse, RoleContent
from llama_cpp.llama_types import ChatCompletionRequestMessage

class Memory:
    def __init__(self, maxlen: int):
        self._memory_queue:deque[RoleContent] = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def get_messages(self, system_prompt:str) -> list[ChatCompletionRequestMessage]:
        sys_prompt = RoleContent.system(system_prompt)
        pydantic_messages = sys_prompt = list(self._memory_queue)
        messages = [msg.model_dump() for msg in pydantic_messages]
        return messages

    def get_action(self, index:int) -> PetAction|None:
        memory = self._memory_queue[index].content
        try:
            return PetAction(**json.loads(memory))
        except:
            return None

    @property
    def length(self):
        return len(self._memory_queue)

    @property
    def is_empty(self):
        return self.length == 0

    @property
    def is_full(self):
        return self.length == self.maxlen
    
    def clear(self):
        self._memory_queue.clear()

    def append(self, message:RoleContent):
        self._memory_queue.append(message)
        assert 1

    def __len__(self):
        return len(self._memory_queue)