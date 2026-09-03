import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from lib.utils import get_logger

logger = get_logger(__name__)

class EnvironmentalInfo(BaseModel):
    mouse: tuple[int,int] = Field(description="the location of the user's mouse")

class Action(Enum):
    move_to = "move_to"
    idle = "idle"
    swim_fast = "swim_fast"

class PetAction(BaseModel, use_enum_values=True):
    thought: str = Field(description="The thought process of the pet.")
    # action: Action = Field(description="The action to take.")
    target_x: int = Field(description="Target X coordinate.")
    target_y: int = Field(description="Target Y coordinate.")

    def get_thought(self):
        try:
            return self.thought.encode().decode()
        except UnicodeEncodeError:
            logger.warning(f"""non utf-8 thought: '{self.thought.encode(errors = "backslashreplace").decode(errors = "backslashreplace")}'""")
            logger.warning(f"""equivalent to: '{self.thought.encode(errors = "namereplace").decode(errors = "namereplace")}'""")
            return self.thought.encode(errors = "replace").decode(errors = "replace")

class Role(Enum):
    user = "user"
    system = "system"
    assistant = "assistant"

class RoleContent(BaseModel, use_enum_values=True):
    role: Role
    content: str

    @classmethod
    def user(cls, content:str):
        return cls(role=Role.user, content=content)

    @classmethod
    def system(cls, content:str):
        return cls(role=Role.system, content=content)

    @classmethod
    def assistant(cls, content:str):
        return cls(role=Role.assistant, content=content)

class MessageChoice(BaseModel):
    index: int
    message: RoleContent
    logprobs: Optional[dict]
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[MessageChoice]
    usage: Usage

    def get_message(self):
        return self.choices[0].message

    def get_action(self):
        return PetAction(**json.loads(self.get_message().content))
