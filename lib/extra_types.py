from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EnvironmentalInfo(BaseModel):
    mouse: tuple[int,int] = Field(description="the location of the user's mouse")

class Action(Enum):
    move_to = "move_to"
    idle = "idle"
    swim_fast = "swim_fast"

class PetAction(BaseModel, use_enum_values=True):
    thought: str = Field(description="The thought process of the pet.")
    action: Action = Field(description="The action to take.")
    target_x: int = Field(description="Target X coordinate.")
    target_y: int = Field(description="Target Y coordinate.")

    # def get_thought(self) -> str:
    #     return self.thought if self.thought else ""

    # def get_action(self) -> str:
    #     return self.action.value if self.action else "idle"

    # def get_target_x(self, default=0.0) -> float:
    #     ret = self.target_x if self.target_x is not None else default
    #     return float(ret)

    # def get_target_y(self, default=0.0) -> float:
    #     ret = self.target_y if self.target_y is not None else default
    #     return float(ret)


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
    
    def get_content(self):
        return self.get_message().content