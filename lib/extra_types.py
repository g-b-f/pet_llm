from enum import Enum
from typing import Optional
import json

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
    
    def get_content(self):
        return self.get_message().content

    def get_action(self):
        return PetAction(**json.loads(self.get_content()))

class MemoryConfig(BaseModel):
    max_length: int = Field(5, description="The maximum number of messages to store in memory")

class ThoughtConfig(BaseModel):
    fallback_thought:str = Field(
        "Mind empty... drifting randomly.",
        description="The fallback thought to use when the LLM attempts an invalid thought"
        )
    initial_thought:str = Field("Waking up...", description="The initial thought to use when the pet is first created")
    initial_prompt:str = Field("Start exploring!", description="The initial prompt to use when the pet is first created")

class ParamsConfig(BaseModel):
    context_size: int = Field(2048, description="The context size for the model")
    temperature: float = Field(2, description="The temperature for the model")
    frequency_penalty: float = Field(
        1, description="The frequency penalty for the model"
    )
    presence_penalty: float = Field(1, description="The presence penalty for the model")
    repeat_penalty: float = Field(1, description="The repeat penalty for the model")
    min_p: float = Field(0.05, description="The minimum probability for the model")
    seed: int|None = None

class BrainConfig(BaseModel):
    thoughts: ThoughtConfig = Field(default_factory=ThoughtConfig.model_construct, description="The configuration for the pet's thoughts")
    params: ParamsConfig = Field(default_factory=ParamsConfig.model_construct, description="The configuration for the pet's model parameters")
    memory: MemoryConfig = Field(default_factory=MemoryConfig.model_construct, description="The configuration for the pet's memory")

class TankConfig(BaseModel):
    runtime: int|None = Field(None, description="The requested runtime of the simulation in seconds")
    screen_width: int = Field(800, description="The width of the screen in pixels")
    screen_height: int = Field(600, description="The height of the screen in pixels")

class SimulationConfig(BaseModel):
    tank: TankConfig = Field(default_factory=TankConfig.model_construct, description="The configuration for the tank simulation")
    brain: BrainConfig = Field(default_factory=BrainConfig.model_construct, description="The configuration for the pet's brain")

class BrainReport(BaseModel):
    iterations: int = Field(0, description="The total number of request/responses by the LLM")
    thought_loops: int = Field(0, description="The number of thought loops detected")
    empty_thoughts: int = Field(0, description="The number of times the LLM has had an empty thought")
    out_of_bounds_attempts: int = Field(0, description="The number of times the LLM has attempted to go out of bounds")
    malformed_json: int = Field(0, description="The number of times the LLM returned unparseable JSON")
    actual_runtime: None | float = Field(None, description="The actual runtime of the simulation in seconds")

class OutputReport(BaseModel):
    config: SimulationConfig
    report: BrainReport