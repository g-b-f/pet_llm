import json
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from lib.utils import get_logger

if TYPE_CHECKING:
    from lib.brain import Brain

logger = get_logger(__name__)

class EnvironmentalInfo(BaseModel):
    mouse: tuple[int,int] = Field(description="the location of the user's mouse")

class Action(Enum):
    move_to = "move_to"
    idle = "idle"
    swim_fast = "swim_fast"

class Direction(StrEnum):
    """A 6-point compass the pet can swim toward.

    Screen coordinates grow downward, so "north" is up (negative y) and
    "south" is down (positive y).
    """
    north = "north"
    northeast = "northeast"
    southeast = "southeast"
    south = "south"
    southwest = "southwest"
    northwest = "northwest"

    @property
    def vector(self) -> tuple[float, float]:
        """Unit (dx, dy) displacement for this direction in tank-local coords."""
        return _DIRECTION_VECTORS[self]

# Unit vectors for each compass point. Diagonals are normalised to length 1 so
# a given `distance` covers the same ground regardless of direction.
_DIRECTION_VECTORS: dict[Direction, tuple[float, float]] = {
    Direction.north: (0.0, -1.0),
    Direction.northeast: (0.7071, -0.7071),
    Direction.southeast: (0.7071, 0.7071),
    Direction.south: (0.0, 1.0),
    Direction.southwest: (-0.7071, 0.7071),
    Direction.northwest: (-0.7071, -0.7071),
}

def direction_vector(direction: Direction) -> tuple[float, float]:
    """Unit (dx, dy) displacement for a compass direction in tank-local coords."""
    return _DIRECTION_VECTORS[Direction(direction)]

class PetAction(BaseModel, use_enum_values=True):
    thought: str = Field(description="The thought process of the pet.")
    # action: Action = Field(description="The action to take.")
    direction: Direction = Field(
        description="The compass direction to swim toward: one of north, northeast, southeast, south, southwest, northwest."
    )
    distance: int = Field(
        description="How far to swim, in pixels, toward the chosen direction."
    )

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

class RenderInfo(BaseModel):
    """Stuff to render"""
    current_x: float
    current_y: float
    target_x: float
    target_y: float
    current_thought: str
    is_thinking: bool
    has_started_thinking: bool
    debug_info: dict

    @classmethod
    def from_brain(cls, brain: "Brain"):
        return cls(
            current_x= brain.current_x,
            current_y= brain.current_y,
            target_x= brain.target_x,
            target_y= brain.target_y,
            current_thought= brain.current_thought,
            is_thinking= brain.is_thinking,
            has_started_thinking = brain.current_thought != brain.config.thoughts.initial_thought,
            debug_info = brain.debug_info
        )