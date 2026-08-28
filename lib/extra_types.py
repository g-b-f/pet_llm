from pydantic import BaseModel, Field
from enum import Enum


class EnvironmentalInfo(BaseModel):
    mouse: tuple[int,int] = Field(description="the location of the user's mouse")

class Action(Enum):
    move_to = "move_to"
    idle = "idle"
    swim_fast = "swim_fast"

class PetAction(BaseModel):
    thought: str = Field(description="The thought process of the pet.")
    action: Action = Field(description="The action to take.")
    target_x: int = Field(description="Target X coordinate.")
    target_y: int = Field(description="Target Y coordinate.")

    def get_thought(self) -> str:
        return self.thought if self.thought else ""

    def get_action(self) -> str:
        return self.action.value if self.action else "idle"

    def get_target_x(self, default=0.0) -> float:
        ret = self.target_x if self.target_x is not None else default
        return float(ret)

    def get_target_y(self, default=0.0) -> float:
        ret = self.target_y if self.target_y is not None else default
        return float(ret)
