from pydantic import BaseModel, Field

class PetAction(BaseModel):
    thought: str = Field(description="The thought process of the pet.")
    action: str = Field(description="The action to take.", json_schema_extra={"enum": ["move_to", "idle", "swim_fast"]})
    target_x: int = Field(description="Target X coordinate.")
    target_y: int = Field(description="Target Y coordinate.")

    def get_thought(self) -> str:
        return self.thought if self.thought else ""

    def get_action(self) -> str:
        return self.action if self.action else "idle"

    def get_target_x(self, default=0.0) -> float:
        ret = self.target_x if self.target_x is not None else default
        return float(ret)

    def get_target_y(self, default=0.0) -> float:
        ret = self.target_y if self.target_y is not None else default
        return float(ret)
