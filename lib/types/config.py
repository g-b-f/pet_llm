from pydantic import BaseModel, Field

# TODO: drop `params` from initial BrainConfig
# Maybe have InitialBrainConfig? 

class LossFunctionWeights(BaseModel):
    thought_loop: float
    empty_thought: float
    out_of_bounds: float
    malformed_json: float
    invalid_chars: float

class MemoryConfig(BaseModel):
    max_length: int = Field(5, description="The maximum number of messages to store in memory")

class ThoughtConfig(BaseModel):
    fallback_thought:str = Field(
        "Mind empty... drifting randomly.",
        description="The fallback thought to use when the LLM attempts an invalid thought"
        )
    initial_thought:str = Field("Waking up...", description="The initial thought to use when the pet is first created")
    initial_prompt:str = Field("Start exploring!", description="The initial prompt to use when the pet is first created")
    out_of_bounds_message:str = Field(
        "You can't leave the tank! Try a coordinate inside ({}, {}).",
        description="The system prompt to send the pet when it attempts to go out of bounds. "
        "Formatted with `.format(x_bounds, y_bounds)`"
        )
    system_prompt:str = Field(
            "You are a small pet living in a glass tank. "
            "Formulate a thought then pick coordinates inside the tank bounds to move toward. "
            "Keep moving and don't stay in the same place."
            # "Do not attempt to leave the bounds of the tank."
            "Adhere strictly to the requested JSON schema.\n"
            "Tank bounds: ({}, {}). "
            "Your position: ({}, {}).\n"
            # f"Your owner's finger is at {self.environment_info.mouse}"
        )

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

class TunerConfig(BaseModel):
    """high/low values for bayes optimisation"""
    temperature: tuple[float,float]
    frequency_penalty: tuple[float,float]
    presence_penalty: tuple[float,float]
    repeat_penalty: tuple[float,float]

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
