from pydantic import BaseModel, Field
from lib.types.config import SimulationConfig, LossFunctionWeights, TunerConfig, ParamsConfig

class BrainReport(BaseModel):
    iterations: int = Field(0, description="The total number of request/responses by the LLM")
    thought_loops: int = Field(0, description="The number of thought loops detected")
    empty_thoughts: int = Field(0, description="The number of times the LLM has had an empty thought")
    out_of_bounds_attempts: int = Field(0, description="The number of times the LLM has attempted to go out of bounds")
    malformed_json: int = Field(0, description="The number of times the LLM returned unparseable JSON")
    non_alphanumeric: int = Field(0, description="The number of times the LLM has had a non-alphanumeric thought")
    actual_runtime: None | float = Field(None, description="The actual runtime of the simulation in seconds")

class OutputReport(BaseModel):
    config: SimulationConfig
    report: BrainReport

class Trial(BaseModel):
    params: ParamsConfig
    report: BrainReport

class StudyReport(BaseModel):
    comments: str = Field("")
    tuner_config: TunerConfig
    loss_function_weights: LossFunctionWeights
    simulation_config: SimulationConfig
    trials: list[Trial]
