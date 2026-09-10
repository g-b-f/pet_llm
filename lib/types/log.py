"""Pydantic models describing the structure of a parsed optimisation log.

These models are produced by :mod:`parse_log` (project root) and describe the
hierarchy of an optimisation run: a log holds trials, each trial holds seed
runs, and each seed run holds a sequence of events.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class Params(BaseModel):
    """Sampling parameters shared by every seed in one optimisation trial."""

    context_size: int = Field(description="The context size for the model")
    temperature: float = Field(description="The temperature for the model")
    frequency_penalty: float = Field(description="The frequency penalty for the model")
    presence_penalty: float = Field(description="The presence penalty for the model")
    repeat_penalty: float = Field(description="The repeat penalty for the model")
    min_p: float = Field(description="The minimum probability for the model")


class ThoughtEvent(BaseModel):
    """A parsed LLM decision.

    ``target`` is only set when the decision's target was out of bounds (the
    brain logs the coordinates in that case); otherwise it is ``None``.
    """

    type: Literal["thought"] = "thought"
    thought: str = Field(description="The pet's thought, as returned by the LLM")
    target: tuple[int, int] | None = Field(
        default=None, description="The out-of-bounds target coordinates, if any"
    )


class OOBResetEvent(BaseModel):
    """Memory was cleared after too many consecutive out-of-bounds targets."""

    type: Literal["oob_reset"] = "oob_reset"


class ThoughtLoopEvent(BaseModel):
    """Memory was cleared after the same thought repeated (a thought loop)."""

    type: Literal["thought_loop"] = "thought_loop"


class MalformedJSONEvent(BaseModel):
    """The LLM returned unparseable JSON; ``content`` is the raw output."""

    type: Literal["malformed_json"] = "malformed_json"
    content: str = Field(description="The raw, unparseable LLM output")


LogEvent = Annotated[
    Union[ThoughtEvent, OOBResetEvent, ThoughtLoopEvent, MalformedJSONEvent],
    Field(discriminator="type"),
]


class SeedRun(BaseModel):
    """One simulation run for a single seed."""

    seed: int = Field(description="The seed used for this run")
    loss: float | None = Field(default=None, description="The loss for this run, if logged")
    events: list[LogEvent] = Field(default_factory=list)


class LogTrial(BaseModel):
    """A set of seed runs that share the same parameters."""

    params: Params | None = Field(default=None, description="The shared parameters, if logged")
    seeds: list[SeedRun] = Field(default_factory=list)


class OptimisationLog(BaseModel):
    """The full parsed optimisation log."""

    runtime: int | None = Field(default=None, description="The requested runtime in seconds")
    trials: list[LogTrial] = Field(default_factory=list)
