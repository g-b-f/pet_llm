"""Pydantic models describing the structure of a parsed optimisation log.

These models are produced by :mod:`parse_log` (project root) and describe the
hierarchy of an optimisation run: a log holds trials, each trial holds seed
runs, and each seed run holds a sequence of events.

Vibecoded with Qwen 3.8 27B
"""

from enum import StrEnum
from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime as dt
from lib.types.config import ParamsConfig


class EventType(StrEnum):
    thought = "thought"
    oob_reset = "oob_reset"
    thought_loop = "thought_loop"
    malformed_json = "malformed_json"

class EventBase(BaseModel):
    run_id: int
    datetime: dt
    type: EventType
    model_config = ConfigDict(extra="allow", use_enum_values=True)

    def get_subclass(self):
        if self.type == "thought":
            return ThoughtEvent(**self.model_dump())
        elif self.type == "oob_reset":
            return OOBResetEvent(**self.model_dump())
        elif self.type == "thought_loop":
            return ThoughtLoopEvent(**self.model_dump())
        elif self.type == "malformed_json":
            return MalformedJSONEvent(**self.model_dump())
        
        raise ValueError("couldn't get type")


class ThoughtEvent(EventBase):
    """A parsed LLM decision.

    ``target`` is only set when the decision's target was out of bounds (the
    brain logs the coordinates in that case); otherwise it is ``None``.
    """

    type= EventType.thought
    thought: str = Field(description="The pet's thought, as returned by the LLM")
    target: tuple[int, int] | None = Field(
        default=None, description="The target coordinates, if any"
        )


class ScoredThoughtEvent(ThoughtEvent):
    """A thought event with a score of how good the thought was"""
    loss: float | None = Field(description="The result of applying the loss function to this thought")


class OOBResetEvent(EventBase):
    """Memory was cleared after too many consecutive out-of-bounds targets."""
    type = EventType.oob_reset


class ThoughtLoopEvent(EventBase):
    """Memory was cleared after the same thought repeated (a thought loop)."""
    type = EventType.thought_loop

class MalformedJSONEvent(EventBase):
    """The LLM returned unparseable JSON; ``content`` is the raw output."""
    type = EventType.malformed_json
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

    params: ParamsConfig | None = Field(default=None, description="The shared parameters, if logged")
    seeds: list[SeedRun] = Field(default_factory=list)


class OptimisationLog(BaseModel):
    """The full parsed optimisation log."""

    runtime: int | None = Field(default=None, description="The requested runtime in seconds")
    trials: list[LogTrial] = Field(default_factory=list)
