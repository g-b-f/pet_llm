"""Pydantic models describing the structure of a parsed optimisation log.

These models are produced by :mod:`parse_log` (project root) and describe the
hierarchy of an optimisation run: a log holds trials, each trial holds seed
runs, and each seed run holds a sequence of events.

Vibecoded with Qwen 3.8 27B
"""

import warnings
from datetime import datetime as dt
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lib.types.config import ParamsConfig
from lib.types.other import PetAction, Role, RoleContent


class EventType(StrEnum):
    thought = "thought"
    memory_cleared = "memory_cleared"
    malformed_json = "malformed_json"
    begin_simulation = "begin_simulation"

class MemoryClearReason(StrEnum):
    too_many_out_of_bounds = "too_many_out_of_bounds"
    thought_loop = "thought_loop"

class EventBase(BaseModel):
    run_id: int
    datetime:dt
    type: EventType
    model_config = ConfigDict(extra="allow", use_enum_values=True)

    def get_subclass(self):
        if self.type == "thought":
            return ThoughtEvent(**self.model_dump())
        elif self.type == "memory_cleared":
            return MemoryClearEvent(**self.model_dump())
        elif self.type == "malformed_json":
            return MalformedJSONEvent(**self.model_dump())

        raise ValueError("couldn't get type")


class ThoughtEvent(EventBase):
    """A parsed LLM decision"""

    thought: str = Field(description="The pet's thought, as returned by the LLM")
    target: tuple[int, int] | None = Field(None, description="The target coordinates, if any")
    type: EventType = EventType.thought

    @classmethod
    def from_action(cls, action: PetAction, run_id: int, datetime: dt | None = None):
        target = None
        if action.target_x is not None and action.target_y:
            target = (action.target_x, action.target_y)

        return cls(
            run_id=run_id,
            datetime=datetime or dt.now(),
            thought=action.thought,
            target=target
        )

    @classmethod
    def from_role_content(cls, data: RoleContent, run_id: int, datetime: dt | None = None):
        if data.role != Role.assistant:
            warnings.warn("Thought role should be assistant, got '{data.role}'")
        action = PetAction.model_validate_json(data.content)
        return cls.from_action(action, run_id=run_id, datetime=datetime or dt.now())

class MemoryClearEvent(EventBase):
    """Memory was cleared; `reason` is why."""

    type: Literal[EventType.memory_cleared] = EventType.memory_cleared # type: ignore[reportIncompatibleVariableOverride]
    reason: MemoryClearReason = Field(description="Why the memory was cleared")

    @classmethod
    def new(cls, reason: MemoryClearReason):
        return cls(run_id=-1, datetime=dt.now(), reason=reason)


class MalformedJSONEvent(EventBase):
    """The LLM returned unparseable JSON; `content` is the raw output."""

    type: Literal[EventType.malformed_json] = EventType.malformed_json # type: ignore[reportIncompatibleVariableOverride]
    content: str = Field(description="The raw, unparseable LLM output")

    @classmethod
    def from_role_content(cls, data: RoleContent, run_id: int, datetime: dt | None = None):
        if data.role.value != Role.assistant.value:
            warnings.warn(f"Malformed JSON role should be assistant, got '{data.role}'")

        content = repr(data.content).removeprefix("'").removesuffix("'")
        return cls(run_id=run_id, datetime=datetime or dt.now(), content=content)


class BeginSimulationEvent(EventBase):
    type: Literal[EventType.begin_simulation] = EventType.begin_simulation # type: ignore[reportIncompatibleVariableOverride]
    params: ParamsConfig

    @classmethod
    def new(cls, run_id: int, params: ParamsConfig, datetime: dt|None = None):
        return cls(
            run_id=run_id,
            params=params,
            datetime=datetime or dt.now()
        )


LogEvent = ThoughtEvent | MemoryClearEvent | MalformedJSONEvent | BeginSimulationEvent


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
