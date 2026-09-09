from pathlib import Path
import random

from lib.brain import Brain
from lib.inference.base import InferenceBase
from lib.types.config import BrainConfig
from lib.types.other import PetAction, RoleContent

random.seed(42)

class ScriptedInference(InferenceBase):
    def __init__(self, actions: list[PetAction] | PetAction):
        if not isinstance(actions, list):
            actions = [actions]
        thoughts = [RoleContent.assistant(a.model_dump_json()) for a in actions]
        self.thoughts = iter(thoughts)

    @classmethod
    def from_thoughts(cls, thoughts: list[RoleContent] | RoleContent):
        if not isinstance(thoughts, list):
            thoughts = [thoughts]
        obj = cls([])
        obj.thoughts = iter(thoughts)
        return obj

    @classmethod
    def infinite(cls, max_target = 100, limit: int|None = None):
        """Returns an inference object that will infinitely give valid responses"""
        obj = cls([])

        def response():
            i = 0
            while limit is None or i < limit:
                target_x = target_y = random.randint(0, max_target)
                thought = f"thought {i}"
                i += 1
                action = PetAction(thought=thought, target_x=target_x, target_y=target_y)
                yield RoleContent.assistant(action.model_dump_json())

        obj.thoughts = response()

        return obj

    def create_chat_completion(self, *args, **kwargs):
        return next(self.thoughts)


class MockInference(InferenceBase):
    """Returns an (invalid) chat completion, regardless of args"""

    def __init__(self):
        pass

    def create_chat_completion(self, *args, **kwargs):
        return RoleContent.assistant("")


class MockValidInference(MockInference):
    """Same as MockInference, but guaranteed to be valid"""

    def create_chat_completion(self, *args, **kwargs):
        return RoleContent.assistant(
            PetAction(thought="hello", target_y=10, target_x=10).model_dump_json()
        )

class InfiniteBrain(Brain):
    def __init__(self, config = BrainConfig.model_construct(), bounds = (500,500)):
        super().__init__(config, bounds, ScriptedInference.infinite())

class BlockingBrain(Brain):
    """returns inference until the iterator is empty, then blocks with is_thinking=True"""

    def __init__(
        self,
        actions: list[PetAction] | PetAction,
        config=BrainConfig.model_construct(),
        bounds = (500, 500)
    ):
        super().__init__(config, bounds, ScriptedInference(actions))
    
    @classmethod
    def from_thoughts(
        cls,
        thoughts: list[RoleContent] | RoleContent,
        config=BrainConfig.model_construct(),
        bounds = (500, 500)
    ):
        if not isinstance(thoughts, list):
            thoughts = [thoughts]
        obj = cls([], bounds, config)
        obj.inference = ScriptedInference.from_thoughts(thoughts)
        return obj

    @classmethod
    def generate_valid_thoughts(
        cls,
        amount: int,
        config=BrainConfig.model_construct(),
        bounds = (500, 500)
    ):
        obj = cls([], config, bounds)
        obj.inference = ScriptedInference.infinite(limit=amount)
        return obj

    def _generate_decision(self, current_x: int, current_y: int):
        try:
            super()._generate_decision(current_x, current_y)
        except StopIteration:
            self.is_thinking = True
