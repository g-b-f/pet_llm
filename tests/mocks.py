from pathlib import Path

from lib.inference.base import InferenceBase
from lib.types.config import BrainConfig
from lib.types.other import PetAction, RoleContent
from lib.brain import Brain

# TODO: swap things round so __init__ is passed `PetAction`,
# and there's also a `from_thoughts` method
class ScriptedInference(InferenceBase):
    def __init__(self, thoughts: list[RoleContent] | RoleContent):
        if not isinstance(thoughts, list): thoughts = [thoughts]
        self.thoughts = iter(thoughts)

    @classmethod
    def from_actions(cls, actions: list[PetAction] | PetAction):
        if not isinstance(actions, list): actions = [actions]
        return cls( [RoleContent.assistant(a.model_dump_json()) for a in actions] )

    def create_chat_completion(self, *args, **kwargs):
        return next(self.thoughts)

class MockInference(InferenceBase):
    """Returns an (invalid) chat completion, regardless of args"""
    def __init__(self): pass
    def create_chat_completion(self, *args, **kwargs):
        return RoleContent.assistant("")

class MockValidInference(InferenceBase):
    """Same as MockInference, but guaranteed to be valid"""
    def __init__(self):
        pass
    def create_chat_completion(self, *args, **kwargs):
        return RoleContent.assistant(
            PetAction(thought="hello", target_y=10, target_x=10).model_dump_json()
        )

class BlockingBrain(Brain):
    """returns inference until the iterator is empty, then blocks with is_thinking=True"""

    def __init__(
            self,
            thoughts: list[RoleContent] | RoleContent,
            model_path = Path("fake/path/model.gguf"),
            config = BrainConfig.model_construct(),
            ):
        super().__init__(model_path, config, ScriptedInference(thoughts))

    def _generate_decision(self, current_x: int, current_y: int):
        try:
            super()._generate_decision(current_x, current_y)
        except StopIteration:
            self.is_thinking = True
    