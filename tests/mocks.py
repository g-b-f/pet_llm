from lib.inference.base import InferenceBase
from lib.types.other import RoleContent


class ScriptedInference(InferenceBase):
    def __init__(self, thoughts: list[RoleContent] | RoleContent):
        if not isinstance(thoughts, list): thoughts = [thoughts]
        self.thoughts = iter(thoughts)

    def create_chat_completion(self, *args, **kwargs):
        return next(self.thoughts)

class MockInference(InferenceBase):
    def __init__(self): pass
    def create_chat_completion(self, *args, **kwargs):
        return RoleContent.assistant("")