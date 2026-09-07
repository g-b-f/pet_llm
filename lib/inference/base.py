from abc import ABCMeta, abstractmethod

from lib.types.config import ParamsConfig
from lib.types.other import RoleContent


class InferenceBase(metaclass=ABCMeta):
    """Base for an adapter between the brain and the inference libraries"""

    @abstractmethod
    def create_chat_completion(self, messages: list[RoleContent]) -> RoleContent:
        raise RuntimeError("Must be subclassed!")

    def __init__(self, config: ParamsConfig):
        self.config = config
