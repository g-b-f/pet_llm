from abc import ABCMeta, abstractmethod
from typing import Any

from lib.types.other import RenderInfo

class DriverBase(metaclass=ABCMeta):
    """Base for a driver that displays information from the LLM and Brain"""

    @abstractmethod
    def render(self, info: RenderInfo) -> None:
        raise RuntimeError("Must be subclassed!")

    @abstractmethod
    def loop(self, info: RenderInfo) -> None:
        raise RuntimeError("Must be subclassed!")

    def __init__(self) -> None:
        self.running = True