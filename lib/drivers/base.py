from abc import ABCMeta, abstractmethod
from typing import Any

from lib.types.other import RenderInfo

class DriverBase(metaclass=ABCMeta):

    @abstractmethod
    def render(self, info:Any) -> None:
        raise RuntimeError("Must be subclassed!")

    @abstractmethod
    def loop(self, runtime: None|int, brain_info: RenderInfo) -> None:
        raise RuntimeError("Must be subclassed!")

    def __init__(self) -> None:
        self.running = True