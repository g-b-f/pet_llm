from abc import ABCMeta, abstractmethod

from lib.types.other import RenderInfo


class DriverBase(metaclass=ABCMeta):
    """Base for a driver that displays information from the LLM and Brain"""

    @abstractmethod
    def loop(self, info: RenderInfo) -> None:
        raise RuntimeError("Must be subclassed!")

    def __init__(self, end_time: int|float|None) -> None:
        self.running = True
        self.end_time = end_time