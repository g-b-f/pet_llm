from abc import ABCMeta
from typing import Any

class DriverBase(ABCMeta):
    def render(self, info:Any) -> None:
        raise RuntimeError("Must be subclassed!")

    def __init__(self) -> None:
        self.running = True