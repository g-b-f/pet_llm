import time

from lib.drivers.base import DriverBase
from lib.types.other import RenderInfo


class DummyDriver(DriverBase):
    """Driver that has no output"""

    def loop(self, info: RenderInfo) -> None:
        if self.end_time is not None and time.time() > self.end_time:
            self.running = False


    def __init__(self, runtime:int|float|None, *args, **kwargs):
        end_time = None if runtime is None else time.time() + runtime
        self.running = True
        super().__init__(end_time)