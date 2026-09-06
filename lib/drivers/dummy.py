from lib.drivers.base import DriverBase
from lib.types.other import RenderInfo
import time

class DummyDriver(DriverBase):
    """Driver that has no output"""

    def loop(self, info: RenderInfo) -> None:
        if self.end_time is not None and time.time() > self.end_time:
            self.running = False


    def __init__(self, runtime:int|float|None, *args, **kwargs):
        if runtime is None:
            end_time = None
        else:
            end_time = time.time() + runtime
        self.running = True
        super().__init__(end_time)