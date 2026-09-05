from lib.drivers.base import DriverBase
from lib.types.other import RenderInfo
import time

class DummyDriver(DriverBase):
    """Driver that has no output"""

    def render(self, info: RenderInfo):
        return

    def loop(self, info: RenderInfo) -> None:
        if self.end_time is not None and time.time() < self.end_time:
            self.running = False
        time.sleep(0.01)


    def __init__(self, runtime:int|None, *args, **kwargs):
        if runtime is None:
            self.end_time = None
        else:
            self.end_time = time.time() + runtime

        self.running = True