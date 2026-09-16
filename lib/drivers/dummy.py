import time

from lib.drivers.base import DriverBase
from lib.types.other import RenderInfo


class DummyDriver(DriverBase):
    """Driver that has no output"""

    def loop(self, info: RenderInfo) -> None:
        if self.end_time is None and self.runtime is not None:
            self.end_time = time.time() + self.runtime
        if self.end_time is not None and time.time() > self.end_time:
            self.running = False

    def __init__(self, runtime: int | float | None, *args, **kwargs):
        self.runtime = runtime
        super().__init__(None)
