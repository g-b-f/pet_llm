
class MemoryHandlerError(Exception):
    pass


class ThoughtLoopError(MemoryHandlerError):
    def __init__(self, last_thought, *args) -> None:
        self.last_thought = last_thought
        super().__init__(*args)
