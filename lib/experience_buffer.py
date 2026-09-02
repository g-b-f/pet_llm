import threading
from collections import deque
from typing import Iterator

from lib.types.other import RoleContent
from lib.memory import Memory
from lib.utils import get_logger

logger = get_logger(__name__)

DEFAULT_REWARD_THRESHOLD = 0.5
DEFAULT_SYSTEM_PROMPT = "You are a small pet living in a glass tank."

# Reward rules: what the experience buffer considers good or bad behaviour.
REWARD_MOVE_TO = 1.0
REWARD_IDLE = 0.0
REWARD_OUT_OF_BOUNDS = -1.0
REWARD_FALLBACK = -0.5


class ExperienceBuffer:
    """Collects rewarded interactions from a ``Memory`` for LoRA fine-tuning.

    Each *experience* is a scored snapshot of the conversation: a copy of the
    messages in the connected ``Memory`` plus a numerical ``reward_score``
    derived from the action the pet just took. Experiences with
    ``reward_score >= threshold`` can be exported as standard chat messages
    (``list[dict]``) suitable for a fine-tuning dataset.
    """

    def __init__(
        self,
        memory: Memory,
        maxlen: int = 64,
        threshold: float = DEFAULT_REWARD_THRESHOLD,
    ) -> None:
        self.memory = memory
        self.threshold = threshold
        self._buffer: deque[tuple[float, list[RoleContent]]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, reward_score: float) -> None:
        """Snapshot the current memory contents under ``reward_score``."""
        with self._lock:
            self._buffer.append((reward_score, list(self.memory._memory_queue)))
        logger.debug(
            "recorded experience (reward=%s, size=%s)", reward_score, self.length
        )

    def export_positive(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        threshold: float | None = None,
    ) -> list[dict]:
        """Export positive experiences in standard chat message format.

        Each exported example is a full chat: a fresh system prompt followed by
        the snapshot's messages. Only experiences with
        ``reward_score >= threshold`` are included; consumed experiences are
        removed from the buffer.
        """
        threshold = self.threshold if threshold is None else threshold
        with self._lock:
            keep: list[tuple[float, list[RoleContent]]] = []
            exported: list[dict] = []
            for score, snapshot in self._buffer:
                if score >= threshold:
                    messages = [RoleContent.system(system_prompt)] + snapshot
                    exported.append(
                        {"messages": [msg.model_dump() for msg in messages]}
                    )
                else:
                    keep.append((score, snapshot))
            self._buffer.clear()
            self._buffer.extend(keep)
        if exported:
            logger.info("exported %s positive experiences", len(exported))
        return exported

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    @property
    def length(self) -> int:
        return len(self._buffer)

    @property
    def maxlen(self) -> int | None:
        return self._buffer.maxlen

    @property
    def is_empty(self) -> bool:
        return self.length == 0

    @property
    def is_full(self) -> bool:
        return self.length == self.maxlen

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[tuple[float, list[RoleContent]]]:
        return iter(list(self._buffer))
