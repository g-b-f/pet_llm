import pytest

from lib.experience_buffer import ExperienceBuffer
from lib.types.other import RoleContent
from lib.types.config import MemoryConfig
from lib.memory import Memory


@pytest.fixture
def memory() -> Memory:
    mem = Memory(MemoryConfig(max_length=5))
    mem += RoleContent.user("Start exploring!")
    mem += RoleContent.assistant('{"thought": "hi", "action": "move_to", "target_x": 1, "target_y": 2}')
    return mem


@pytest.fixture
def buffer(memory: Memory) -> ExperienceBuffer:
    return ExperienceBuffer(memory, maxlen=4)


class TestExperienceBufferBasics:
    def test_init(self, buffer: ExperienceBuffer):
        assert buffer.is_empty
        assert buffer.length == 0
        assert buffer.maxlen == 4

    def test_record_snapshots_memory(self, buffer: ExperienceBuffer, memory: Memory):
        buffer.record(1.0)
        assert buffer.length == 1
        memory += RoleContent.user("later message")
        # snapshot is a copy: later memory mutations don't leak in
        _, snapshot = next(iter(buffer))
        assert len(snapshot) == 2

    def test_record_multiple(self, buffer: ExperienceBuffer):
        buffer.record(1.0)
        buffer.record(-1.0)
        assert buffer.length == 2

    def test_maxlen_eviction(self, buffer: ExperienceBuffer):
        for score in (0.1, 0.2, 0.3, 0.4, 0.9):
            buffer.record(score)
        assert buffer.length == 4
        scores = [score for score, _ in buffer]
        assert 0.1 not in scores
        assert 0.9 in scores

    def test_is_full(self, buffer: ExperienceBuffer):
        for _ in range(4):
            buffer.record(1.0)
        assert buffer.is_full

    def test_len(self, buffer: ExperienceBuffer):
        buffer.record(1.0)
        assert len(buffer) == 1

    def test_clear(self, buffer: ExperienceBuffer):
        buffer.record(1.0)
        buffer.clear()
        assert buffer.is_empty


class TestExportPositive:
    def test_export_format(self, buffer: ExperienceBuffer):
        buffer.record(1.0)
        exported = buffer.export_positive()
        assert len(exported) == 1
        messages = exported[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "Start exploring!"}

    def test_export_filters_below_threshold(self, buffer: ExperienceBuffer):
        buffer.record(0.1)  # below default threshold
        buffer.record(1.0)  # above
        exported = buffer.export_positive()
        assert len(exported) == 1

    def test_export_custom_threshold(self, buffer: ExperienceBuffer):
        buffer.record(0.8)
        exported = buffer.export_positive(threshold=0.9)
        assert exported == []
        exported = buffer.export_positive(threshold=0.7)
        assert len(exported) == 1

    def test_export_consumed(self, buffer: ExperienceBuffer):
        buffer.record(1.0)
        buffer.export_positive()
        assert buffer.is_empty
        assert buffer.export_positive() == []

    def test_export_keeps_negative_experiences(self, buffer: ExperienceBuffer):
        buffer.record(-1.0)
        assert buffer.export_positive() == []
        assert buffer.length == 1

    def test_export_empty_buffer(self, buffer: ExperienceBuffer):
        assert buffer.export_positive() == []

    def test_export_custom_system_prompt(self, buffer: ExperienceBuffer):
        buffer.record(1.0)
        exported = buffer.export_positive(system_prompt="Be a fish")
        assert exported[0]["messages"][0]["content"] == "Be a fish"
