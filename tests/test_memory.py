import json

import pytest

from lib.extra_types import Action, MemoryConfig, PetAction, RoleContent
from lib.memory import Memory, ThoughtLoopError


@pytest.fixture
def memory() -> Memory:
    return Memory(MemoryConfig(max_length=3))


@pytest.fixture
def sample_action() -> PetAction:
    return PetAction(
        thought="I want to swim",
        action=Action.move_to,
        target_x=10,
        target_y=20,
    )


class TestMemoryBasics:
    def test_init(self, memory: Memory):
        assert memory.config.max_length == 3
        assert memory.length == 0
        assert memory.is_empty
        assert not memory.is_full

    def test_add_message(self, memory: Memory):
        msg = RoleContent.user("Hello")
        result = memory + msg
        assert result is memory
        assert memory.length == 1
        assert not memory.is_empty

    def test_len(self, memory: Memory):
        assert len(memory) == 0
        memory += RoleContent.user("A")
        assert len(memory) == 1

    def test_maxlen_eviction(self, memory: Memory):
        for i in range(5):
            memory += RoleContent.user(f"msg{i}")
        assert memory.length == 3
        assert memory.is_full
        messages = memory.get_messages("system")
        contents = [m["content"] for m in messages]
        assert "msg0" not in contents
        assert "msg1" not in contents
        assert "msg2" in contents
        assert "msg4" in contents

    def test_clear(self, memory: Memory):
        memory += RoleContent.user("Hello")
        memory.clear()
        assert memory.length == 0
        assert memory.is_empty


class TestGetMessages:
    def test_system_prompt_prepended(self, memory: Memory):
        memory += RoleContent.user("Hello")
        messages = memory.get_messages("Be a pet")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be a pet"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_empty_memory_returns_only_system(self, memory: Memory):
        messages = memory.get_messages("Be a pet")
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


class TestGetAction:
    def test_valid_action(self, memory: Memory, sample_action: PetAction):
        memory += RoleContent(role="assistant", content=sample_action.model_dump_json())
        action = memory.get_action(0)
        assert action is not None
        assert action.thought == "I want to swim"
        assert action.target_x == 10

    def test_invalid_json_returns_none(self, memory: Memory):
        memory += RoleContent.user("not json")
        action = memory.get_action(0)
        assert action is None

    def test_non_action_json_returns_none(self, memory: Memory):
        memory += RoleContent.user(json.dumps({"foo": "bar"}))
        action = memory.get_action(0)
        assert action is None


class TestSupervise:
    def test_not_full_no_error(self, memory: Memory, sample_action: PetAction):
        memory += RoleContent(role="assistant", content=sample_action.model_dump_json())
        memory.supervise()

    def test_matching_thoughts_raise_loop_error(
        self, memory: Memory, sample_action: PetAction
    ):
        for _ in range(3):
            memory += RoleContent(
                role="assistant", content=sample_action.model_dump_json()
            )
        with pytest.raises(ThoughtLoopError) as exc_info:
            memory.supervise()
        assert exc_info.value.last_thought == "I want to swim"

    def test_different_thoughts_no_error(self, memory: Memory):
        for i in range(3):
            action = PetAction(
                thought=f"thought {i}",
                action="move_to",
                target_x=i,
                target_y=i,
            )
            memory += RoleContent(role="assistant", content=action.model_dump_json())
        memory.supervise()

    def test_none_actions_no_error(self, memory: Memory):
        memory += RoleContent.user("not json")
        memory += RoleContent.user("also not json")
        memory += RoleContent.user("still not json")
        memory.supervise()
