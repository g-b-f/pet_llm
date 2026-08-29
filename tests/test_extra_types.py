import pytest

from lib.extra_types import (
    Action,
    ChatCompletionResponse,
    PetAction,
    Role,
    RoleContent,
)


class TestPetAction:
    def test_valid_creation(self):
        action = PetAction(
            thought="swim around",
            action=Action.move_to,
            target_x=50,
            target_y=100,
        )
        assert action.thought == "swim around"
        assert action.action == "move_to"
        assert action.target_x == 50
        assert action.target_y == 100

    def test_enum_values_stored(self):
        action = PetAction(
            thought="test",
            action=Action.idle,
            target_x=0,
            target_y=0,
        )
        assert action.action == "idle"

    def test_json_round_trip(self):
        action = PetAction(
            thought="round trip",
            action=Action.swim_fast,
            target_x=10,
            target_y=20,
        )
        json_str = action.model_dump_json()
        restored = PetAction(**action.model_validate_json(json_str).model_dump())
        assert restored.thought == action.thought
        assert restored.target_x == action.target_x


class TestRoleContent:
    def test_user_factory(self):
        msg = RoleContent.user("hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_system_factory(self):
        msg = RoleContent.system("be a pet")
        assert msg.role == "system"
        assert msg.content == "be a pet"

    def test_model_dump(self):
        msg = RoleContent.user("test")
        dumped = msg.model_dump()
        assert dumped == {"role": "user", "content": "test"}


class TestChatCompletionResponse:
    @pytest.fixture
    def response(self) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"thought":"hi"}'},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def test_get_message(self, response: ChatCompletionResponse):
        msg = response.get_message()
        assert msg.role == "assistant"
        assert msg.content == '{"thought":"hi"}'

    def test_get_content(self, response: ChatCompletionResponse):
        content = response.get_content()
        assert content == '{"thought":"hi"}'
