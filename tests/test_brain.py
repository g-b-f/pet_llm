from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.brain import Brain
from lib.types.other import Action, EnvironmentalInfo, PetAction
from lib.types.config import BrainConfig


@pytest.fixture
def brain_config() -> BrainConfig:
    return BrainConfig.model_construct()


@pytest.fixture
def brain(brain_config: BrainConfig) -> Brain:
    return Brain(Path("fake/model/path.gguf"), brain_config)


@pytest.fixture
def awake_brain(brain: Brain) -> Brain:
    with patch("lib.brain.Llama") as mock_llama:
        mock_llama.return_value = MagicMock()
        brain.wake_up((100, 100))
    return brain


@pytest.fixture
def env_info() -> EnvironmentalInfo:
    return EnvironmentalInfo(mouse=(0, 0))


def _make_llm_response(thought: str, x: int, y: int) -> dict:
    action = PetAction(thought=thought, action=Action.move_to, target_x=x, target_y=y)
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": action.model_dump_json()},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class TestWakeUp:
    def test_awake_after_wake_up(self, awake_brain: Brain):
        assert awake_brain.awake

    def test_initial_position_centered(self, awake_brain: Brain):
        assert awake_brain.current_x == 50.0
        assert awake_brain.current_y == 50.0

    def test_initial_thought(self, awake_brain: Brain):
        assert (
            awake_brain.current_thought == awake_brain.config.thoughts.initial_thought
        )

    def test_initial_memory_has_one_entry(self, awake_brain: Brain):
        assert awake_brain.memory.length == 1

    def test_not_thinking_initially(self, awake_brain: Brain):
        assert not awake_brain.is_thinking

    def test_iterations_zero(self, awake_brain: Brain):
        assert awake_brain.iterations == 0

    def test_oob_count_zero(self, awake_brain: Brain):
        assert awake_brain.current_oob_count == 0


class TestUpdate:
    def test_pet_moves_toward_target(
        self, awake_brain: Brain, env_info: EnvironmentalInfo
    ):
        awake_brain.target_x = 100.0
        awake_brain.target_y = 50.0
        initial_x = awake_brain.current_x
        awake_brain.update(env_info)
        assert awake_brain.current_x > initial_x

    def test_pet_speed(self, awake_brain: Brain, env_info: EnvironmentalInfo):
        awake_brain.target_x = 100.0
        awake_brain.target_y = 50.0
        initial_x = awake_brain.current_x
        awake_brain.update(env_info)
        assert abs(awake_brain.current_x - (initial_x + Brain.PET_SPEED)) < 0.01

    def test_queued_decision_applied(
        self, awake_brain: Brain, env_info: EnvironmentalInfo
    ):
        decision = PetAction(
            thought="new thought", action=Action.move_to, target_x=10, target_y=20
        )
        awake_brain.result_queue.put(decision)
        awake_brain.update(env_info)
        assert awake_brain.current_thought == "new thought"
        assert awake_brain.target_x == 10
        assert awake_brain.target_y == 20

    def test_arrival_triggers_decision_request(
        self, awake_brain: Brain, env_info: EnvironmentalInfo
    ):
        awake_brain.target_x = awake_brain.current_x
        awake_brain.target_y = awake_brain.current_y
        with patch.object(awake_brain, "request_decision_async") as mock_req:
            awake_brain.update(env_info)
            mock_req.assert_called_once()


class TestFallback:
    def test_fallback_queues_decision(self, awake_brain: Brain):
        awake_brain._fallback()
        assert not awake_brain.result_queue.empty()
        decision = awake_brain.result_queue.get()
        assert decision.thought == awake_brain.config.thoughts.fallback_thought

    def test_fallback_within_bounds(self, awake_brain: Brain):
        awake_brain._fallback()
        decision = awake_brain.result_queue.get()
        assert 0 <= decision.target_x <= awake_brain.x_bounds
        assert 0 <= decision.target_y <= awake_brain.y_bounds


class TestRequestDecisionAsync:
    def test_asserts_if_not_awake(self, brain: Brain):
        with pytest.raises(AssertionError, match="still asleep"):
            brain.request_decision_async(0, 0)

    def test_skips_if_already_thinking(self, awake_brain: Brain):
        awake_brain.is_thinking = True
        with patch("lib.brain.threading.Thread") as mock_thread:
            awake_brain.request_decision_async(50, 50)
            mock_thread.assert_not_called()

    def test_starts_thread(self, awake_brain: Brain):
        with patch("lib.brain.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            awake_brain.request_decision_async(50, 50)
            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()

    def test_sets_thinking_flag(self, awake_brain: Brain):
        with patch("lib.brain.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            awake_brain.request_decision_async(50, 50)
            assert awake_brain.is_thinking


class TestTargetOutOfBounds:
    @pytest.mark.parametrize(
        ("target_x", "target_y"), [(101, 50), (-1, 50), (50, 101), (50, -1)]
    )
    def test_target_out_of_bounds(
        self, awake_brain: Brain, target_x: int, target_y: int
    ):
        action = PetAction(
            thought="t", action=Action.move_to, target_x=target_x, target_y=target_y
        )
        assert awake_brain.target_out_of_bounds(action)

    @pytest.mark.parametrize(
        ("target_x", "target_y"), [(50, 50), (100, 100), (0, 0), (100, 0), (0, 100)]
    )
    def test_target_in_bounds(self, awake_brain: Brain, target_x: int, target_y: int):
        action = PetAction(
            thought="t", action=Action.move_to, target_x=target_x, target_y=target_y
        )
        assert not awake_brain.target_out_of_bounds(action)


class TestGenerateDecision:
    def _setup_brain_for_generation(
        self, brain: Brain, response_thought: str = "hello", x: int = 10, y: int = 20
    ):
        brain.llm = MagicMock()
        brain.llm.create_chat_completion.return_value = _make_llm_response(
            response_thought, x, y
        )
        brain._generate_decision(50, 50)

    def test_successful_decision_queued(self, awake_brain: Brain):
        self._setup_brain_for_generation(awake_brain)
        assert not awake_brain.result_queue.empty()
        decision = awake_brain.result_queue.get()
        assert decision.thought == "hello"

    def test_increments_iterations(self, awake_brain: Brain):
        self._setup_brain_for_generation(awake_brain)
        assert awake_brain.iterations == 1

    def test_clears_thinking_flag(self, awake_brain: Brain):
        awake_brain.is_thinking = True
        self._setup_brain_for_generation(awake_brain)
        assert not awake_brain.is_thinking

    def test_oob_decision_not_queued(self, awake_brain: Brain):
        self._setup_brain_for_generation(awake_brain, x=999, y=999)
        assert awake_brain.result_queue.empty()

    def test_oob_increments_oob_count(self, awake_brain: Brain):
        self._setup_brain_for_generation(awake_brain, x=999, y=999)
        assert awake_brain.current_oob_count == 1

    def test_max_oob_triggers_fallback(self, awake_brain: Brain):
        awake_brain.current_oob_count = Brain.MAX_OOB_COUNT - 1
        self._setup_brain_for_generation(awake_brain, x=999, y=999)
        assert awake_brain.current_oob_count == 0
        assert not awake_brain.result_queue.empty()
        decision = awake_brain.result_queue.get()
        assert decision.thought == awake_brain.config.thoughts.fallback_thought

    def test_memory_cleared_on_max_oob(self, awake_brain: Brain):
        awake_brain.current_oob_count = Brain.MAX_OOB_COUNT - 1
        self._setup_brain_for_generation(awake_brain, x=999, y=999)
        assert awake_brain.memory.length == 0

    def test_memory_updated_on_success(self, awake_brain: Brain):
        initial_len = awake_brain.memory.length
        self._setup_brain_for_generation(awake_brain)
        assert awake_brain.memory.length > initial_len

    def test_malformed_json_resets_thinking_and_counts(self, awake_brain: Brain):
        response = _make_llm_response("hello", 10, 20)
        response["choices"][0]["message"]["content"] = "not valid json {{"
        awake_brain.llm = MagicMock()
        awake_brain.llm.create_chat_completion.return_value = response

        awake_brain.is_thinking = True
        awake_brain._generate_decision(50, 50)

        assert not awake_brain.is_thinking
        assert awake_brain.report.malformed_json == 1
        assert not awake_brain.result_queue.empty()
        decision = awake_brain.result_queue.get()
        assert decision.thought == awake_brain.config.thoughts.fallback_thought
