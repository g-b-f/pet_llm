from unittest.mock import MagicMock, patch

import pytest

from lib.brain import Brain
from lib.dynamic_adapter_llm import DynamicAdapterLLM
from lib.extra_types import Action, BrainConfig, EnvironmentalInfo, PetAction


@pytest.fixture
def learning_config() -> BrainConfig:
    config = BrainConfig.model_construct()
    config.learning.enabled = True
    return config


@pytest.fixture
def awake_brain(learning_config: BrainConfig) -> Brain:
    brain = Brain("fake/model/path.gguf", learning_config)
    with (
        patch("lib.brain.DynamicAdapterLLM") as mock_llm_cls,
        patch("lib.brain.BackgroundTrainer") as mock_trainer_cls,
    ):
        mock_llm_cls.return_value = MagicMock(spec=DynamicAdapterLLM)
        mock_trainer_cls.return_value = MagicMock()
        brain.wake_up((100, 100))
    return brain


@pytest.fixture
def env_info() -> EnvironmentalInfo:
    return EnvironmentalInfo(mouse=(0, 0))


def _llm_response(thought: str, x: int, y: int) -> dict:
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


class TestLearningSetup:
    def test_dynamic_llm_used_when_learning(self, awake_brain: Brain):
        assert awake_brain.llm is not None
        assert awake_brain.experience_buffer is not None
        assert awake_brain.trainer is not None

    def test_trainer_started(self, awake_brain: Brain):
        awake_brain.trainer.start.assert_called_once()

    def test_buffer_connected_to_memory(self, awake_brain: Brain):
        assert awake_brain.experience_buffer.memory is awake_brain.memory


class TestRewardRecording:
    def _run_decision(self, brain: Brain, thought: str, x: int, y: int):
        brain.llm.create_chat_completion.return_value = _llm_response(thought, x, y)
        brain.is_thinking = True
        brain._generate_decision(int(brain.current_x), int(brain.current_y))

    def test_good_move_rewarded(self, awake_brain: Brain):
        self._run_decision(awake_brain, "swimming over there", 10, 10)
        scores = [score for score, _ in awake_brain.experience_buffer]
        assert scores == [awake_brain.config.learning.reward_move_to]

    def test_empty_thought_discouraged(self, awake_brain: Brain):
        self._run_decision(awake_brain, "   ", 10, 10)
        scores = [score for score, _ in awake_brain.experience_buffer]
        assert scores == [awake_brain.config.learning.reward_empty_thought]

    def test_out_of_bounds_discouraged(self, awake_brain: Brain):
        self._run_decision(awake_brain, "escaping!", 9999, 9999)
        scores = [score for score, _ in awake_brain.experience_buffer]
        assert scores == [awake_brain.config.learning.reward_out_of_bounds]

    def test_empty_thought_oob_gets_worse_reward(self, awake_brain: Brain):
        self._run_decision(awake_brain, "", 9999, 9999)
        scores = [score for score, _ in awake_brain.experience_buffer]
        expected = min(
            awake_brain.config.learning.reward_out_of_bounds,
            awake_brain.config.learning.reward_empty_thought,
        )
        assert scores == [expected]

    def test_empty_thought_still_moves(self, awake_brain: Brain):
        self._run_decision(awake_brain, "", 10, 10)
        assert not awake_brain.result_queue.empty()

    def test_positive_rewards_exportable(self, awake_brain: Brain):
        self._run_decision(awake_brain, "good thought", 10, 10)
        self._run_decision(awake_brain, "", 9999, 9999)  # bad: not exported
        exported = awake_brain.experience_buffer.export_positive()
        assert len(exported) == 1


class TestAdapterHotSwap:
    def test_update_applies_completed_adapter(
        self, awake_brain: Brain, env_info, tmp_path
    ):
        adapter = tmp_path / "adapter_1.gguf"
        adapter.touch()
        awake_brain.trainer.get_completed_adapter.return_value = adapter
        awake_brain.target_x = (
            awake_brain.current_x + 50
        )  # keep pet moving; no new inference
        awake_brain.update(env_info)
        awake_brain.llm.apply_lora_from_path.assert_called_once_with(adapter)

    def test_update_no_adapter_pending(self, awake_brain: Brain, env_info):
        awake_brain.trainer.get_completed_adapter.return_value = None
        awake_brain.target_x = awake_brain.current_x + 50
        awake_brain.update(env_info)
        awake_brain.llm.apply_lora_from_path.assert_not_called()

    def test_update_swap_failure_logged_not_raised(
        self, awake_brain: Brain, env_info, tmp_path
    ):
        adapter = tmp_path / "bad.gguf"
        adapter.touch()
        awake_brain.trainer.get_completed_adapter.return_value = adapter
        awake_brain.llm.apply_lora_from_path.side_effect = RuntimeError("boom")
        awake_brain.target_x = awake_brain.current_x + 50
        awake_brain.update(env_info)  # must not propagate
