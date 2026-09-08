from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_cpp.llama_types import CreateChatCompletionResponse

from tests.mocks import MockInference, MockValidInference, ScriptedInference
from lib.brain import Brain
from lib.types.config import BrainConfig
from lib.types.other import EnvironmentalInfo, PetAction, RoleContent



@pytest.fixture
def brain() -> Brain:
    config = BrainConfig.model_construct()
    _brain = Brain(Path("fake/model/path.gguf"), config, MockInference())
    _brain.wake_up((100, 100))
    return _brain


@pytest.fixture
def env_info() -> EnvironmentalInfo:
    return EnvironmentalInfo(mouse=(0, 0))


class TestWakeUp:
    def test_awake_after_wake_up(self, brain: Brain):
        assert brain.awake

    def test_initial_position_centered(self, brain: Brain):
        assert brain.current_x == 50.0
        assert brain.current_y == 50.0

    def test_initial_thought(self, brain: Brain):
        assert (
            brain.current_thought == brain.config.thoughts.initial_thought
        )

    def test_initial_memory_has_one_entry(self, brain: Brain):
        assert brain.memory.length == 1

    def test_not_thinking_initially(self, brain: Brain):
        assert not brain.is_thinking

    def test_iterations_zero(self, brain: Brain):
        assert brain.iterations == 0

    def test_oob_count_zero(self, brain: Brain):
        assert brain.current_oob_count == 0


class TestUpdate:
    def test_pet_moves_toward_target(
        self, brain: Brain, env_info: EnvironmentalInfo
    ):
        brain.target_x = 100.0
        brain.target_y = 50.0
        initial_x = brain.current_x
        brain.update(env_info)
        assert brain.current_x > initial_x

    def test_pet_speed(self, brain: Brain, env_info: EnvironmentalInfo):
        brain.target_x = 100.0
        brain.target_y = 50.0
        initial_x = brain.current_x
        brain.update(env_info)
        assert abs(brain.current_x - (initial_x + Brain.PET_SPEED)) < 0.01

    def test_queued_decision_applied(
        self, brain: Brain, env_info: EnvironmentalInfo
    ):
        decision = PetAction(thought="new thought", target_x=10, target_y=20)
        brain.result_queue.put(decision)
        brain.update(env_info)
        assert brain.current_thought == "new thought"
        assert brain.target_x == 10
        assert brain.target_y == 20

    def test_arrival_triggers_decision_request(
        self, brain: Brain, env_info: EnvironmentalInfo
    ):
        brain.target_x = brain.current_x
        brain.target_y = brain.current_y
        with patch.object(brain, "request_decision_async") as mock_req:
            brain.update(env_info)
            mock_req.assert_called_once()


class TestFallback:
    def test_fallback_queues_decision(self, brain: Brain):
        brain._fallback()
        assert not brain.result_queue.empty()
        decision = brain.result_queue.get()
        assert decision.thought == brain.config.thoughts.fallback_thought

    def test_fallback_within_bounds(self, brain: Brain):
        brain._fallback()
        decision = brain.result_queue.get()
        assert 0 <= decision.target_x <= brain.x_bounds
        assert 0 <= decision.target_y <= brain.y_bounds


class TestRequestDecisionAsync:
    def test_asserts_if_not_awake(self):
        config = BrainConfig.model_construct()
        asleep_brain = Brain(Path("fake/model/path.gguf"), config, MockInference())
        with pytest.raises(AssertionError, match="still asleep"):
            asleep_brain.request_decision_async(0, 0)

    def test_skips_if_already_thinking(self, brain: Brain):
        brain.is_thinking = True
        with patch("lib.brain.threading.Thread") as mock_thread:
            brain.request_decision_async(50, 50)
            mock_thread.assert_not_called()

    def test_starts_thread(self, brain: Brain):
        with patch("lib.brain.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            brain.request_decision_async(50, 50)
            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()

    def test_sets_thinking_flag(self, brain: Brain):
        with patch("lib.brain.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            brain.request_decision_async(50, 50)
            assert brain.is_thinking


class TestTargetOutOfBounds:
    @pytest.mark.parametrize(
        ("target_x", "target_y"), [(101, 50), (-1, 50), (50, 101), (50, -1)]
    )
    def test_target_out_of_bounds(self, brain: Brain, target_x: int, target_y: int):
        action = PetAction(thought="t", target_x=target_x, target_y=target_y)
        assert brain.target_out_of_bounds(action)

    @pytest.mark.parametrize(
        ("target_x", "target_y"), [(50, 50), (100, 100), (0, 0), (100, 0), (0, 100)]
    )
    def test_target_in_bounds(self, brain: Brain, target_x: int, target_y: int):
        action = PetAction(thought="t", target_x=target_x, target_y=target_y)
        assert not brain.target_out_of_bounds(action)


class TestGenerateDecision:
    def _setup_brain_for_generation(
            self,
            action: PetAction|None = None,
            config = BrainConfig.model_construct(),
            ):
        
        if action is None:
            action = PetAction(thought="hello", target_x=10, target_y=20)
        inference = ScriptedInference(RoleContent.assistant(action.model_dump_json()))
        brain = Brain(Path("fake/model/path.gguf"), config, inference)
        brain.wake_up((50,50))

        brain._generate_decision(50, 50)
        return brain

    def test_successful_decision_queued(self):
        brain = self._setup_brain_for_generation()
        assert not brain.result_queue.empty()
        decision = brain.result_queue.get()
        assert decision.thought == "hello"

    def test_increments_iterations(self):
        brain = self._setup_brain_for_generation()
        assert brain.iterations == 1

    def test_clears_thinking_flag(self, brain: Brain):
        brain.is_thinking = True
        brain._generate_decision(50, 50)
        assert not brain.is_thinking

    def test_oob_decision_not_queued(self):
        action = PetAction(thought="oob", target_x=999, target_y=999)
        brain = self._setup_brain_for_generation(action)
        assert brain.result_queue.empty()

    def test_oob_increments_oob_count(self):
        action = PetAction(thought="oob", target_x=999, target_y=999)
        brain = self._setup_brain_for_generation(action)
        assert brain.current_oob_count == 1

    def test_max_oob_triggers_fallback(self, brain: Brain):
        brain.current_oob_count = Brain.MAX_OOB_COUNT - 1
        action = PetAction(thought="oob", target_x=999, target_y=999)
        thought = RoleContent.assistant(action.model_dump_json())
        brain.inference = ScriptedInference(thought)
        brain._generate_decision(10, 10)

        assert brain.current_oob_count == 0
        assert not brain.result_queue.empty()
        decision = brain.result_queue.get()
        assert decision.thought == brain.config.thoughts.fallback_thought

    def test_memory_cleared_on_max_oob(self, brain: Brain):
        brain.current_oob_count = Brain.MAX_OOB_COUNT - 1
        action = PetAction(thought="oob", target_x=999, target_y=999)
        thought = RoleContent.assistant(action.model_dump_json())
        brain.inference = ScriptedInference(thought)
        brain._generate_decision(10, 10)

        assert brain.memory.length == 0

    def test_memory_updated_on_success(self, brain: Brain):
        brain.inference = MockValidInference()
        initial_len = brain.memory.length
        brain._generate_decision(10,10)
        assert brain.memory.length > initial_len

    def test_malformed_json_resets_thinking_and_counts(self, brain: Brain):
        thought = RoleContent.assistant("not valid json {{")
        brain.inference = ScriptedInference(thought)
        
        brain.is_thinking = True
        brain._generate_decision(50, 50)

        assert not brain.is_thinking
        assert brain.report.malformed_json == 1
        assert not brain.result_queue.empty()

        decision = brain.result_queue.get()
        assert decision.thought == brain.config.thoughts.fallback_thought
