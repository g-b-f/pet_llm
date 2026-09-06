"""End-to-end integration tests for the Tank + Brain + Driver pipeline.

These run the real simulation loop (real Brain, real Tank, a headless
DummyDriver) and only mock the expensive LLM call, so the threading,
queue, memory, and report machinery all run for real.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_cpp.llama_types import CreateChatCompletionResponse

from lib.brain import Brain
from lib.drivers import DummyDriver
from lib.tank import Tank
from lib.types.config import SimulationConfig
from lib.types.other import Direction, PetAction
from lib.types.report import OutputReport

RUNTIME_SECONDS = 2



def _make_llm_response(content: str) -> CreateChatCompletionResponse:
    return {
        "id": "chatcmpl-e2e",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    file = tmp_path / "model.gguf"
    file.touch()
    return file


@pytest.fixture
def config() -> SimulationConfig:
    cfg = SimulationConfig.model_construct()
    cfg.tank.runtime = RUNTIME_SECONDS
    return cfg


@pytest.fixture
def mock_llm():
    """Mock LLM returning a valid, in-bounds decision for every call."""
    action = PetAction(thought="swimming along", direction=Direction.southeast, distance=100)
    llm = MagicMock()
    llm.create_chat_completion.return_value = _make_llm_response(
        action.model_dump_json()
    )
    return llm


def _chatty_llm() -> MagicMock:
    """A mock LLM that cycles through distinct thoughts and all 8 directions.

    Cycling the directions makes the pet trace a closed path (net displacement
    over a full cycle is zero), so it never reaches the tank wall and the
    out-of-bounds handler never clears memory. Distinct thoughts keep the
    thought-loop / similar-message supervision from firing. This makes the
    memory-accumulation assertion deterministic regardless of wall-clock timing.
    """
    directions = list(Direction)
    thoughts = [
        "chasing a bubble",
        "drifting by the glass",
        "darting to the light",
        "circling the tank",
        "napping on the gravel",
        "exploring the corner",
        "resting in the current",
        "watching the owner",
    ]
    state = {"n": 0}

    def _respond(*args, **kwargs):
        i = state["n"]
        state["n"] += 1
        action = PetAction(
            thought=thoughts[i % len(thoughts)],
            direction=directions[i % len(directions)],
            distance=20,
        )
        return _make_llm_response(action.model_dump_json())

    llm = MagicMock()
    llm.create_chat_completion.side_effect = _respond
    return llm


class TestEndToEnd:
    def test_simulation_wakes_brain_and_produces_output_report(
        self, config: SimulationConfig, mock_llm: MagicMock, model_path: Path
    ):
        with patch("lib.brain.Llama", return_value=mock_llm):
            brain = Brain(model_path, config.brain)
            driver = DummyDriver(config.tank.runtime)
            tank = Tank(brain, config.tank, driver)
            report = tank.run()

        assert brain.awake
        assert isinstance(report, OutputReport)
        assert report.report.iterations > 0
        assert report.report.actual_runtime is not None

    def test_simulation_honors_requested_runtime(
        self, config: SimulationConfig, mock_llm: MagicMock, model_path: Path
    ):
        with patch("lib.brain.Llama", return_value=mock_llm):
            brain = Brain(model_path, config.brain)
            driver = DummyDriver(config.tank.runtime)
            tank = Tank(brain, config.tank, driver)

            start = time.time()
            tank.run()
            elapsed = time.time() - start

        assert elapsed >= RUNTIME_SECONDS - 0.5

    def test_simulation_accumulates_llm_messages_in_memory(
        self, config: SimulationConfig, model_path: Path
    ):
        with patch("lib.brain.Llama", return_value=_chatty_llm()):
            brain = Brain(model_path, config.brain)
            driver = DummyDriver(config.tank.runtime)
            tank = Tank(brain, config.tank, driver)
            tank.run()

        assert brain.memory.length > 1

    def test_pet_moves_and_stays_in_bounds(
        self, config: SimulationConfig, mock_llm: MagicMock, model_path: Path
    ):
        with patch("lib.brain.Llama", return_value=mock_llm):
            brain = Brain(model_path, config.brain)
            driver = DummyDriver(config.tank.runtime)
            tank = Tank(brain, config.tank, driver)
            tank.run()

        expected_w = config.tank.screen_width - 2 * Tank.TANK_PADDING_X
        expected_h = config.tank.screen_height - Tank.TEXT_BOX_HEIGHT
        assert 0 <= brain.current_x <= expected_w
        assert 0 <= brain.current_y <= expected_h

    def test_malformed_llm_output_uses_fallback(
        self, config: SimulationConfig, mock_llm: MagicMock, model_path: Path
    ):
        mock_llm.create_chat_completion.return_value = _make_llm_response(
            "not valid json {"
        )

        with patch("lib.brain.Llama", return_value=mock_llm):
            brain = Brain(model_path, config.brain)
            driver = DummyDriver(config.tank.runtime)
            tank = Tank(brain, config.tank, driver)
            report = tank.run()

        assert report.report.malformed_json > 0
