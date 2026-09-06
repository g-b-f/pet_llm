"""End-to-end integration tests for the Tank + Brain + Driver pipeline.

These run the real simulation loop (real Brain, real Tank, a headless
DummyDriver) and only mock the expensive LLM call, so the threading,
queue, memory, and report machinery all run for real.
"""

from pathlib import Path
import time
from unittest.mock import MagicMock, patch

import pytest

from lib.brain import Brain
from lib.drivers import DummyDriver
from lib.tank import Tank
from lib.types.config import SimulationConfig
from lib.types.other import PetAction
from lib.types.report import OutputReport
from models.download import Model, get_model

RUNTIME_SECONDS = 2

def _make_llm_response(content: str) -> dict:
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
    action = PetAction(thought="swimming along", target_x=100, target_y=100)
    llm = MagicMock()
    llm.create_chat_completion.return_value = _make_llm_response(
        action.model_dump_json()
    )
    return llm


class TestEndToEnd:
    def test_full_simulation_runs_and_reports(
        self, config: SimulationConfig,
        mock_llm: MagicMock,
        model_path: Path
    ):
        with patch("lib.brain.Llama", return_value=mock_llm):
            brain = Brain(model_path, config.brain)
            driver = DummyDriver(config.tank.runtime)
            tank = Tank(brain, config.tank, driver)

            start = time.time()
            report = tank.run()
            elapsed = time.time() - start

        # brain was woken and produced a real report
        assert brain.awake
        assert isinstance(report, OutputReport)
        assert report.report.iterations > 0
        assert report.report.actual_runtime is not None

        # roughly honored the requested runtime
        assert elapsed >= RUNTIME_SECONDS - 0.5

        # memory accumulated LLM messages
        assert brain.memory.length > 1

    def test_pet_moves_and_stays_in_bounds(
        self, config: SimulationConfig,
        mock_llm: MagicMock,
        model_path: Path
    ):
        with patch("lib.brain.Llama", return_value=mock_llm):
            brain = Brain(model_path, config.brain)
            driver = DummyDriver(config.tank.runtime)
            tank = Tank(brain, config.tank, driver)
            tank.run()

        # position stays within tank-local bounds
        expected_w = config.tank.screen_width - 2 * Tank.TANK_PADDING_X
        expected_h = config.tank.screen_height - Tank.TEXT_BOX_HEIGHT
        assert 0 <= brain.current_x <= expected_w
        assert 0 <= brain.current_y <= expected_h

    def test_malformed_llm_output_uses_fallback(
        self, config: SimulationConfig,
        mock_llm: MagicMock,
        model_path: Path
    ):
        mock_llm.create_chat_completion.return_value = _make_llm_response(
            "not valid json {"
        )

        with patch("lib.brain.Llama", return_value=mock_llm):
            brain = Brain(model_path, config.brain)
            driver = DummyDriver(config.tank.runtime)
            tank = Tank(brain, config.tank, driver)
            report = tank.run()

        # malformed responses were counted and fell back without crashing
        assert report.report.malformed_json > 0