import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from lib.background_trainer import BackgroundTrainer
from lib.experience_buffer import ExperienceBuffer
from lib.types.other import RoleContent
from lib.types.config import MemoryConfig
from lib.memory import Memory


@pytest.fixture
def buffer() -> ExperienceBuffer:
    mem = Memory(MemoryConfig(max_length=5))
    mem += RoleContent.user("Start exploring!")
    buf = ExperienceBuffer(mem)
    buf.record(1.0)
    return buf


@pytest.fixture
def trainer(buffer: ExperienceBuffer, tmp_path: Path) -> BackgroundTrainer:
    return BackgroundTrainer(
        buffer,
        model_path="fake/model.gguf",
        output_dir=tmp_path,
        trigger_capacity=2,
    )


class TestBackgroundTrainerInit:
    def test_defaults(self, trainer: BackgroundTrainer):
        assert not trainer.is_training
        assert trainer.trigger_capacity == 2
        assert trainer.completion_queue.empty()

    def test_default_trainer_command(self, trainer: BackgroundTrainer):
        assert trainer.trainer_command == ["llama-finetune"]


class TestLaunchTraining:
    def test_launches_subprocess(self, trainer: BackgroundTrainer, tmp_path: Path):
        with patch("lib.background_trainer.subprocess.Popen") as mock_popen:
            trainer.launch_training()
            assert mock_popen.called
            command = mock_popen.call_args[0][0]
            assert command[0] == "llama-finetune"
            assert "--model" in command
            assert "--dataset" in command
            assert "--output" in command
            assert str(tmp_path / "adapter_1.gguf") in command
        assert trainer.is_training  # stays true until the wait thread finishes

    def test_low_priority_flags(self, trainer: BackgroundTrainer):
        with patch("lib.background_trainer.subprocess.Popen") as mock_popen:
            trainer.launch_training()
            kwargs = mock_popen.call_args[1]
            if sys.platform == "win32":
                assert kwargs["creationflags"] == subprocess.IDLE_PRIORITY_CLASS
            else:
                assert kwargs["nice"] == 19

    def test_writes_jsonl_dataset(self, trainer: BackgroundTrainer):
        with patch("lib.background_trainer.subprocess.Popen") as mock_popen:
            trainer.launch_training()
            command = mock_popen.call_args[0][0]
            dataset_path = Path(command[command.index("--dataset") + 1])
            lines = dataset_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1
            assert '"messages"' in lines[0]

    def test_consumes_exported_experiences(
        self, trainer: BackgroundTrainer, buffer: ExperienceBuffer
    ):
        with patch("lib.background_trainer.subprocess.Popen"):
            trainer.launch_training()
        assert buffer.is_empty

    def test_skips_when_no_positive_experiences(
        self, trainer: BackgroundTrainer, buffer: ExperienceBuffer
    ):
        buffer.clear()
        buffer.record(-1.0)
        with patch("lib.background_trainer.subprocess.Popen") as mock_popen:
            trainer.launch_training()
            mock_popen.assert_not_called()
        assert not trainer.is_training

    def test_noop_while_training(
        self, trainer: BackgroundTrainer, buffer: ExperienceBuffer
    ):
        with patch("lib.background_trainer.subprocess.Popen") as mock_popen:
            trainer.launch_training()
            buffer.record(1.0)
            trainer.launch_training()  # second run blocked while first in flight
            assert mock_popen.call_count == 1

    def test_popen_failure_resets_flag(self, trainer: BackgroundTrainer):
        with (
            patch(
                "lib.background_trainer.subprocess.Popen",
                side_effect=FileNotFoundError("llama-finetune"),
            ),
            pytest.raises(FileNotFoundError),
        ):
            trainer.launch_training()
        assert not trainer.is_training


class TestCompletionNotification:
    def test_adapter_queued_on_completion(
        self, trainer: BackgroundTrainer, tmp_path: Path
    ):
        with patch("lib.background_trainer.subprocess.Popen"):
            trainer.launch_training()
        adapter_path = tmp_path / "adapter_1.gguf"
        adapter_path.touch()  # trainer subprocess would produce this
        trainer._wait_for_completion(adapter_path)
        assert trainer.get_completed_adapter() == adapter_path
        assert not trainer.is_training

    def test_missing_adapter_not_queued(
        self, trainer: BackgroundTrainer, tmp_path: Path
    ):
        trainer._is_training = True
        trainer._wait_for_completion(tmp_path / "never_created.gguf")
        assert trainer.get_completed_adapter() is None
        assert not trainer.is_training

    def test_get_completed_adapter_empty(self, trainer: BackgroundTrainer):
        assert trainer.get_completed_adapter() is None


class TestMonitorThread:
    def test_start_and_stop(self, trainer: BackgroundTrainer):
        trainer.start()
        assert trainer._monitor_thread is not None
        assert trainer._monitor_thread.is_alive()
        trainer.stop()
        trainer._monitor_thread.join(timeout=2.0)
        assert not trainer._monitor_thread.is_alive()

    def test_monitor_triggers_at_capacity(
        self, trainer: BackgroundTrainer, buffer: ExperienceBuffer
    ):
        fired = threading.Event()

        def fake_launch():
            fired.set()
            trainer._is_training = True  # prevent repeated triggering

        with patch.object(trainer, "launch_training", side_effect=fake_launch):
            trainer.start()
            buffer.record(1.0)  # buffer now at trigger_capacity=2
            assert fired.wait(timeout=3.0)
        trainer.stop()
