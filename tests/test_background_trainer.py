import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from lib.background_trainer import BackgroundTrainer
from lib.experience_buffer import ExperienceBuffer
from lib.extra_types import MemoryConfig, RoleContent
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
        from lib.background_trainer import DEFAULT_TRAINER_COMMAND

        assert trainer.trainer_command == DEFAULT_TRAINER_COMMAND


class TestLaunchTraining:
    def test_launches_subprocess(self, trainer: BackgroundTrainer, tmp_path: Path):
        with patch("lib.background_trainer.subprocess.Popen") as mock_popen:
            trainer.launch_training()
            assert mock_popen.called
            command = mock_popen.call_args[0][0]
            assert command[0] == sys.executable
            assert command[1].endswith("train_lora.py")
            assert "--model" in command
            assert "--dataset" in command
            assert "--output" in command
            assert str(tmp_path / "adapter_1.gguf") in command

    def test_writes_dataset(self, trainer: BackgroundTrainer):
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
                side_effect=FileNotFoundError("train_lora.py"),
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
        trainer._wait_for_completion(None, adapter_path)
        assert trainer.get_completed_adapter() == adapter_path
        assert not trainer.is_training

    def test_missing_adapter_not_queued(
        self, trainer: BackgroundTrainer, tmp_path: Path
    ):
        trainer._is_training = True
        trainer._wait_for_completion(None, tmp_path / "never_created.gguf")
        assert trainer.get_completed_adapter() is None
        assert not trainer.is_training

    def test_waits_for_process_exit(self, trainer: BackgroundTrainer, tmp_path: Path):
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        adapter_path = tmp_path / "out.gguf"
        adapter_path.touch()
        trainer._is_training = True
        trainer._wait_for_completion(process, adapter_path)
        assert process.poll() is not None  # process reaped
        assert trainer.get_completed_adapter() == adapter_path

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


PROJECT_ROOT = Path(__file__).parent.parent
TRAINER_SCRIPT = PROJECT_ROOT / "models" / "train_lora.py"
# A tiny HF model keeps the end-to-end training test fast on CPU. The production
# code passes the GGUF base model path; peft needs a HF-format model id/path.
TINY_HF_MODEL = "hf-internal-testing/tiny-random-SmolLM2"

# Command prefix that invokes the trainer as an isolated Python process, matching
# BackgroundTrainer's default. `_build_command` appends --model/--dataset/--output.
TRAINER_CMD = [sys.executable, str(TRAINER_SCRIPT)]


def _chat_buffer() -> ExperienceBuffer:
    mem = Memory(MemoryConfig(max_length=5))
    mem += RoleContent.user("Start exploring!")
    mem += RoleContent.assistant(
        content='{"thought": "exploring", "action": "move_to", "target_x": 5, "target_y": 5}'
    )
    buffer = ExperienceBuffer(mem)
    buffer.record(1.0)
    return buffer


@pytest.mark.slow
class TestRealTrainer:
    """Run the real, unmocked Python LoRA trainer (``models/train_lora.py``).

    These exercise the full subprocess pipeline: JSONL export, peft training,
    and GGUF conversion. They need network access on first run to download the
    HF model and llama.cpp's convert_lora_to_gguf.py.
    """

    def test_trainer_help_invocable(self):
        result = subprocess.run(
            [*TRAINER_CMD, "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0

    def test_trainer_rejects_missing_dataset(self, tmp_path: Path):
        result = subprocess.run(
            [
                *TRAINER_CMD,
                "--model",
                TINY_HF_MODEL,
                "--dataset",
                str(tmp_path / "nope.jsonl"),
                "--output",
                str(tmp_path / "out.gguf"),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode != 0

    def test_launch_training_invokes_trainer(self, tmp_path: Path):
        """Real launch against a bad model: process runs, fails, no adapter."""
        trainer = BackgroundTrainer(
            _chat_buffer(),
            model_path="nonexistent-model",
            output_dir=tmp_path,
            trainer_command=TRAINER_CMD,
        )
        trainer.launch_training()
        deadline = threading.Event()
        while trainer.is_training and not deadline.wait(0.5):
            pass
        assert not trainer.is_training
        assert trainer.get_completed_adapter() is None

    def test_full_training_produces_gguf_adapter(self, tmp_path: Path):
        """End-to-end: real training produces a .gguf LoRA adapter."""
        buffer = _chat_buffer()
        for _ in range(2):
            buffer.record(1.0)
        trainer = BackgroundTrainer(
            buffer,
            model_path=TINY_HF_MODEL,
            output_dir=tmp_path,
            trainer_command=TRAINER_CMD,
        )
        trainer.launch_training()
        adapter = None
        deadline = threading.Event()
        for _ in range(1200):  # up to ~10 minutes on CPU
            adapter = trainer.get_completed_adapter()
            if adapter is not None or not trainer.is_training:
                break
            deadline.wait(0.5)
        assert adapter is not None, "training did not produce an adapter"
        assert adapter.exists()
        assert adapter.suffix == ".gguf"
