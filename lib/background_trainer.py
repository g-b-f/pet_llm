import json
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from lib.experience_buffer import ExperienceBuffer
from lib.utils import get_logger

logger = get_logger(__name__)

ADAPTERS_DIR = Path(__file__).parent.parent / "adapters"

# Low-priority creation flags so training never starves the render thread.
LOW_PRIORITY_KWARGS: dict = (
    {"creationflags": subprocess.IDLE_PRIORITY_CLASS}
    if sys.platform == "win32"
    else {"nice": 19}
)


class BackgroundTrainer:
    """Manages background LoRA fine-tuning runs.

    Polls an ``ExperienceBuffer`` from a daemon monitor thread. Once the buffer
    reaches ``trigger_capacity`` entries, positive experiences are exported to
    a temporary JSONL dataset and an external trainer (``llama-finetune`` or
    any compatible command) is launched as a low-priority subprocess.

    When the subprocess finishes and produced a ``.gguf`` adapter, the
    adapter's path is put on ``completion_queue`` so the main thread can
    hot-swap it into a ``DynamicAdapterLLM`` (drain the queue with
    ``get_completed_adapter``).
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        model_path: str | Path,
        output_dir: str | Path = ADAPTERS_DIR,
        trigger_capacity: int = 32,
        trainer_command: list[str] | None = None,
    ) -> None:
        self.buffer = buffer
        self.model_path = str(model_path)
        self.output_dir = Path(output_dir)
        self.trigger_capacity = trigger_capacity
        self.trainer_command = trainer_command or ["llama-finetune"]

        self.completion_queue: queue.Queue[Path] = queue.Queue()
        self._is_training = False
        self._run_count = 0
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    @property
    def is_training(self) -> bool:
        return self._is_training

    def start(self) -> None:
        """Start the background monitor thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="background-trainer",
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Signal the monitor thread to stop."""
        self._stop_event.set()

    def get_completed_adapter(self) -> Path | None:
        """Non-blocking drain of the completion queue; call from the main thread."""
        try:
            return self.completion_queue.get_nowait()
        except queue.Empty:
            return None

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(1.0):
            if not self._is_training and len(self.buffer) >= self.trigger_capacity:
                self.launch_training()

    def launch_training(self) -> None:
        """Export positive experiences and launch the trainer subprocess.

        Only one training run can be in flight at a time; calls while a run is
        active are no-ops.
        """
        if self._is_training:
            logger.debug("training already in progress, skipping trigger")
            return

        dataset = self.buffer.export_positive()
        if not dataset:
            logger.info("no positive experiences to train on, skipping")
            return

        self._is_training = True
        self._run_count += 1
        logger.info(
            "launching training run %s on %s examples", self._run_count, len(dataset)
        )

        dataset_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w",
            suffix=".jsonl",
            prefix=f"pet_train_{self._run_count}_",
            delete=False,
            encoding="utf-8",
        )
        try:
            with dataset_file:
                for example in dataset:
                    dataset_file.write(json.dumps(example) + "\n")

            self.output_dir.mkdir(parents=True, exist_ok=True)
            adapter_path = self.output_dir / f"adapter_{self._run_count}.gguf"
            command = self._build_command(dataset_file.name, adapter_path)

            subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **LOW_PRIORITY_KWARGS,
            )
            threading.Thread(
                target=self._wait_for_completion,
                args=(adapter_path,),
                daemon=True,
                name=f"trainer-wait-{self._run_count}",
            ).start()
        except Exception:
            self._is_training = False
            raise

    def _build_command(self, dataset_path: str, adapter_path: Path) -> list[str]:
        return [
            *self.trainer_command,
            "--model",
            self.model_path,
            "--dataset",
            dataset_path,
            "--output",
            str(adapter_path),
        ]

    def _wait_for_completion(self, adapter_path: Path) -> None:
        try:
            if adapter_path.exists():
                logger.info("adapter ready: %s", adapter_path)
                self.completion_queue.put(adapter_path)
            else:
                logger.warning(
                    "training finished but produced no adapter at %s", adapter_path
                )
        finally:
            self._is_training = False
