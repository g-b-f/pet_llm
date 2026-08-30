import ctypes
import threading
from pathlib import Path

from llama_cpp import Llama, llama_cpp

from lib.utils import get_logger

logger = get_logger(__name__)


class DynamicAdapterError(Exception):
    """Raised when a LoRA adapter operation fails at the llama.cpp level."""


class DynamicAdapterLLM:
    """Thread-safe wrapper around ``llama_cpp.Llama`` with dynamic LoRA support.

    ``llama_cpp.Llama`` only applies a LoRA adapter at construction time. This
    wrapper adds runtime adapter management by calling the low-level
    ``llama_adapter_lora_*`` / ``llama_set_adapters_lora`` bindings directly,
    so a fine-tuned ``.gguf`` adapter can be hot-swapped into a live context.

    All public methods are guarded by a single reentrant lock: inference and
    adapter mutation are mutually exclusive, so it is safe to swap adapters
    while a background inference thread is running.
    """

    def __init__(self, model_path: str | Path, **llama_kwargs) -> None:
        self._lock = threading.RLock()
        self.llm = Llama(model_path=str(model_path), **llama_kwargs)
        self._adapter: llama_cpp.llama_adapter_lora_p | None = None
        self._adapter_path: str | None = None
        self._scale: float = 1.0

    def apply_lora_from_path(self, path: str | Path, scale: float = 1.0) -> None:
        """Load a GGUF LoRA adapter from ``path`` and apply it to the context.

        Applying a second adapter transparently swaps the first one out.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {path}")

        with self._lock:
            # Hold the old adapter until the new one is fully applied, so a
            # failure leaves the previous adapter untouched.
            old_adapter = self._adapter

            adapter = llama_cpp.llama_adapter_lora_init(
                self.llm._model.model,
                str(path).encode("utf-8"),
            )
            if adapter is None:
                raise DynamicAdapterError(f"Failed to initialize LoRA adapter: {path}")

            adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(adapter)  # type: ignore[misc]
            scales = (ctypes.c_float * 1)(scale)
            if llama_cpp.llama_set_adapters_lora(
                self.llm._ctx.ctx, adapters, 1, scales
            ):
                llama_cpp.llama_adapter_lora_free(adapter)
                raise DynamicAdapterError(f"Failed to apply LoRA adapter: {path}")

            if old_adapter is not None:
                llama_cpp.llama_adapter_lora_free(old_adapter)

            self._adapter = adapter
            self._adapter_path = str(path)
            self._scale = scale
            logger.info("applied LoRA adapter %s (scale=%s)", path.name, scale)

    def scale_lora(self, scale: float) -> None:
        """Rescale the currently applied adapter in-place."""
        with self._lock:
            if self._adapter is None:
                raise DynamicAdapterError("No LoRA adapter applied")

            adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(self._adapter)  # type: ignore[misc]
            scales = (ctypes.c_float * 1)(scale)
            if llama_cpp.llama_set_adapters_lora(
                self.llm._ctx.ctx, adapters, 1, scales
            ):
                raise DynamicAdapterError(f"Failed to scale LoRA adapter to {scale}")

            self._scale = scale
            logger.debug("rescaled LoRA adapter to %s", scale)

    def clear_lora(self) -> None:
        """Remove the currently applied adapter and free its memory."""
        with self._lock:
            if self._adapter is None:
                return

            # Passing NULL adapters/scales removes all adapters from the context.
            if llama_cpp.llama_set_adapters_lora(self.llm._ctx.ctx, None, 0, None):
                raise DynamicAdapterError("Failed to clear LoRA adapter")

            llama_cpp.llama_adapter_lora_free(self._adapter)
            logger.info("cleared LoRA adapter %s", Path(self._adapter_path or "?").name)
            self._adapter = None
            self._adapter_path = None
            self._scale = 1.0

    def create_chat_completion(self, messages, **kwargs):
        """Run inference with the adapter state frozen for the call."""
        with self._lock:
            return self.llm.create_chat_completion(messages, **kwargs)

    @property
    def has_adapter(self) -> bool:
        return self._adapter is not None

    @property
    def adapter_path(self) -> str | None:
        return self._adapter_path

    @property
    def adapter_scale(self) -> float:
        return self._scale

    def __getattr__(self, name: str):
        # Delegate anything not explicitly wrapped to the inner Llama.
        # Bypass __getattr__ for `llm` itself to avoid infinite recursion if
        # __init__ failed before assigning it.
        if name == "llm":
            raise AttributeError(name)
        return getattr(self.llm, name)

    def close(self) -> None:
        """Free the applied adapter (if any) and release the model."""
        adapter = self.__dict__.get("_adapter")
        if adapter is not None:
            llama_cpp.llama_adapter_lora_free(adapter)
            self._adapter = None
