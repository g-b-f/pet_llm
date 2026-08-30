from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib.dynamic_adapter_llm import DynamicAdapterError, DynamicAdapterLLM


@pytest.fixture
def mock_llama():
    with patch("lib.dynamic_adapter_llm.Llama") as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_llama_cpp():
    with patch("lib.dynamic_adapter_llm.llama_cpp") as mock:
        mock.llama_adapter_lora_init.return_value = MagicMock(name="adapter")
        mock.llama_set_adapters_lora.return_value = 0
        yield mock


@pytest.fixture
def adapter_path(tmp_path: Path) -> Path:
    path = tmp_path / "adapter.gguf"
    path.touch()
    return path


@pytest.fixture
def llm(mock_llama: MagicMock, mock_llama_cpp: MagicMock) -> DynamicAdapterLLM:
    return DynamicAdapterLLM("fake/model.gguf", n_ctx=512)


class TestDynamicAdapterLLMInit:
    def test_constructs_llama(self, mock_llama: MagicMock):
        with patch("lib.dynamic_adapter_llm.Llama") as mock_cls:
            DynamicAdapterLLM("fake/model.gguf", n_ctx=512)
            mock_cls.assert_called_once_with(model_path="fake/model.gguf", n_ctx=512)

    def test_no_adapter_initially(self, llm: DynamicAdapterLLM):
        assert not llm.has_adapter
        assert llm.adapter_path is None
        assert llm.adapter_scale == 1.0


class TestApplyLora:
    def test_apply_success(
        self, llm: DynamicAdapterLLM, adapter_path: Path, mock_llama_cpp: MagicMock
    ):
        llm.apply_lora_from_path(adapter_path, scale=0.7)
        assert llm.has_adapter
        assert llm.adapter_path == str(adapter_path)
        assert llm.adapter_scale == 0.7
        mock_llama_cpp.llama_adapter_lora_init.assert_called_once()
        mock_llama_cpp.llama_set_adapters_lora.assert_called_once()

    def test_apply_missing_file(self, llm: DynamicAdapterLLM):
        with pytest.raises(FileNotFoundError):
            llm.apply_lora_from_path("does/not/exist.gguf")

    def test_apply_init_failure(
        self, llm: DynamicAdapterLLM, adapter_path: Path, mock_llama_cpp: MagicMock
    ):
        mock_llama_cpp.llama_adapter_lora_init.return_value = None
        with pytest.raises(DynamicAdapterError):
            llm.apply_lora_from_path(adapter_path)
        assert not llm.has_adapter

    def test_apply_set_failure(
        self, llm: DynamicAdapterLLM, adapter_path: Path, mock_llama_cpp: MagicMock
    ):
        mock_llama_cpp.llama_set_adapters_lora.return_value = -1
        with pytest.raises(DynamicAdapterError):
            llm.apply_lora_from_path(adapter_path)
        assert not llm.has_adapter

    def test_swap_frees_old_adapter(
        self, llm: DynamicAdapterLLM, adapter_path: Path, mock_llama_cpp: MagicMock
    ):
        llm.apply_lora_from_path(adapter_path)
        old_adapter = mock_llama_cpp.llama_adapter_lora_init.return_value
        new_adapter = MagicMock(name="new_adapter")
        mock_llama_cpp.llama_adapter_lora_init.return_value = new_adapter

        llm.apply_lora_from_path(adapter_path, scale=0.5)
        mock_llama_cpp.llama_adapter_lora_free.assert_called_once_with(old_adapter)
        assert llm.adapter_scale == 0.5

    def test_failed_swap_keeps_old_adapter(
        self, llm: DynamicAdapterLLM, adapter_path: Path, mock_llama_cpp: MagicMock
    ):
        llm.apply_lora_from_path(adapter_path)
        mock_llama_cpp.llama_set_adapters_lora.return_value = -1
        with pytest.raises(DynamicAdapterError):
            llm.apply_lora_from_path(adapter_path)
        assert llm.has_adapter  # original adapter still applied


class TestScaleLora:
    def test_scale(self, llm: DynamicAdapterLLM, adapter_path: Path):
        llm.apply_lora_from_path(adapter_path)
        llm.scale_lora(0.3)
        assert llm.adapter_scale == 0.3

    def test_scale_without_adapter(self, llm: DynamicAdapterLLM):
        with pytest.raises(DynamicAdapterError):
            llm.scale_lora(0.5)

    def test_scale_failure(
        self, llm: DynamicAdapterLLM, adapter_path: Path, mock_llama_cpp: MagicMock
    ):
        llm.apply_lora_from_path(adapter_path)
        mock_llama_cpp.llama_set_adapters_lora.return_value = -1
        with pytest.raises(DynamicAdapterError):
            llm.scale_lora(0.5)
        assert llm.adapter_scale == 1.0


class TestClearLora:
    def test_clear(
        self, llm: DynamicAdapterLLM, adapter_path: Path, mock_llama_cpp: MagicMock
    ):
        llm.apply_lora_from_path(adapter_path)
        llm.clear_lora()
        assert not llm.has_adapter
        assert llm.adapter_path is None
        assert llm.adapter_scale == 1.0
        mock_llama_cpp.llama_adapter_lora_free.assert_called_once()

    def test_clear_without_adapter_is_noop(
        self, llm: DynamicAdapterLLM, mock_llama_cpp: MagicMock
    ):
        llm.clear_lora()
        mock_llama_cpp.llama_set_adapters_lora.assert_not_called()

    def test_clear_failure(
        self, llm: DynamicAdapterLLM, adapter_path: Path, mock_llama_cpp: MagicMock
    ):
        llm.apply_lora_from_path(adapter_path)
        mock_llama_cpp.llama_set_adapters_lora.return_value = -1
        with pytest.raises(DynamicAdapterError):
            llm.clear_lora()
        assert llm.has_adapter


class TestInference:
    def test_create_chat_completion_delegates(
        self, llm: DynamicAdapterLLM, mock_llama: MagicMock
    ):
        mock_llama.create_chat_completion.return_value = {"ok": True}
        result = llm.create_chat_completion(
            [{"role": "user", "content": "hi"}], temperature=0.5
        )
        assert result == {"ok": True}
        mock_llama.create_chat_completion.assert_called_once_with(
            [{"role": "user", "content": "hi"}], temperature=0.5
        )

    def test_getattr_delegates(self, llm: DynamicAdapterLLM, mock_llama: MagicMock):
        mock_llama.n_ctx = MagicMock(return_value=512)
        assert llm.n_ctx() == 512
