from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.download import Model, get_model, mapping


class TestModelEnum:
    def test_all_models_have_mapping(self):
        for model in Model:
            assert model.value in mapping, f"{model.name} missing from mapping"

class TestGetModel:
    def test_returns_existing_file(self, tmp_path: Path):
        fake_model = tmp_path / "smollm2-1.7b-q8_0.gguf"
        fake_model.write_text("fake model data")
        with patch("models.download.Path") as mock_path_cls:
            mock_path_instance = MagicMock()
            mock_path_instance.parent = tmp_path
            mock_path_cls.return_value = mock_path_instance
            mock_path_cls.side_effect = lambda *a: Path(*a)
            with patch("models.download.huggingface_hub"):
                result = get_model(Model.smollm)
        assert result.name == "smollm2-1.7b-q8_0.gguf"

    def test_raises_when_error_downloading(self, tmp_path: Path):
        with patch("models.download.huggingface_hub") as mock_hf:
            mock_hf.hf_hub_download = MagicMock()
            with patch.object(Path, "exists", return_value=False):
                with pytest.raises(RuntimeError, match="error downloading"):
                    get_model(Model.smollm)

    def test_hf_download_called_with_correct_args(self, tmp_path: Path):
        with patch("models.download.huggingface_hub") as mock_hf:
            mock_hf.hf_hub_download = MagicMock()
            with patch.object(Path, "exists", return_value=False):
                with pytest.raises(RuntimeError):
                    get_model(Model.smollm)
            call_kwargs = mock_hf.hf_hub_download.call_args
            assert call_kwargs.kwargs["repo_id"] == mapping["smollm2-1.7b-q8_0"]
            assert call_kwargs.kwargs["filename"] == "smollm2-1.7b-q8_0.gguf"
