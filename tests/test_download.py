from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.download import Model, get_model, mapping


class TestModelEnum:
    def test_all_models_have_mapping(self):
        for model in Model:
            assert model in mapping, f"{model.name} missing from mapping"

    def test_mapping_values_are_repo_ids(self):
        for repo_id in mapping.values():
            assert "/" in repo_id


class TestGetModel:
    def test_returns_existing_file(self):
        with (
            patch.object(Path, "exists", return_value=True),
            patch("models.download.huggingface_hub"),
        ):
            result = get_model(Model.smollm2)
        assert result.name == "smollm2-1.7b-q8_0.gguf"

    def test_raises_when_error_downloading(self):
        with (
            patch("models.download.huggingface_hub") as mock_hf,
            patch.object(Path, "exists", return_value=False),
            patch("models.download.token_path") as mock_token,
        ):
            mock_hf.hf_hub_download = MagicMock()
            mock_token.exists.return_value = False
            with pytest.raises(RuntimeError, match="error downloading"):
                get_model(Model.smollm2)

    def test_hf_download_called_with_correct_args(self):
        with (
            patch("models.download.huggingface_hub") as mock_hf,
            patch.object(Path, "exists", return_value=False),
            patch("models.download.token_path") as mock_token,
        ):
            mock_hf.hf_hub_download = MagicMock()
            mock_token.exists.return_value = False
            with pytest.raises(RuntimeError):
                get_model(Model.smollm2)
            call_kwargs = mock_hf.hf_hub_download.call_args
            assert call_kwargs.kwargs["repo_id"] == mapping[Model.smollm2]
            assert call_kwargs.kwargs["filename"] == "smollm2-1.7b-q8_0.gguf"
