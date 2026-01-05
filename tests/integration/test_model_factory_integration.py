"""Integration tests for ModelFactory."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from llm_assignment.models.factory import ModelFactory


@pytest.mark.integration
@patch("llm_assignment.models.factory.load_base_model")
@patch("llm_assignment.models.factory.apply_lora")
def test_model_factory_create_and_load_flow(mock_apply_lora, mock_load_base):
    """Test the end-to-end flow of creating and loading a model via factory."""
    # 1. Setup mocks
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model.config.num_hidden_layers = 32
    mock_model.config.max_window_layers = 32
    mock_model.config.layer_types = None
    mock_model.model.layers = [MagicMock() for _ in range(32)]

    mock_load_base.return_value = (mock_model, mock_tokenizer)
    mock_apply_lora.return_value = mock_model

    # 2. Test creation of pruned model
    model, _tokenizer = ModelFactory.create_model(model_type="pruned", keep_layers=16, use_lora=True)

    assert model == mock_model
    assert len(mock_model.model.layers) == 16
    mock_apply_lora.assert_called_once()

    # 3. Test loading for inference (adapter case)
    with (
        patch("pathlib.Path.exists", return_value=True),  # For adapter_config.json
        patch("llm_assignment.models.factory.load_base_model") as mock_load_inf,
        patch("unsloth.FastLanguageModel.for_inference"),
    ):
        mock_load_inf.return_value = (mock_model, mock_tokenizer)
        # Reset mock_model layers for fresh test
        mock_model.model.layers = [MagicMock() for _ in range(32)]

        ModelFactory.load_for_inference(model_path="some/path", model_type="dropped", layer_to_drop=10)

        assert len(mock_model.model.layers) == 31
        mock_model.load_adapter.assert_called_once()
