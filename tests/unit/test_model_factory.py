"""Unit tests for ModelFactory."""

from unittest.mock import MagicMock
from unittest.mock import patch

from llm_assignment.models.factory import ModelFactory


@patch("llm_assignment.models.factory.load_base_model")
@patch("llm_assignment.models.factory.apply_lora")
def test_create_original_model(mock_apply_lora, mock_load_base):
    """Test creating original model."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Configure mock config
    mock_model.config.num_hidden_layers = 12
    mock_model.config.max_window_layers = 12

    mock_load_base.return_value = (mock_model, mock_tokenizer)
    mock_apply_lora.return_value = mock_model

    model, tokenizer = ModelFactory.create_model(model_type="original", model_name="test-model", use_lora=True)

    mock_load_base.assert_called_with("test-model", 4096)
    mock_apply_lora.assert_called_once()
    assert model == mock_model
    assert tokenizer == mock_tokenizer


@patch("llm_assignment.models.factory.load_base_model")
def test_create_dropped_model(mock_load_base):
    """Test creating dropped model (layer removal)."""
    # Mock model structure
    mock_model = MagicMock()
    # Simulate layers list
    mock_layers = [MagicMock() for _ in range(32)]
    mock_model.model.layers = mock_layers
    mock_model.config.num_hidden_layers = 32
    # Important: Set max_window_layers to integer to avoid comparison error
    mock_model.config.max_window_layers = 32
    mock_model.config.layer_types = None

    mock_load_base.return_value = (mock_model, MagicMock())

    # Create dropped model (drop layer 16)
    ModelFactory.create_model(model_type="dropped", layer_to_drop=16, use_lora=False)

    # Verify layer 16 was removed (length should be 31)
    assert len(mock_model.model.layers) == 31
    assert mock_model.config.num_hidden_layers == 31
    # Verify max_window_layers updated (32 > 31)
    assert mock_model.config.max_window_layers == 31


@patch("llm_assignment.models.factory.load_base_model")
def test_create_pruned_model(mock_load_base):
    """Test creating pruned model (truncation)."""
    # Mock model structure
    mock_model = MagicMock()
    mock_layers = [MagicMock() for _ in range(32)]
    mock_model.model.layers = mock_layers
    mock_model.config.num_hidden_layers = 32
    # Important: Set max_window_layers to integer
    mock_model.config.max_window_layers = 32
    mock_model.config.layer_types = None

    mock_load_base.return_value = (mock_model, MagicMock())

    # Create pruned model (keep 24)
    ModelFactory.create_model(model_type="pruned", keep_layers=24, use_lora=False)

    # Verify truncated
    assert len(mock_model.model.layers) == 24
    assert mock_model.config.num_hidden_layers == 24
    assert mock_model.config.max_window_layers == 24
