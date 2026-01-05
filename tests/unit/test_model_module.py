"""Unit tests for models/model.py wrapper functions."""

from unittest.mock import MagicMock
from unittest.mock import patch

from llm_assignment.models.model import load_model_for_inference_auto


@patch("llm_assignment.models.model.ModelFactory.load_for_inference")
def test_load_model_for_inference_auto_dict(mock_load):
    """Test auto loading with config dict."""
    config = {"model_type": "original", "max_seq_length": 512}
    load_model_for_inference_auto("path/to/model", config=config)

    mock_load.assert_called_with(model_path="path/to/model", model_type="original", max_seq_length=512)


@patch("llm_assignment.models.model.ModelFactory.load_for_inference")
@patch("llm_assignment.training.trainer.TrainingConfig.from_yaml")
def test_load_model_for_inference_auto_yaml(mock_from_yaml, mock_load):
    """Test auto loading with yaml path."""
    mock_config = MagicMock()
    mock_config.model_type = "original"
    mock_config.model_name = "Qwen/Qwen3-8B"
    mock_config.max_seq_length = 2048
    mock_from_yaml.return_value = mock_config

    load_model_for_inference_auto("path/to/model", config="config.yaml")

    mock_from_yaml.assert_called_with("config.yaml")
    mock_load.assert_called_with(model_path="path/to/model", model_type="original", max_seq_length=2048)
