"""Unit tests for TrainingConfig and trainer utilities."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import yaml

from llm_assignment.training.trainer import TrainingConfig


def test_config_defaults():
    """Test default configuration values."""
    config = TrainingConfig()
    assert config.model_name == "Qwen/Qwen3-8B"
    assert config.use_lora is True


def test_from_yaml_inheritance(tmp_path):
    """Test configuration inheritance via _extends."""
    # Create base config
    base_config = {"model_name": "base-model", "use_lora": True, "lora": {"r": 8, "alpha": 8}, "learning_rate": 1e-4}
    base_file = tmp_path / "base.yaml"
    with base_file.open("w") as f:
        yaml.dump(base_config, f)

    # Create child config
    child_config = {
        "_extends": "base.yaml",
        "model_name": "child-model",
        "lora": {"r": 16},  # Should partial override
    }
    child_file = tmp_path / "child.yaml"
    with child_file.open("w") as f:
        yaml.dump(child_config, f)

    # Load child config
    config = TrainingConfig.from_yaml(child_file)

    # Verify inheritance and overrides
    assert config.model_name == "child-model"  # Overridden
    assert config.learning_rate == 1e-4  # Inherited
    assert config.use_lora is True  # Inherited

    # Verify partial nested merge (custom logic in from_yaml)
    assert config.lora.r == 16  # Overridden
    assert config.lora.alpha == 8  # Inherited (if merge logic works)


def test_from_yaml_missing_base(tmp_path):
    """Test error when base config is missing."""
    child_config = {"_extends": "nonexistent.yaml"}
    child_file = tmp_path / "bad_child.yaml"
    with child_file.open("w") as f:
        yaml.dump(child_config, f)

    with pytest.raises(FileNotFoundError):
        TrainingConfig.from_yaml(child_file)


def test_get_training_args():
    """Test creation of SFTConfig from TrainingConfig."""
    from llm_assignment.training.trainer import get_training_args

    config = TrainingConfig(model_name="test-model", learning_rate=5e-5, max_seq_length=1024, packing=True)
    args = get_training_args(config)

    assert args.learning_rate == 5e-5
    assert args.max_length == 1024
    assert args.packing is True
    assert args.output_dir == "checkpoints"


@patch("llm_assignment.models.model.load_model_for_training")
def test_load_model_original(mock_load):
    """Test load_model for original type."""
    from llm_assignment.training.trainer import load_model

    mock_load.return_value = (MagicMock(), MagicMock())

    config = TrainingConfig(model_type="original")
    load_model(config)

    mock_load.assert_called_once()


def test_load_model_invalid():
    """Test load_model with invalid type raises ValueError."""
    from llm_assignment.training.trainer import load_model

    config = TrainingConfig(model_type="invalid")
    with pytest.raises(ValueError, match="Unknown model_type"):
        load_model(config)
