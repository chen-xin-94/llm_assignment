"""Trainer classes and functions for BMW LLM fine-tuning with Unsloth.

This module contains the core training logic, separated from configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import os
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from dotenv import load_dotenv
from trl import SFTConfig
from trl import SFTTrainer

# Load environment variables from .env file in project root
_project_root = Path(__file__).resolve().parents[3]
load_dotenv(_project_root / ".env")


from dataclasses import field


@dataclass
class LoraConfig:
    """LoRA configuration."""

    r: int = 16
    alpha: int = 16


DEFAULT_REPORT_TO = ["none"]


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    model_type: str = "original"  # "original", "dropped", or "pruned"
    max_seq_length: int = 4096
    layer_to_drop: int = 16  # Only used for "dropped" models
    keep_layers: int = 24  # Only used for "pruned" models

    # LoRA
    use_lora: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Training
    output_dir: str = "checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    packing: bool = False

    # Logging
    logging_steps: int = 10
    logging_dir: str = "logs"
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 10

    # W&B
    wandb_project: str = "bmw-llm-finetuning"
    wandb_run_name: str | None = None
    report_to: list[str] = field(default_factory=lambda: DEFAULT_REPORT_TO)

    # Data
    dataset_path: str = "data/processed"

    # Other
    seed: int = 42
    fp16: bool = False
    bf16: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> TrainingConfig:
        """Load configuration from a YAML file.

        Supports inheritance via `_extends` key (relative path to base config).

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            TrainingConfig instance
        """
        import yaml

        path = Path(yaml_path)
        with path.open() as f:
            config_dict = yaml.safe_load(f)

        # Handle inheritance
        if "_extends" in config_dict:
            base_filename = config_dict.pop("_extends")
            base_path = path.parent / base_filename

            if not base_path.exists():
                raise FileNotFoundError(f"Base config {base_path} not found")

            with base_path.open() as f:
                base_dict = yaml.safe_load(f)

            # Recursive merge base_dict into config_dict (overriding base)
            # Simple 1-level merge for now, but handle nested 'lora' dict specifically
            merged_dict = base_dict.copy()

            # Deep merge logic could be added here if needed, but current usage is flat + lora
            for key, value in config_dict.items():
                if (
                    key == "lora"
                    and "lora" in merged_dict
                    and isinstance(value, dict)
                    and isinstance(merged_dict["lora"], dict)
                ):
                    merged_dict["lora"] = merged_dict["lora"].copy()
                    merged_dict["lora"].update(value)
                else:
                    merged_dict[key] = value

            config_dict = merged_dict

        # Parse lora as nested config for dot access
        lora_dict = config_dict.get("lora", {})
        if not isinstance(lora_dict, dict):
            # Handle case where lora might be just a boolean or something weird (though unlikely with type hints)
            lora_dict = {}

        lora = LoraConfig(
            r=lora_dict.get("r", 16),
            alpha=lora_dict.get("alpha", 16),
        )

        return cls(
            # Model settings
            model_name=config_dict.get("model_name", cls.model_name),
            model_type=config_dict.get("model_type", cls.model_type),
            max_seq_length=config_dict.get("max_seq_length", cls.max_seq_length),
            layer_to_drop=config_dict.get("layer_to_drop", cls.layer_to_drop),
            keep_layers=config_dict.get("keep_layers", cls.keep_layers),
            use_lora=config_dict.get("use_lora", cls.use_lora),
            # LoRA (nested for dot access)
            lora=lora,
            # Training settings
            output_dir=config_dict.get("output_dir", cls.output_dir),
            num_train_epochs=config_dict.get("num_train_epochs", cls.num_train_epochs),
            per_device_train_batch_size=config_dict.get("per_device_train_batch_size", cls.per_device_train_batch_size),
            gradient_accumulation_steps=config_dict.get("gradient_accumulation_steps", cls.gradient_accumulation_steps),
            learning_rate=config_dict.get("learning_rate", cls.learning_rate),
            warmup_ratio=config_dict.get("warmup_ratio", cls.warmup_ratio),
            weight_decay=config_dict.get("weight_decay", cls.weight_decay),
            gradient_checkpointing=config_dict.get("gradient_checkpointing", cls.gradient_checkpointing),
            packing=config_dict.get("packing", cls.packing),
            # Logging settings
            logging_steps=config_dict.get("logging_steps", cls.logging_steps),
            logging_dir=config_dict.get("logging_dir", cls.logging_dir),
            eval_strategy=config_dict.get("eval_strategy", cls.eval_strategy),
            eval_steps=config_dict.get("eval_steps", cls.eval_steps),
            save_steps=config_dict.get("save_steps", cls.save_steps),
            save_total_limit=config_dict.get("save_total_limit", cls.save_total_limit),
            # W&B settings
            wandb_project=config_dict.get("wandb_project", cls.wandb_project),
            wandb_run_name=config_dict.get("wandb_run_name"),
            report_to=config_dict.get("report_to", DEFAULT_REPORT_TO),
            # Data settings
            dataset_path=config_dict.get("dataset_path", cls.dataset_path),
            # Other settings
            seed=config_dict.get("seed", cls.seed),
            fp16=config_dict.get("fp16", cls.fp16),
            bf16=config_dict.get("bf16", cls.bf16),
        )


def get_training_args(config: TrainingConfig) -> SFTConfig:
    """Create SFTConfig from config."""

    return SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        bf16=config.bf16,
        seed=config.seed,
        report_to=config.report_to,
        run_name=config.wandb_run_name,
        logging_dir=config.logging_dir,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_text_field="text",
        max_length=config.max_seq_length,
        packing=config.packing,
    )


def load_model(config: TrainingConfig) -> tuple[Any, Any]:
    """Load model and tokenizer based on config.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    if config.model_type == "original":
        from llm_assignment.models.model import load_model_for_training

        model, tokenizer = load_model_for_training(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            lora_r=config.lora.r,
            lora_alpha=config.lora.alpha,
            use_lora=config.use_lora,
        )
    elif config.model_type == "dropped":
        from llm_assignment.models.dropped_model import create_dropped_model

        model, tokenizer = create_dropped_model(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            lora_r=config.lora.r,
            lora_alpha=config.lora.alpha,
            layer_to_drop=config.layer_to_drop,
            use_lora=config.use_lora,
        )
    elif config.model_type == "pruned":
        from llm_assignment.models.pruned_model import create_pruned_model

        model, tokenizer = create_pruned_model(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            lora_r=config.lora.r,
            lora_alpha=config.lora.alpha,
            keep_layers=config.keep_layers,
            use_lora=config.use_lora,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}. Must be 'original', 'dropped', or 'pruned'.")

    return model, tokenizer


def train_model(config: TrainingConfig) -> dict:  # pragma: no cover
    """Train the model with given configuration.

    Args:
        config: Training configuration

    Returns:
        Training metrics dictionary
    """
    # Determine run name (used by Trainer's wandb integration via SFTConfig.run_name)
    if config.wandb_run_name:
        wandb_run_name = config.wandb_run_name
    else:
        wandb_run_name = f"{config.model_name.split('/')[-1]}_{config.model_type}"
        wandb_run_name += "_lora" if config.use_lora else ""
        timestamp = datetime.now(tz=UTC).strftime("_%Y%m%d_%H%M%S")
        wandb_run_name += timestamp

        config.wandb_run_name = wandb_run_name

    # specify output_dir
    model_subdir = config.model_type
    if config.use_lora:
        model_subdir += "_lora"
    config.output_dir = f"{config.output_dir}/{model_subdir}"
    config.logging_dir = f"{config.logging_dir}/{model_subdir}"

    # Set wandb project via environment variable (Trainer reads this automatically)
    os.environ["WANDB_PROJECT"] = config.wandb_project

    # Load model
    model, tokenizer = load_model(config)

    # Load dataset
    dataset = load_from_disk(config.dataset_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Create trainer
    training_args = get_training_args(config)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Train
    print(f"\nStarting training for {config.model_type} model...")
    train_result = trainer.train()
    # Note: With load_best_model_at_end=True (set in SFTConfig), the Trainer
    # automatically reloads the best checkpoint after training completes.
    # The Trainer also handles wandb.finish() internally.

    # Save the best model (loaded automatically due to load_best_model_at_end=True)
    final_path = Path(config.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Best model saved to {final_path}")

    # Get metrics from training
    # Note: We don't call trainer.evaluate() again because:
    # 1. The best model's eval_loss was already logged during training
    # 2. The wandb run is already closed by the Trainer
    # 3. Re-evaluating would create a confusing drop in the eval/loss chart
    return {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        # best_metric contains the eval_loss of the best checkpoint (since metric_for_best_model="eval_loss")
        "eval_loss": trainer.state.best_metric,
    }
