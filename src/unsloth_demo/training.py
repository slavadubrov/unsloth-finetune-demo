"""Training logic for Unsloth fine-tuning.

This module provides the core training functionality using SFTTrainer
with Unsloth optimizations for faster, memory-efficient training.
"""

import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from .config import (
    BATCH_SIZE,
    DEFAULT_OUTPUT_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    RANDOM_SEED,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    WARMUP_RATIO,
)


def create_training_args(output_dir: str = DEFAULT_OUTPUT_DIR) -> TrainingArguments:
    """Create TrainingArguments with optimized defaults.

    Args:
        output_dir: Directory for checkpoints and final model.

    Returns:
        Configured TrainingArguments instance.
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        optim="adamw_8bit",
        seed=RANDOM_SEED,
        report_to="none",  # Set to "wandb" for W&B logging
    )


def train(model, tokenizer, dataset, output_dir: str = DEFAULT_OUTPUT_DIR) -> SFTTrainer:
    """Run SFT training with Unsloth optimizations.

    Uses sequence packing for efficient training of variable-length
    conversations, reducing padding waste.

    Args:
        model: Model with LoRA adapters applied.
        tokenizer: Associated tokenizer.
        dataset: Prepared dataset with 'text' field.
        output_dir: Directory for checkpoints.

    Returns:
        Trained SFTTrainer instance.
    """
    print("Starting training...")

    training_args = create_training_args(output_dir)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,  # Efficient packing of short sequences
    )

    trainer.train()

    return trainer
