"""SFT training with TRL's SFTTrainer (patched by Unsloth on import)."""

import torch
from trl import SFTConfig, SFTTrainer

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


def train(
    model, tokenizer, train_dataset, eval_dataset, output_dir: str = DEFAULT_OUTPUT_DIR
) -> SFTTrainer:
    """Run SFT on the rendered `text` rows and report held-out loss once per epoch.

    Held-out token loss is a diagnostic. It does not measure whether tool calls are correct.
    """
    print("Starting training...")
    bf16 = torch.cuda.is_bf16_supported()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            packing=False,  # One row per sequence, so row boundaries are easy to inspect
            assistant_only_loss=False,  # The Nemotron template has no generation spans
            eval_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=WARMUP_RATIO,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            bf16=bf16,
            fp16=not bf16,
            optim="adamw_8bit",
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            seed=RANDOM_SEED,
            report_to="none",  # Set to "wandb" for W&B logging
        ),
    )

    trainer.train()
    return trainer
