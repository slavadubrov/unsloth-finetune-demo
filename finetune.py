"""
Fine-tune Nemotron-Nano-4B for function calling using Unsloth.

Unsloth provides 2x faster training with 60% less memory usage through
optimized kernels and efficient LoRA implementation.

Usage:
    uv run python finetune.py

Requirements:
    See pyproject.toml for dependencies
"""

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Configuration
MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
DATASET_NAME = "glaiveai/glaive-function-calling-v2"
# Training parameters
MAX_SEQ_LENGTH = 4096
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
OUTPUT_DIR = "./outputs/unsloth-nemotron-function-calling"


def load_model_and_tokenizer():
    """Load base model with Unsloth optimizations."""
    print(f"Loading {MODEL_NAME}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
        trust_remote_code=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model, tokenizer


def prepare_dataset(tokenizer, max_samples: int | None = None):
    """Load and prepare glaive-function-calling-v2 dataset."""
    print("Loading glaive-function-calling-v2 dataset...")

    dataset = load_dataset(DATASET_NAME, split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def format_conversation(example):
        """Convert glaive format to chat messages."""
        chat = example.get("chat", "")

        # Simple conversion for demonstration
        formatted = chat.replace("SYSTEM:", "<|im_start|>system\n")
        formatted = formatted.replace("USER:", "<|im_end|>\n<|im_start|>user\n")
        formatted = formatted.replace("ASSISTANT:", "<|im_end|>\n<|im_start|>assistant\n")
        formatted = formatted.replace("FUNCTION RESPONSE:", "<|im_end|>\n<|im_start|>tool\n")

        if not formatted.endswith("<|im_end|>"):
            formatted += "<|im_end|>"

        return {"text": formatted}

    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)

    print(f"Prepared {len(dataset)} examples")
    return dataset


def train(model, tokenizer, dataset):
    """Run SFT training with Unsloth optimizations."""
    print("Starting training...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=42,
        report_to="none",  # Set to "wandb" for W&B logging
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,  # Efficient packing of short sequences
    )

    # Train
    trainer.train()

    return trainer


def save_model(model, tokenizer, output_dir: str):
    """Save the fine-tuned model."""
    print(f"Saving model to {output_dir}...")

    # Save LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✓ Saved LoRA adapter")


def save_merged_model(model, tokenizer, output_dir: str):
    """Merge LoRA and save full model."""
    print(f"Merging and saving to {output_dir}-merged...")

    merged_output = f"{output_dir}-merged"
    model.save_pretrained_merged(merged_output, tokenizer, save_method="merged_16bit")

    print("✓ Saved merged model")


def save_gguf(model, tokenizer, output_dir: str, quant: str = "q4_k_m"):
    """Export to GGUF format for llama.cpp."""
    print(f"Exporting to GGUF ({quant})...")

    gguf_output = f"{output_dir}-gguf"
    model.save_pretrained_gguf(gguf_output, tokenizer, quantization_method=quant)

    print(f"✓ Saved GGUF to {gguf_output}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Nemotron-Nano-4B with Unsloth")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max training samples (for testing)"
    )
    parser.add_argument(
        "--merge", action="store_true", help="Merge LoRA weights and save full model"
    )
    parser.add_argument(
        "--gguf",
        type=str,
        default=None,
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="Export to GGUF format with specified quantization",
    )
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Prepare dataset
    dataset = prepare_dataset(tokenizer, max_samples=args.max_samples)

    # Train
    train(model, tokenizer, dataset)

    # Save
    save_model(model, tokenizer, args.output_dir)

    if args.merge:
        save_merged_model(model, tokenizer, args.output_dir)

    if args.gguf:
        save_gguf(model, tokenizer, args.output_dir, quant=args.gguf)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
