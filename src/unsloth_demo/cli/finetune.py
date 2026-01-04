"""Fine-tune Nemotron-Nano-4B for function calling using Unsloth.

Unsloth provides 2x faster training with 60% less memory usage through
optimized kernels and efficient LoRA implementation.

Usage:
    uv run finetune
    uv run finetune --max-samples 1000
    uv run finetune --merge --gguf q4_k_m
"""

import argparse

from unsloth_demo.config import DEFAULT_OUTPUT_DIR
from unsloth_demo.data import load_glaive_dataset
from unsloth_demo.model import (
    load_model_and_tokenizer,
    save_gguf,
    save_lora_adapter,
    save_merged_model,
)
from unsloth_demo.training import train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Nemotron-Nano-4B with Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  finetune                        # Full training, save LoRA adapter
  finetune --max-samples 1000     # Quick test with 1000 samples
  finetune --merge                # Also save merged model
  finetune --gguf q4_k_m          # Also export to GGUF format
        """,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max training samples (for quick testing)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA weights and save full model",
    )
    parser.add_argument(
        "--gguf",
        type=str,
        default=None,
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="Export to GGUF format with specified quantization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main():
    """Main entry point for fine-tuning."""
    args = parse_args()

    # Load model with LoRA adapters
    model, tokenizer = load_model_and_tokenizer()

    # Prepare dataset
    dataset = load_glaive_dataset(max_samples=args.max_samples)

    # Train
    train(model, tokenizer, dataset, output_dir=args.output_dir)

    # Save LoRA adapter (always)
    save_lora_adapter(model, tokenizer, args.output_dir)

    # Optional: save merged model
    if args.merge:
        save_merged_model(model, tokenizer, args.output_dir)

    # Optional: export to GGUF
    if args.gguf:
        save_gguf(model, tokenizer, args.output_dir, quantization=args.gguf)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
