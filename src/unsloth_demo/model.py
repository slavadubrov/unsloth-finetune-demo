"""Model loading and saving utilities.

This module handles loading the base model with Unsloth optimizations,
applying LoRA adapters, and saving in various formats.
"""

import glob
import shutil
from pathlib import Path

from unsloth import FastLanguageModel

from .config import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
)


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
):
    """Load base model with Unsloth optimizations and LoRA adapters.

    Args:
        model_name: HuggingFace model identifier or local path.
        max_seq_length: Maximum sequence length.
        load_in_4bit: Enable 4-bit quantization for memory efficiency.

    Returns:
        Tuple of (model, tokenizer) with LoRA adapters applied.
    """
    print(f"Loading {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model, tokenizer


def load_model_for_inference(
    model_path: str,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
):
    """Load a fine-tuned model for inference.

    Args:
        model_path: Path to fine-tuned model (LoRA adapter or merged).
        max_seq_length: Maximum sequence length.
        load_in_4bit: Enable 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    print(f"Loading model from: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def save_lora_adapter(model, tokenizer, output_dir: str):
    """Save LoRA adapter weights only.

    This is the most storage-efficient option, saving only the trained
    adapter weights (~100-500MB) instead of the full model.

    Args:
        model: Fine-tuned model with LoRA adapters.
        tokenizer: Associated tokenizer.
        output_dir: Directory to save adapter weights.
    """
    print(f"Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✓ Saved LoRA adapter")


def save_merged_model(model, tokenizer, output_dir: str):
    """Merge LoRA weights into base model and save.

    Creates a standalone model that doesn't require the base model
    for inference. Larger file size (~8-16GB) but simpler deployment.

    Args:
        model: Fine-tuned model with LoRA adapters.
        tokenizer: Associated tokenizer.
        output_dir: Base directory (will append '-merged').
    """
    merged_output = f"{output_dir}-merged"
    print(f"Merging and saving to {merged_output}...")
    model.save_pretrained_merged(merged_output, tokenizer, save_method="merged_16bit")
    print("✓ Saved merged model")


def save_gguf(model, tokenizer, output_dir: str, quantization: str = "q4_k_m"):
    """Export model to GGUF format for llama.cpp.

    GGUF models can run on CPU with llama.cpp, making them ideal
    for edge deployment or systems without GPUs.

    Args:
        model: Fine-tuned model with LoRA adapters.
        tokenizer: Associated tokenizer.
        output_dir: Base directory (will append '-gguf').
        quantization: Quantization method (q4_k_m, q5_k_m, q8_0, f16).
    """

    gguf_output = f"{output_dir}-gguf"
    print(f"Exporting to GGUF ({quantization}) at {gguf_output}...")
    model.save_pretrained_gguf(gguf_output, tokenizer, quantization_method=quantization)

    # Unsloth saves GGUF files to the current working directory
    # Move them to the specified output directory
    gguf_output_path = Path(gguf_output)
    gguf_output_path.mkdir(parents=True, exist_ok=True)

    # Find and move all GGUF files from CWD to output directory
    cwd_gguf_files = glob.glob("*.gguf")
    for gguf_file in cwd_gguf_files:
        src = Path(gguf_file)
        dst = gguf_output_path / gguf_file
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"  Moved {gguf_file} → {dst}")

    print(f"✓ Saved GGUF to {gguf_output}")
