"""Configuration constants for fine-tuning.

This module centralizes all configuration values used across the project.
Modify these values to customize your fine-tuning run.
"""

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
"""Base model to fine-tune. Nemotron-Nano-4B is a 4B parameter model with 128K context."""

DATASET_NAME = "glaiveai/glaive-function-calling-v2"
"""Training dataset. Contains 113K function calling examples."""

# =============================================================================
# LoRA Configuration
# =============================================================================

MAX_SEQ_LENGTH = 4096
"""Maximum sequence length for training. Reduce if running out of memory."""

LORA_R = 16
"""LoRA rank. Higher values = more parameters = better quality but slower."""

LORA_ALPHA = 32
"""LoRA alpha scaling factor. Usually set to 2x LORA_R."""

LORA_DROPOUT = 0.05
"""Dropout rate for LoRA layers. Helps prevent overfitting."""

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
"""Transformer modules to apply LoRA to."""

# =============================================================================
# Training Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = "./outputs/unsloth-nemotron-function-calling"
"""Default directory for saving trained models."""

BATCH_SIZE = 2
"""Per-device training batch size. Reduce if running out of memory."""

GRADIENT_ACCUMULATION_STEPS = 4
"""Number of gradient accumulation steps. Effective batch = BATCH_SIZE * this."""

WARMUP_RATIO = 0.03
"""Portion of training for learning rate warmup."""

NUM_EPOCHS = 3
"""Number of training epochs."""

LEARNING_RATE = 2e-4
"""Learning rate for AdamW optimizer."""

LOGGING_STEPS = 10
"""Log training metrics every N steps."""

SAVE_STEPS = 500
"""Save checkpoint every N steps."""

SAVE_TOTAL_LIMIT = 3
"""Maximum number of checkpoints to keep."""

RANDOM_SEED = 42
"""Random seed for reproducibility."""

# =============================================================================
# Chat Template Tokens
# =============================================================================

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
"""ChatML-style tokens for formatting conversations."""
