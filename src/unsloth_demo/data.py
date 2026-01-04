"""Dataset loading and formatting utilities.

This module handles loading the glaive-function-calling-v2 dataset
and formatting it for training with ChatML-style templates.
"""

from datasets import load_dataset

from .config import DATASET_NAME, IM_END, IM_START


def format_conversation(example: dict) -> dict:
    """Convert glaive format to ChatML-style messages.

    The glaive dataset uses a simple text format:
        SYSTEM: ...
        USER: ...
        ASSISTANT: ...
        FUNCTION RESPONSE: ...

    This converts to ChatML format:
        <|im_start|>system
        ...<|im_end|>
        <|im_start|>user
        ...<|im_end|>
        etc.

    Args:
        example: Dataset example with 'chat' field.

    Returns:
        Dict with 'text' field containing formatted conversation.
    """
    chat = example.get("chat", "")

    formatted = chat.replace("SYSTEM:", f"{IM_START}system\n")
    formatted = formatted.replace("USER:", f"{IM_END}\n{IM_START}user\n")
    formatted = formatted.replace("ASSISTANT:", f"{IM_END}\n{IM_START}assistant\n")
    formatted = formatted.replace("FUNCTION RESPONSE:", f"{IM_END}\n{IM_START}tool\n")

    if not formatted.endswith(IM_END):
        formatted += IM_END

    return {"text": formatted}


def load_glaive_dataset(max_samples: int | None = None):
    """Load and prepare glaive-function-calling-v2 dataset.

    Args:
        max_samples: Limit number of training samples. None for full dataset.

    Returns:
        Formatted dataset ready for training.
    """
    print(f"Loading {DATASET_NAME} dataset...")

    dataset = load_dataset(DATASET_NAME, split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)

    print(f"Prepared {len(dataset)} examples")
    return dataset
