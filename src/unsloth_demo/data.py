"""Turn glaive-function-calling-v2 rows into training text in the model's own chat format.

Each glaive row has two string columns:

    system: "SYSTEM: You are a helpful assistant with access to the following
             functions. Use them if required -\\n{function JSON}\\n\\n{function JSON}"
    chat:   "USER: ...\\n\\n\\nASSISTANT: <functioncall> {...} <|endoftext|>\\n\\n\\n
             FUNCTION RESPONSE: {...}\\n\\n\\nASSISTANT: ... <|endoftext|>"

`to_messages` parses a row into OpenAI-style `messages` and `tools`. `to_text` renders
them with `tokenizer.apply_chat_template`, so a training row uses the same template
(and the same <AVAILABLE_TOOLS> and <TOOLCALL> markers) that vLLM uses at serving time.
"""

import json
import re

from datasets import load_dataset

from .config import DATASET_NAME, EVAL_FRACTION, RANDOM_SEED

TURN = re.compile(r"(?:^|\n\n)(USER|ASSISTANT|FUNCTION RESPONSE): ?")
# Most glaive calls quote the arguments object: "arguments": '{"city": "Paris"}'
QUOTED_ARGUMENTS = re.compile(r"'(\{.*\})'(\s*\}\s*)$", re.DOTALL)
ROLES = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}


def parse_tools(system: str) -> list[dict]:
    """Return every function JSON object in the glaive `system` string as a tool."""
    decoder = json.JSONDecoder()
    tools = []
    start = system.find("{")
    while start != -1:
        function, end = decoder.raw_decode(system, start)
        tools.append({"type": "function", "function": function})
        start = system.find("{", end)
    return tools


def parse_chat(chat: str) -> list[dict]:
    """Split the glaive `chat` string into messages; assistant calls become `tool_calls`."""
    parts = TURN.split(chat)
    messages = []
    for marker, text in zip(parts[1::2], parts[2::2]):
        text = text.replace("<|endoftext|>", "").strip()
        message = {"role": ROLES[marker], "content": text}
        if marker == "ASSISTANT" and "<functioncall>" in text:
            before, _, payload = text.partition("<functioncall>")
            call = json.loads(QUOTED_ARGUMENTS.sub(r"\1\2", payload.strip()))
            message["content"] = before.strip()
            message["tool_calls"] = [
                {
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
            ]
        messages.append(message)
    # A row that ends on a user turn or a tool result has no answer to learn.
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    return messages


def to_messages(example: dict) -> tuple[list[dict], list[dict]]:
    """Return (messages, tools) for one glaive row. Raises ValueError on a malformed row."""
    return parse_chat(example["chat"]), parse_tools(example["system"])


def to_text(example: dict, tokenizer) -> dict:
    """Render one glaive row with the model's chat template. Empty text marks a dropped row."""
    try:
        messages, tools = to_messages(example)
    except (ValueError, KeyError):  # bad JSON in a tool definition or call
        return {"text": ""}
    if not messages:
        return {"text": ""}
    text = tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False)
    # The trainer tokenizes this text with special tokens on, which adds BOS again.
    return {"text": text.removeprefix(tokenizer.bos_token or "")}


def load_glaive_dataset(tokenizer, max_samples: int | None = None):
    """Load glaive-function-calling-v2 and return (train, eval) datasets with a `text` column.

    Args:
        tokenizer: Tokenizer of the model being trained; its chat template formats each row.
        max_samples: Use a random subset of this many rows. None for the full dataset.
    """
    print(f"Loading {DATASET_NAME} dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")

    if max_samples:
        dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(min(max_samples, len(dataset))))

    rows = len(dataset)
    dataset = dataset.map(
        to_text, fn_kwargs={"tokenizer": tokenizer}, remove_columns=dataset.column_names
    )
    dataset = dataset.filter(lambda row: bool(row["text"]))
    print(f"Prepared {len(dataset)} of {rows} rows ({rows - len(dataset)} malformed rows dropped)")

    split = dataset.train_test_split(test_size=EVAL_FRACTION, seed=RANDOM_SEED)
    print(f"Train: {len(split['train'])} rows, eval: {len(split['test'])} rows")
    return split["train"], split["test"]
