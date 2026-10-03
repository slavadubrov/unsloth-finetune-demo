"""CPU-only checks for the data preparation. The render test downloads the model's tokenizer.

Run: uv run --no-project --with pytest --with datasets --with transformers --with jinja2 pytest
"""

import json
import re

import pytest

from unsloth_demo.config import CHAT_TEMPLATE_PATH, MODEL_NAME
from unsloth_demo.data import to_messages, to_text

# Two rows copied from glaiveai/glaive-function-calling-v2 (train split, trimmed).
TOOL_ROW = {
    "system": (
        "SYSTEM: You are a helpful assistant with access to the following functions. "
        'Use them if required -\n{\n    "name": "get_news_headlines",\n'
        '    "description": "Get the latest news headlines",\n    "parameters": {\n'
        '        "type": "object",\n        "properties": {\n            "country": {\n'
        '                "type": "string",\n'
        '                "description": "The country for which to fetch news"\n'
        '            }\n        },\n        "required": [\n            "country"\n'
        "        ]\n    }\n}\n\n"
        '{\n    "name": "get_random_quote",\n    "description": "Get a random quote",\n'
        '    "parameters": {}\n}\n'
    ),
    "chat": (
        "USER: Can you tell me the latest news headlines for the United States?\n\n\n"
        'ASSISTANT: <functioncall> {"name": "get_news_headlines", '
        '"arguments": \'{"country": "United States"}\'} <|endoftext|>\n\n\n'
        'FUNCTION RESPONSE: {"headlines": ["Apple unveils new iPhone"]}\n\n\n'
        "ASSISTANT: Here are the latest headlines: Apple unveils new iPhone. <|endoftext|>\n\n\n"
        "USER: And a quote, please.\n\n\n"
        'ASSISTANT: <functioncall> {"name": "get_random_quote", "arguments": {}}'
        " <|endoftext|>\n\n\n"
        'FUNCTION RESPONSE: {"quote": "Stay hungry."}\n\n\n'
        'ASSISTANT: "Stay hungry." <|endoftext|>\n\n\n'
    ),
}
NO_TOOL_ROW = {
    "system": "SYSTEM: You are a helpful assistant, with no access to external functions.\n\n",
    "chat": (
        "USER: Hi!\n\n\nASSISTANT: Hello! How can I help? <|endoftext|>\n\n\nUSER: Thanks.\n\n\n"
    ),
}


def test_tools_come_from_the_system_column():
    _, tools = to_messages(TOOL_ROW)
    assert [t["function"]["name"] for t in tools] == ["get_news_headlines", "get_random_quote"]
    assert tools[0]["function"]["parameters"]["required"] == ["country"]
    assert to_messages(NO_TOOL_ROW)[1] == []


def test_calls_become_structured_tool_calls():
    messages, _ = to_messages(TOOL_ROW)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"] * 2
    first, second = messages[1]["tool_calls"][0], messages[5]["tool_calls"][0]
    assert first["function"] == {
        "name": "get_news_headlines",
        "arguments": {"country": "United States"},
    }
    assert second["function"] == {"name": "get_random_quote", "arguments": {}}
    contents = " ".join(m["content"] for m in messages)
    assert "<functioncall>" not in contents and "<|endoftext|>" not in contents


def test_trailing_user_turn_is_dropped():
    messages, _ = to_messages(NO_TOOL_ROW)
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_render_matches_the_serving_format():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.chat_template = CHAT_TEMPLATE_PATH.read_text()

    text = to_text(TOOL_ROW, tokenizer)["text"]

    assert not text.startswith(tokenizer.bos_token)  # the trainer adds BOS itself
    assert "<|im_start|>" not in text
    assert '<AVAILABLE_TOOLS>[{"name": "get_news_headlines"' in text
    # The model card's vLLM parser (llama_nemotron_json) runs json.loads on this span.
    calls = [json.loads(c) for c in re.findall(r"<TOOLCALL>(.*?)</TOOLCALL>", text, re.DOTALL)]
    assert calls == [
        [{"name": "get_news_headlines", "arguments": {"country": "United States"}}],
        [{"name": "get_random_quote", "arguments": {}}],
    ]
