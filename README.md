# Unsloth Fine-Tuning Demo

This repository fine-tunes NVIDIA's [Llama-3.1-Nemotron-Nano-4B-v1.1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1) for function calling with [Unsloth](https://docs.unsloth.ai/), LoRA on a 4-bit base, and TRL's `SFTTrainer`. The training data is [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2). It accompanies the article [LLM Fine-Tuning Guide](https://slavadubrov.com/blog/2026/01/04/llm-fine-tuning-guide/).

## Requirements

- Python 3.10 to 3.12 and [uv](https://docs.astral.sh/uv/)
- Linux (or WSL2) with an NVIDIA GPU. Unsloth trains on CUDA only.

## Quick start

```bash
git clone https://github.com/slavadubrov/unsloth-finetune-demo.git
cd unsloth-finetune-demo
uv sync

# Train on 1,000 random rows, then also save a merged model and a GGUF file
uv run finetune --max-samples 1000 --merge --gguf q4_k_m

# First check: ask the adapter to call the demo tool
uv run infer --prompt "Book a flight to Tokyo"
```

Every `finetune` run trains. `--merge` and `--gguf` add exports after training; there is no export-only command.

## What the training run does

1. Loads the base model in 4-bit and adds LoRA adapters (`r=16`, `alpha=32`, all attention and MLP projections).
2. Sets the tokenizer's chat template to [`chat_template.jinja`](src/unsloth_demo/chat_template.jinja), the model card's tool-calling template with one fix (below).
3. Converts each glaive row in [`data.py`](src/unsloth_demo/data.py). The function definitions in the `system` column become a `tools` list. The `chat` column becomes `messages`, and each `<functioncall>` becomes a structured `tool_calls` entry. Rows with malformed JSON are dropped (149 of 112,960).
4. Renders each row with `tokenizer.apply_chat_template(messages, tools=tools)`. The tools appear in `<AVAILABLE_TOOLS>` and each call in `<TOOLCALL>`, the same format vLLM's Nemotron parser reads at serving time.
5. Holds out 5% of the rows as an evaluation split (seed 42) and prints the first training row before training starts.
6. Trains with `SFTConfig`: `max_length=4096`, no packing, full-sequence loss, held-out loss once per epoch. Held-out loss is a diagnostic. It does not show whether the tool calls are correct.

The template fix: the model card's template renders a past tool call as `{"name": ..., "arguments": {...}` without the closing brace. The model would learn to emit that text, and the vLLM parser could not read it as JSON. The fixed template adds the brace. `tests/test_data.py` checks that every rendered `<TOOLCALL>` parses.

## Outputs

| Run flag        | Directory                                           | Load with                                                   |
| --------------- | --------------------------------------------------- | ----------------------------------------------------------- |
| (always)        | `outputs/unsloth-nemotron-function-calling/`        | `uv run infer`, vLLM `--enable-lora` (needs the base model) |
| `--merge`       | `outputs/unsloth-nemotron-function-calling-merged/` | `uv run infer --model ...`, vLLM, `lm_eval`, Transformers   |
| `--gguf q4_k_m` | `outputs/unsloth-nemotron-function-calling-gguf/`   | llama.cpp, Ollama                                           |

Other GGUF options: `q5_k_m`, `q8_0`, `f16`.

## Check the adapter

```bash
uv run infer --prompt "Book a flight to Tokyo"
```

`infer` sends the prompt with one tool, `book_flight(destination)`, using the training template. A trained adapter should answer with `<TOOLCALL>[{"name": "book_flight", "arguments": {"destination": "Tokyo"}}]</TOOLCALL>`. Use `--model outputs/unsloth-nemotron-function-calling-merged` to check the merged model.

## Serve with vLLM

The model card serves tool calls with the `llama_nemotron_json` parser from its `llama_nemotron_nano_toolcall_parser.py` plugin. Serve the adapter with that parser and this repository's template, so serving uses the format the adapter was trained on.

```bash
# vLLM has its own dependencies; use a separate environment
uv venv .venv-vllm --python 3.12
source .venv-vllm/bin/activate
uv pip install vllm huggingface_hub
hf download nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 \
    llama_nemotron_nano_toolcall_parser.py --local-dir serving

vllm serve nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 \
    --trust-remote-code \
    --enable-lora \
    --lora-modules function-calling=./outputs/unsloth-nemotron-function-calling \
    --enable-auto-tool-choice \
    --tool-parser-plugin ./serving/llama_nemotron_nano_toolcall_parser.py \
    --tool-call-parser llama_nemotron_json \
    --chat-template ./src/unsloth_demo/chat_template.jinja \
    --max-model-len 4096
```

In another terminal, from the project environment:

```bash
uv run --with openai infer-vllm --prompt "Book a flight to Tokyo"
```

`infer-vllm` sends the same `book_flight` tool and prints `message.tool_calls`. It prints `none` when the server parsed no call. To serve the merged model instead, replace the model path with `./outputs/unsloth-nemotron-function-calling-merged` and drop `--enable-lora` and `--lora-modules`.

## Run the GGUF file

```bash
ls outputs/unsloth-nemotron-function-calling-gguf/   # the file name depends on the Unsloth version
llama-cli -m outputs/unsloth-nemotron-function-calling-gguf/MODEL_FILE.gguf \
    -p "What's the weather in Tokyo?" --ctx-size 4096
```

This checks a plain-text answer only. It does not test a tool call.

## Tests

The tests run on CPU, without the GPU dependencies. The render test downloads the model's tokenizer.

```bash
uv run --no-project --with pytest --with datasets --with transformers --with jinja2 pytest
```

## Configuration

All settings are in [`src/unsloth_demo/config.py`](src/unsloth_demo/config.py): model and dataset names, LoRA rank and alpha, sequence length, batch size, learning rate, evaluation share, and the demo tool. If training runs out of memory, lower `BATCH_SIZE` (and raise `GRADIENT_ACCUMULATION_STEPS` to keep the effective batch) or lower `MAX_SEQ_LENGTH`. With this tokenizer and template, the longest of the 112,811 rendered rows is 4,076 tokens (median 370), so 4096 truncates none of them.

## License

MIT
