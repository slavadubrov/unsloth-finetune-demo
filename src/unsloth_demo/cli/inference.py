"""Inference CLI for fine-tuned models.

Provides two inference modes:
- Unsloth inference: Direct model inference using Unsloth
- vLLM inference: Query a running vLLM server via OpenAI API

Usage:
    uv run infer                    # Unsloth inference
    uv run infer-vllm               # vLLM inference
"""

import argparse

from unsloth_demo.config import DEFAULT_OUTPUT_DIR, MAX_SEQ_LENGTH


def run_unsloth_inference(args: argparse.Namespace):
    """Run inference using Unsloth."""
    from unsloth_demo.model import load_model_for_inference

    model, tokenizer = load_model_for_inference(
        model_path=args.model,
        max_seq_length=args.max_seq_length,
    )

    print("Model loaded. Generating response...\n")

    # Prepare the prompt
    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    # Generate response
    outputs = model.generate(inputs, max_new_tokens=args.max_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("=" * 50)
    print("PROMPT:")
    print(args.prompt)
    print("=" * 50)
    print("RESPONSE:")
    print(response)
    print("=" * 50)


def run_vllm_inference(args: argparse.Namespace):
    """Run inference via vLLM server."""
    from openai import OpenAI

    print(f"Connecting to vLLM server at: {args.base_url}")

    client = OpenAI(base_url=args.base_url, api_key="dummy")

    print(f"Sending prompt to model: {args.model}\n")

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
    )

    print("=" * 50)
    print("PROMPT:")
    print(args.prompt)
    print("=" * 50)
    print("RESPONSE:")
    print(response.choices[0].message.content)
    print("=" * 50)


def create_unsloth_parser() -> argparse.ArgumentParser:
    """Create parser for Unsloth inference."""
    parser = argparse.ArgumentParser(
        description="Run inference on fine-tuned model using Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  infer                                           # Default model and prompt
  infer --model ./outputs/my-model                # Custom model path
  infer --prompt "Book a flight to Tokyo"         # Custom prompt
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to fine-tuned model (LoRA adapter or merged)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What's the weather in Paris?",
        help="User prompt to send to the model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum sequence length",
    )
    return parser


def create_vllm_parser() -> argparse.ArgumentParser:
    """Create parser for vLLM inference."""
    parser = argparse.ArgumentParser(
        description="Query vLLM server with OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
  1. Train with merged output: finetune --merge
  2. Start vLLM server:
     uv venv .venv-vllm --python 3.12
     source .venv-vllm/bin/activate
     uv pip install vllm openai
     vllm serve ./outputs/unsloth-nemotron-function-calling-merged --port 8000

Examples:
  infer-vllm                                      # Default settings
  infer-vllm --prompt "Book a flight to Tokyo"    # Custom prompt
  infer-vllm --base-url http://server:8000/v1     # Remote server
        """,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=f"{DEFAULT_OUTPUT_DIR}-merged",
        help="Model name (must match path used when starting vLLM server)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What's the weather in Paris?",
        help="User prompt to send to the model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    return parser


def main_unsloth():
    """Entry point for Unsloth inference."""
    parser = create_unsloth_parser()
    args = parser.parse_args()
    run_unsloth_inference(args)


def main_vllm():
    """Entry point for vLLM inference."""
    parser = create_vllm_parser()
    args = parser.parse_args()
    run_vllm_inference(args)


if __name__ == "__main__":
    main_unsloth()
