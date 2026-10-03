"""Inference CLI for fine-tuned models.

Provides two inference modes:
- Unsloth inference: Direct model inference using Unsloth
- vLLM inference: Query a running vLLM server via OpenAI API

Usage:
    uv run infer                    # Unsloth inference
    uv run infer-vllm               # vLLM inference
"""

import argparse

from unsloth_demo.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT,
    DEMO_TOOL,
    MAX_SEQ_LENGTH,
    SERVED_MODEL_NAME,
)


def run_unsloth_inference(args: argparse.Namespace):
    """Run inference using Unsloth."""
    from unsloth_demo.model import load_model_for_inference

    model, tokenizer = load_model_for_inference(
        model_path=args.model,
        max_seq_length=args.max_seq_length,
    )

    print("Model loaded. Generating response...\n")

    # Same template and tool format as the training rows
    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tools=[DEMO_TOOL], add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    # Generate and decode only the new tokens
    outputs = model.generate(input_ids=inputs, max_new_tokens=args.max_tokens)
    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)

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
        tools=[DEMO_TOOL],
        tool_choice="auto",
        max_tokens=args.max_tokens,
    )
    message = response.choices[0].message

    print("=" * 50)
    print("PROMPT:")
    print(args.prompt)
    print("=" * 50)
    print("CONTENT:")
    print(message.content)
    print("TOOL CALLS:")
    for call in message.tool_calls or []:
        print(f"{call.function.name}({call.function.arguments})")
    if not message.tool_calls:
        print("none: the server parsed no tool call from this response")
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
        default=DEFAULT_PROMPT,
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
Prerequisite: a running vLLM server that serves the adapter as "function-calling"
with the model card's tool parser. See "Serve with vLLM" in README.md.

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
        default=SERVED_MODEL_NAME,
        help="Served model name (the name given in --lora-modules)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
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
