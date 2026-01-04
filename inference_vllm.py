"""
vLLM Inference Demo

Query the vLLM server using OpenAI-compatible API.

Prerequisites:
    1. Train with merged output: uv run python finetune.py --merge
    2. Start vLLM server:
       uv venv .venv-vllm --python 3.12
       source .venv-vllm/bin/activate
       uv pip install vllm openai
       vllm serve ./outputs/unsloth-nemotron-function-calling-merged --port 8000

Usage:
    python inference_vllm.py
    python inference_vllm.py --prompt "Book a flight to Tokyo"
    python inference_vllm.py --base-url http://localhost:8000/v1
"""

import argparse

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Query vLLM server with OpenAI API")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./outputs/unsloth-nemotron-function-calling-merged",
        help="Model name (must match the path used when starting vLLM server)",
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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
