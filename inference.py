"""
Unsloth Inference Demo

Run inference on the fine-tuned model using Unsloth.

Usage:
    uv run python inference.py
    uv run python inference.py --model ./outputs/unsloth-nemotron-function-calling
    uv run python inference.py --prompt "What's the weather in Tokyo?"
"""

import argparse

from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned model")
    parser.add_argument(
        "--model",
        type=str,
        default="./outputs/unsloth-nemotron-function-calling",
        help="Path to the fine-tuned model (LoRA adapter or merged)",
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
        default=4096,
        help="Maximum sequence length",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    
    # Load the fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print(f"Model loaded. Generating response...\n")

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


if __name__ == "__main__":
    main()
