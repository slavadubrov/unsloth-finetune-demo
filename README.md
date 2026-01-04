# Unsloth Fine-Tuning Demo

A streamlined demo for fine-tuning NVIDIA's Nemotron-Nano-4B model using [Unsloth](https://github.com/unslothai/unsloth) - featuring 2x faster training and 60% less memory usage.

## Features

- 🚀 **2x faster training** through optimized CUDA kernels
- 💾 **60% less VRAM** usage with efficient LoRA implementation
- 📦 **Multiple export formats**: LoRA adapter, merged model, or GGUF
- 🎯 **Function calling dataset**: Pre-configured for glaive-function-calling-v2

## Quick Start

```bash
# Clone repository
git clone https://github.com/slavadubrov/unsloth-finetune-demo.git
cd fine-tuning-repo

# Sync dependencies
uv sync

# Run fine-tuning (test with 1000 samples)
uv run python finetune.py --max-samples 1000

# Full training with merged output
uv run python finetune.py --merge

# Export to GGUF for llama.cpp
uv run python finetune.py --gguf q4_k_m
```

## Requirements

- **Python**: 3.10-3.12
- **GPU**: NVIDIA with 12GB+ VRAM
- **CUDA**: 11.8+ or 12.1+
- **OS**: Linux (or WSL2 on Windows)

## Installation

### Option 1: Using UV (Recommended)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate  # On Windows WSL: source .venv/bin/activate

# Install dependencies
uv sync
```

### Option 2: Manual Installation

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install Unsloth (CUDA 12.1+)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or for CUDA 11.8
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"

# Install other dependencies
pip install peft transformers datasets trl accelerate
```

## Model & Dataset

- **Model**: [`nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1`](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1)
    - 4B parameters optimized for RTX GPUs
    - 128K context length
    - Native tool calling support

- **Dataset**: [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
    - 113K function calling examples
    - Diverse tool definitions
    - Multi-turn conversations

## Configuration

Key hyperparameters in `finetune.py`:

```python
MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
MAX_SEQ_LENGTH = 4096
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training args
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
num_train_epochs = 3
learning_rate = 2e-4
```

## Memory Requirements

| GPU         | VRAM | Batch Size | Status        |
| ----------- | ---- | ---------- | ------------- |
| RTX 3060    | 12GB | 1          | ✓ Works       |
| RTX 4070 Ti | 12GB | 1-2        | ✓ Works       |
| RTX 4080    | 16GB | 2          | ✓ Recommended |
| RTX 4090    | 24GB | 4          | ✓ Fast        |

> **Tip**: If you get OOM errors, reduce `per_device_train_batch_size` in `finetune.py`

## Usage Examples

### Basic Training

```bash
# Train with full dataset
uv run python finetune.py

# Output: ./outputs/unsloth-nemotron-function-calling/
```

### Test Run (Fast)

```bash
# Train with only 1000 samples for testing
uv run python finetune.py --max-samples 1000
```

### Export Merged Model

```bash
# Merge LoRA weights into base model
uv run python finetune.py --merge

# Output: ./outputs/unsloth-nemotron-function-calling-merged/
```

### Export to GGUF

```bash
# Export for llama.cpp with Q4_K_M quantization
uv run python finetune.py --gguf q4_k_m

# Other options: q5_k_m, q8_0, f16
```

### Custom Output Directory

```bash
uv run python finetune.py \
    --output-dir ./my-custom-output \
    --merge \
    --gguf q4_k_m
```

## Inference

### With Python (Unsloth)

```python
from unsloth import FastLanguageModel

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./outputs/unsloth-nemotron-function-calling",
    max_seq_length=4096,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Generate response
messages = [{"role": "user", "content": "What's the weather in Paris?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With llama.cpp (GGUF)

```bash
# Download llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Run inference
./main -m ../outputs/unsloth-nemotron-function-calling-gguf/model-q4_k_m.gguf \
    -p "What's the weather in Tokyo?" \
    --ctx-size 4096
```

## Troubleshooting

### "Cannot find Unsloth kernels"

Check your CUDA version and reinstall:

```bash
# Check CUDA version
nvcc --version

# For CUDA 12.1+
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# For CUDA 11.8
uv pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
```

### Out of Memory (OOM)

Reduce batch size or sequence length in `finetune.py`:

```python
MAX_SEQ_LENGTH = 2048  # Reduce from 4096
per_device_train_batch_size = 1  # Reduce from 2
```

### Slow Training

Enable gradient checkpointing and packing (already enabled by default):

```python
use_gradient_checkpointing="unsloth"  # Already set
packing=True  # Already set
```

## Project Structure

```
fine-tuning-repo/
├── finetune.py          # Main training script
├── pyproject.toml       # Dependencies
├── README.md            # This file
├── .python-version      # Python version spec
└── outputs/             # Training outputs (created automatically)
    ├── unsloth-nemotron-function-calling/         # LoRA adapter
    ├── unsloth-nemotron-function-calling-merged/  # Merged model
    └── unsloth-nemotron-function-calling-gguf/    # GGUF export
```

## Advanced: WandB Logging

To enable Weights & Biases logging:

1. Install wandb: `uv pip install wandb`
2. Login: `wandb login`
3. Edit `finetune.py`:
    ```python
    report_to="wandb"  # Change from "none"
    ```

## References

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Nemotron Model Card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1)
- [Glaive Function Calling Dataset](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

## License

MIT
