# Unsloth Fine-Tuning Demo

A beginner-friendly demo for fine-tuning NVIDIA's Nemotron-Nano-4B model using [Unsloth](https://github.com/unslothai/unsloth).

## Features

> **Performance compared to vanilla Hugging Face Transformers + PEFT fine-tuning**

- 🚀 **2x faster training** through optimized CUDA kernels
- 💾 **60% less VRAM** usage with efficient LoRA implementation
- 📦 **Multiple export formats**: LoRA adapter, merged model, or GGUF
- 🎯 **Function calling dataset**: Pre-configured for glaive-function-calling-v2

## Requirements

- **Python**: 3.10-3.12
- **GPU**: NVIDIA with 12GB+ VRAM
- **CUDA**: 11.8+ or 12.1+
- **OS**: Linux (or WSL2 on Windows)

## Quick Start

```bash
# Clone repository
git clone https://github.com/slavadubrov/unsloth-finetune-demo.git
cd unsloth-finetune-demo

# Sync dependencies (installs uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh)
uv sync

# Run fine-tuning with 1000 samples (quick test)
uv run finetune --max-samples 1000
```

## Output Format Options

**For beginners**: Choose the format based on how you plan to use your fine-tuned model:

| Format           | Command                         | Size       | Best For                   |
| ---------------- | ------------------------------- | ---------- | -------------------------- |
| **LoRA Adapter** | `uv run finetune`               | ~100-500MB | Python/PyTorch inference   |
| **Merged Model** | `uv run finetune --merge`       | ~8-16GB    | Sharing, simple deployment |
| **GGUF**         | `uv run finetune --gguf q4_k_m` | ~2-4GB     | CPU inference, llama.cpp   |

### 1️⃣ LoRA Adapter (Default)

```bash
uv run finetune --max-samples 1000
```

- ✅ Smallest file size - only saves adapter weights
- ✅ Most flexible - can load with different base model versions
- 📁 Output: `./outputs/unsloth-nemotron-function-calling/`

### 2️⃣ Merged Model

```bash
uv run finetune --merge
```

- ✅ Easier to share - single model with all weights
- ✅ Simpler deployment - no separate base model needed
- 📁 Output: `./outputs/unsloth-nemotron-function-calling-merged/`

### 3️⃣ GGUF Format

```bash
uv run finetune --gguf q4_k_m
```

- ✅ Runs on CPU with llama.cpp
- ✅ Perfect for edge/local deployment
- 📁 Output: `./outputs/unsloth-nemotron-function-calling-gguf/`
- Other quantization options: `q5_k_m`, `q8_0`, `f16`

> **Tip**: Start with LoRA adapter for testing. Use `--merge` to share, or `--gguf` for CPU deployment.

## Configuration

Default settings in [`src/unsloth_demo/config.py`](src/unsloth_demo/config.py):

```python
MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"  # 4B params, 128K context
DATASET_NAME = "glaiveai/glaive-function-calling-v2"   # 113K examples
MAX_SEQ_LENGTH = 4096
LORA_R = 16
LORA_ALPHA = 32
DEFAULT_OUTPUT_DIR = "./outputs/unsloth-nemotron-function-calling"
```

### Memory Requirements

| GPU         | VRAM | Batch Size | Status        |
| ----------- | ---- | ---------- | ------------- |
| RTX 3060    | 12GB | 1          | ✓ Works       |
| RTX 4070 Ti | 12GB | 1-2        | ✓ Works       |
| RTX 4080    | 16GB | 2          | ✓ Recommended |
| RTX 4090    | 24GB | 4          | ✓ Fast        |

> **OOM?** Reduce `BATCH_SIZE` or `MAX_SEQ_LENGTH` in `src/unsloth_demo/config.py`

## Inference

### With Unsloth (LoRA/Merged)

```bash
# Default: uses ./outputs/unsloth-nemotron-function-calling
uv run infer

# Custom model path
uv run infer --model ./outputs/unsloth-nemotron-function-calling-merged

# Custom prompt
uv run infer --prompt "Book a flight to Tokyo for tomorrow"
```

### With llama.cpp (GGUF)

**Install llama.cpp:**

```bash
# macOS (Homebrew)
brew install llama.cpp

# Or build from source (Linux/Windows WSL)
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
```

**Run inference:**

```bash
# GGUF files are saved to: ./outputs/unsloth-nemotron-function-calling-gguf/
# Filename pattern: {ModelName}.{Quantization}.gguf

llama-cli -m ./outputs/unsloth-nemotron-function-calling-gguf/Llama-3.1-Nemotron-Nano-4B-v1.1.Q4_K_M.gguf \
    -p "What's the weather in Tokyo?" --ctx-size 4096
```

### With vLLM (Merged Model)

> **Requires**: Merged model format (`--merge` flag during training)

```bash
# Create separate venv for vLLM (recommended - different dependencies)
uv venv .venv-vllm --python 3.12
source .venv-vllm/bin/activate

# Install vLLM
uv pip install vllm openai

# Start OpenAI-compatible API server
# Note: --max-model-len limits context to fit in VRAM (adjust based on your GPU)
vllm serve ./outputs/unsloth-nemotron-function-calling-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```

> **OOM Error?** Reduce `--max-model-len` (try 2048) or increase `--gpu-memory-utilization 0.95`

Query the server (in another terminal):

```bash
# Activate the vLLM venv
source .venv-vllm/bin/activate

# Run inference
uv run infer-vllm
uv run infer-vllm --prompt "Book a flight to Tokyo for tomorrow"
```

| Format       | vLLM Compatible | Notes                    |
| ------------ | --------------- | ------------------------ |
| LoRA Adapter | ❌              | Use `--merge` to convert |
| Merged Model | ✅              | Full GPU inference       |
| GGUF         | ❌              | Use llama.cpp instead    |

## Installation (Alternative Methods)

### Manual Installation (without UV)

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# For CUDA 12.1+
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or for CUDA 11.8
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"

pip install peft transformers datasets trl accelerate
```

## Troubleshooting

### "Cannot find Unsloth kernels"

Check CUDA version and reinstall:

```bash
nvcc --version

# For CUDA 12.1+
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# For CUDA 11.8
uv pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
```

### Slow Training

Already optimized by default with:

- `use_gradient_checkpointing="unsloth"`
- `packing=True`

## Project Structure

```
unsloth-finetune-demo/
├── src/
│   └── unsloth_demo/
│       ├── __init__.py       # Package init
│       ├── config.py         # Constants & configuration
│       ├── data.py           # Dataset loading & formatting
│       ├── model.py          # Model loading & saving
│       ├── training.py       # Training logic
│       └── cli/
│           ├── __init__.py
│           ├── finetune.py   # Training CLI entry point
│           └── inference.py  # Inference CLI entry points
├── docs/
│   ├── architecture.md       # System overview with diagrams
│   ├── training.md           # Training pipeline details
│   └── inference.md          # Inference options guide
├── outputs/                  # Training outputs (auto-created)
├── pyproject.toml            # Dependencies & entry points
└── README.md                 # This file
```

## Documentation

- [Architecture Overview](docs/architecture.md) - Package structure and data flow diagrams
- [Training Guide](docs/training.md) - Detailed training pipeline explanation
- [Inference Guide](docs/inference.md) - All inference options explained

## References

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Nemotron Model Card](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1)
- [Glaive Dataset](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

## License

MIT
