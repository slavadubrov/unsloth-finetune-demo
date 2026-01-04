# Architecture

This document provides an overview of the Unsloth Fine-Tuning Demo architecture.

## Package Structure

```mermaid
graph TD
    subgraph "unsloth_demo package"
        CONFIG[config.py<br/>Constants & Settings]
        DATA[data.py<br/>Dataset Loading]
        MODEL[model.py<br/>Model Operations]
        TRAINING[training.py<br/>Training Logic]

        subgraph "cli/"
            CLI_FT[finetune.py<br/>Training CLI]
            CLI_INF[inference.py<br/>Inference CLI]
        end
    end

    CLI_FT --> CONFIG
    CLI_FT --> DATA
    CLI_FT --> MODEL
    CLI_FT --> TRAINING

    CLI_INF --> CONFIG
    CLI_INF --> MODEL

    DATA --> CONFIG
    MODEL --> CONFIG
    TRAINING --> CONFIG
```

## Module Responsibilities

| Module                                                   | Purpose                                |
| -------------------------------------------------------- | -------------------------------------- |
| [config.py](../src/unsloth_demo/config.py)               | All constants and configuration values |
| [data.py](../src/unsloth_demo/data.py)                   | Dataset loading and ChatML formatting  |
| [model.py](../src/unsloth_demo/model.py)                 | Model loading, LoRA setup, saving      |
| [training.py](../src/unsloth_demo/training.py)           | SFTTrainer configuration and execution |
| [cli/finetune.py](../src/unsloth_demo/cli/finetune.py)   | Command-line interface for training    |
| [cli/inference.py](../src/unsloth_demo/cli/inference.py) | Command-line interfaces for inference  |

---

## Training Pipeline

```mermaid
flowchart LR
    subgraph Input
        HF_MODEL[(HuggingFace<br/>Nemotron-4B)]
        HF_DATA[(Glaive Dataset<br/>113K examples)]
    end

    subgraph Processing
        LOAD[Load Model<br/>+ 4-bit Quantization]
        LORA[Apply LoRA<br/>Adapters]
        FORMAT[Format Dataset<br/>to ChatML]
        TRAIN[SFT Training<br/>with Packing]
    end

    subgraph Output
        ADAPTER[LoRA Adapter<br/>~100-500MB]
        MERGED[Merged Model<br/>~8-16GB]
        GGUF[GGUF Format<br/>~2-4GB]
    end

    HF_MODEL --> LOAD --> LORA --> TRAIN
    HF_DATA --> FORMAT --> TRAIN
    TRAIN --> ADAPTER
    ADAPTER -.->|--merge| MERGED
    ADAPTER -.->|--gguf| GGUF
```

### Key Optimizations

1. **4-bit Quantization** - Reduces memory usage by loading model weights in 4-bit precision
2. **LoRA Adapters** - Only trains ~1% of parameters, dramatically reducing memory and compute
3. **Unsloth Kernels** - Custom CUDA kernels for 2x faster training
4. **Gradient Checkpointing** - Trades compute for memory during backward pass
5. **Sequence Packing** - Efficiently packs multiple short sequences into one batch

---

## Inference Options

```mermaid
flowchart TD
    MODEL[Fine-tuned Model]

    MODEL --> LORA_PATH[LoRA Adapter]
    MODEL --> MERGED_PATH[Merged Model]
    MODEL --> GGUF_PATH[GGUF File]

    LORA_PATH --> UNSLOTH[Unsloth Inference<br/><code>uv run infer</code>]
    MERGED_PATH --> UNSLOTH
    MERGED_PATH --> VLLM[vLLM Server<br/><code>vllm serve</code>]
    GGUF_PATH --> LLAMA[llama.cpp<br/>CPU Inference]

    VLLM --> OPENAI[OpenAI API<br/><code>uv run infer-vllm</code>]
```

| Format       | Best For                  | Command                |
| ------------ | ------------------------- | ---------------------- |
| LoRA Adapter | Development, testing      | `uv run infer`         |
| Merged Model | Production, sharing, vLLM | `uv run infer` or vLLM |
| GGUF         | CPU deployment, edge      | llama.cpp              |

---

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI (finetune.py)
    participant Data as data.py
    participant Model as model.py
    participant Train as training.py
    participant HF as HuggingFace Hub

    User->>CLI: uv run finetune
    CLI->>Model: load_model_and_tokenizer()
    Model->>HF: Download Nemotron-4B
    HF-->>Model: Model + Tokenizer
    Model->>Model: Apply LoRA adapters
    Model-->>CLI: (model, tokenizer)

    CLI->>Data: load_glaive_dataset()
    Data->>HF: Download dataset
    HF-->>Data: Raw examples
    Data->>Data: Format to ChatML
    Data-->>CLI: Formatted dataset

    CLI->>Train: train(model, tokenizer, dataset)
    Train->>Train: SFTTrainer with packing
    Train-->>CLI: Trained model

    CLI->>Model: save_lora_adapter()
    Model-->>User: ./outputs/model/
```

---

## Configuration

All configurable values are centralized in [config.py](../src/unsloth_demo/config.py):

```python
# Model
MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
DATASET_NAME = "glaiveai/glaive-function-calling-v2"

# LoRA
LORA_R = 16           # Rank (more = better quality, slower)
LORA_ALPHA = 32       # Scaling factor (usually 2x rank)
MAX_SEQ_LENGTH = 4096 # Context window

# Training
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
```

To customize, edit these values before training or create a configuration override system.
