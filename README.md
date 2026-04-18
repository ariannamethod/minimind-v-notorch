<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/ariannamethod/minimind-v?style=social)](https://github.com/ariannamethod/minimind-v/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/ariannamethod/minimind-v?v=1)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/ariannamethod/minimind-v)](https://github.com/ariannamethod/minimind-v/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/ariannamethod/minimind-v/pulls)

</div>

<div align="center">
  <h3>"The Greatest Path is the Simplest"</h3>
</div>

---

# MiniMind-V

A **67M-parameter Vision Language Model (VLM)** built entirely on **notorch** and the **Chuck optimizer** — no PyTorch, no CUDA, no pip-installed deep learning frameworks. Just C and Python (ctypes).

<div align="center">

![minimind-3v](./images/minimind-3v.gif)

</div>

---

## Overview

MiniMind-V is a minimalist multimodal model that can see images and hold conversations about them. It proves that you don't need multi-gigabyte frameworks to build and train a vision-language model.

| Property | Value |
|---|---|
| **Total parameters** | 67M (0.067B) |
| **Framework** | notorch (pure C library via ctypes) |
| **Optimizer** | Chuck (self-aware optimizer, also via ctypes) |
| **Vision encoder** | SigLIP2 patch embedding (siglip2-base-p16-ve) |
| **Text decoder** | Llama-style transformer (RMSNorm, RoPE, SwiGLU, causal attention) |
| **Vision projector** | Two-layer MLP mapping vision features to LLM hidden dim |
| **Dependencies** | Zero `pip install` dependencies — the C library builds from source |

### Model Variants

| Model | Size | Notes |
|---|---|---|
| minimind-3v | 67M | Dense architecture |
| minimind-3v-moe | 201M (A67M) | Mixture-of-Experts variant |

---

## Architecture

```
image → patches (C) → patch_embed (Linear) → vision_proj (MLP) → inject into token sequence
tokens → embedding → [attention + FFN] × L → rmsnorm → lm_head → logits
```

![VLM structure](./images/VLM-structure.jpg)

The model adds two submodules on top of a base language model:

1. **Visual Encoder** — extracts patches from the input image using the C library (`notorch_vision.h` / `stb_image.h`), then projects them through a linear patch embedding.
2. **Vision Projector** — a two-layer MLP that maps vision features into the same hidden dimension as the text decoder, allowing image tokens to be injected directly into the token sequence.

The text decoder is a standard Llama-style transformer with RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU activations, and causal multi-head attention.

### What is notorch?

**notorch** is a pure C neural network library (`notorch.c` / `notorch.h`) that replaces PyTorch entirely. It provides:

- Multi-dimensional tensors with reference counting
- Forward and backward passes (autograd tape)
- Common layers: Linear, Embedding, RMSNorm, etc.
- BLAS backend support (OpenBLAS / Apple Accelerate)
- Python bindings via ctypes (`notorch_nn.py`)

The **Chuck optimizer** (`chuck.py`) is a self-aware optimizer also backed by the C library.

---

## Repository Structure

```
minimind-v/
├── ariannamethod/              # notorch C library + Python bindings
│   ├── notorch.c / notorch.h   # Core C neural framework
│   ├── notorch_nn.py           # Python ctypes API (drop-in for torch.nn)
│   ├── notorch_vision.h        # Image loading, transforms, patch extraction
│   ├── notorch_vision_wrapper.c
│   ├── stb_image.h             # JPEG/PNG/BMP decoder (header-only)
│   ├── chuck.py                # Chuck optimizer (ctypes to C)
│   └── Makefile                # Build libnotorch.so / libnotorch_vision.so
├── model/
│   ├── model_vlm.py            # VLM model definition (notorch)
│   ├── tokenizer.json          # Tokenizer
│   └── tokenizer_config.json
├── trainer/
│   └── train_pretrain_vlm.py   # VLM pretraining script (notorch + Chuck)
├── dataset/
│   └── eval_images/            # Sample evaluation images
├── images/                     # Documentation images
├── LICENSE
└── CODE_OF_CONDUCT.md
```

---

## Quick Start

### 1. Build the C library

```bash
cd ariannamethod
make
```

This compiles `libnotorch.so` (or `.dylib` on macOS) and `libnotorch_vision.so`. No pip packages required.

### 2. Download the vision encoder

The model requires the SigLIP2 vision encoder weights:

```bash
git lfs install
git clone https://huggingface.co/jingyaogong/siglip2-base-p16-ve model/siglip2-base-p16-ve
```

### 3. Train (pretraining)

```bash
python -m trainer.train_pretrain_vlm
```

Options:

```bash
python -m trainer.train_pretrain_vlm \
  --data_path dataset/pretrain_i2t.parquet \
  --epochs 3 \
  --lr 4e-4 \
  --hidden_size 512 \
  --num_hidden_layers 8
```

### 4. Use the model

```python
from model.model_vlm import MiniMindVLM, VLMConfig

config = VLMConfig(hidden_size=512, num_hidden_layers=8)
model = MiniMindVLM(config)

# Load weights
model.load_weights("out/pretrain_vlm_512.bin")

# Run inference with an image
tokens = [1, 42, 100, 200]  # your token ids
logits = model.forward_inference(tokens, image_path="path/to/image.jpg")
```

---

## Design Philosophy

- **Zero dependencies**: no `requirements.txt`, no `pip install torch`. The C library compiles with `make`.
- **Transparency**: every layer, every operation is visible in ~2000 lines of C and ~1000 lines of Python.
- **Minimalism**: 67M parameters, single-GPU friendly, fast iteration.

---

## License

This project is licensed under the GNU GPLv3 License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [LLaVA](https://arxiv.org/pdf/2304.08485) — visual instruction tuning
- [LLaVA-VL](https://arxiv.org/pdf/2310.03744) — improved visual language alignment
- [MiniMind](https://github.com/jingyaogong/minimind) — the original LLM base
- [SigLIP2](https://huggingface.co/jingyaogong/siglip2-base-p16-ve) — vision encoder
