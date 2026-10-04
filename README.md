# transformer.cpp 🚀

Learn Transformers by reading a single C++ file.

A complete GPT-style Transformer implemented from scratch in one file with zero external dependencies. The project includes tokenization, embeddings, multi-head self-attention, backpropagation, AdamW optimization, model serialization, and text generation, all in plain C++.

---

## Features

- **Zero External Dependencies:** Built entirely with standard C++.
- **Complete Architecture:** Implements tokenization, embeddings, and multi-head self-attention.
- **Training & Optimization:** Includes backpropagation and AdamW optimization.
- **Persistence & Generation:** Supports model serialization and text generation.

---

## Quick Start

```bash
git clone https://github.com/messagecompass/transformer.cpp.git
cd transformer.cpp
g++ -O3 -march=native -ffast-math transformer.cpp -o transformer
./transformer "once upon a time"