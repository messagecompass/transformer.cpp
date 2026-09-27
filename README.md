# TinyTransformer-CPP 🚀

A lightweight, zero-dependency educational C++ Transformer trained from scratch on the **TinyStories** dataset. 

This project is adapted from and inspired by the fantastic educational work in **[dratman/train_a_tiny_GPT_in_cpp](https://github.com/dratman/train_a_tiny_GPT_in_cpp)**. It implements a complete transformer pipeline—including a frequency-based tokenizer, multi-head causal self-attention, feed-forward layers with GeLU, a cosine learning rate schedule, and backpropagation—written entirely in native C++ with zero external libraries.

---

## ✨ Features
* **Zero Dependencies:** No PyTorch, no CUDA, no heavy math libraries. Just standard C++ (`<vector>`, `<cmath>`, `<string>`, `<random>`).
* **Custom Frequency-Based Tokenizer:** Automatically builds a vocabulary from text while properly preserving special tokens like `<|endoftext|>`.
* **Complete Training Pipeline:** Handles forward passes, loss calculation, backpropagation, gradient updates, and a cosine learning rate schedule from scratch.
* **CPU Optimized:** Designed to compile cleanly and run efficiently on standard CPU scalar loops.

---

## 🛠️ Requirements & Compilation
You only need a modern C++ compiler (`g++` or `clang++`). 

Compile the code with optimization flags (`-O3` is crucial for performance):
```bash
g++ -O3 -std=c++17 transformer.cpp -o transformer

🚀 Usage
1. Training the Model
To train the transformer on a TinyStories corpus (TinyStories-valid.txt):

Bash
./transformer train
(The model will save its weights to tinystories_model.bin upon completion).

2. Generating Text
To load the trained weights and prompt the model to generate a story:

Bash
./transformer "once upon a time"
🧠 Architectural Specifications
Architecture: Decoder-only Transformer (GPT-style)

Embedding Dimension (D_MODEL): 128

Layers (N_LAYERS): 4

Context Length (SEQ_LEN): 12

Activation Function: GeLU

Optimizer: Adam/SGD with Cosine LR Schedule

📊 Results & Progression
Through iterative debugging and custom tokenization tuning, training a 19,000+ vocabulary model over 20,000 steps successfully drove cross-entropy loss down to ~4.63, enabling the model to transition from random <unk> token spam into generating coherent multi-word phrase structures ("once upon a", "the little girl", "to play with her mom").

🔗 Credits & Acknowledgements
Based on the original implementation by Ralph Dratman: dratman/train_a_tiny_GPT_in_cpp.

📜 License
This project is open-source and free for educational and experimental use.