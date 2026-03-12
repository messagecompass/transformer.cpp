# Transformer Forward Pass Test

## Files
- `train_and_export.py` — trains a tiny PyTorch transformer, exports weights and reference logits
- `transformer_test.cpp` — scalar C++ forward pass, loads weights, compares against PyTorch

## Steps

### 1. Train and export weights
```bash
python train_and_export.py
```
This produces:
- `weights.bin` — trained weights in raw float32
- `ref_logits.txt` — PyTorch logits for comparison

### 2. Compile the C++ program
```bash
g++ -O2 -o transformer_test transformer_test.cpp -lm
```

### 3. Run and compare
```bash
./transformer_test
```

Expected output if everything matches:
```
PASS: outputs match within floating-point tolerance.
```

## Notes
- All three files (weights.bin, ref_logits.txt, transformer_test) must be in the same directory,
  or adjust the paths at the top of main() in transformer_test.cpp.
- The model is tiny (D_MODEL=32, 4 layers, 4 heads, VOCAB=256) purely for testing purposes.
- The C++ uses ReLU; the PyTorch model uses the same, so they should match closely.
- Max absolute difference should be well under 1e-4 if the implementations agree.
