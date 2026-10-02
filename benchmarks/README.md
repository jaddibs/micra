# Benchmarks

micra against PyTorch on the same checkpoints. Both sides run the same
weights in fp32, decode one token at a time through the same KV-cached
architecture, and pick tokens greedily, so they should generate identical
text before any timing is compared.

## Test machine

Everything in [results/](results/) was measured on one workstation:

- NVIDIA GeForce RTX 2060 SUPER (8 GB, 34 SMs, compute capability 7.5,
  448 GB/s memory bandwidth on the spec sheet)
- driver 615.71.09, RHEL 8
- CUDA 13.4.92 (nvcc), gcc 8.5.0, CMake Release build
- Python 3.12.14 with torch 2.14.1+cu130 for the baselines, system Python
  3.6.8 for the export

## Results

- [01-tiny](results/01-tiny.md): correctness checks and first timings on
  the trained 4K-param names model
- [02-scaling](results/02-scaling.md): decode bandwidth, prefill, flash,
  and eager PyTorch at 3M / 25M / 152M params
- [03-compile](results/03-compile.md): the rematch against torch.compile

## Running it

### 1. Build micra and export the tiny model

```bash
mkdir -p data
curl -o data/names.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt
python python/export_microgpt.py --data data/names.txt --output artifacts/names.micra

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 2. Make bigger checkpoints

The names model is too small to test the kernels, so make bigger
checkpoints with seeded random weights. They run at the same speed as
trained ones:

```bash
python python/make_checkpoint.py --n-layer 4 --n-embd 256 --n-head 8 \
    --block-size 256 --output artifacts/bench-256.micra   # ~3M params
python python/make_checkpoint.py --n-layer 8 --n-embd 512 --n-head 8 \
    --block-size 256 --output artifacts/bench-512.micra   # ~25M params, ~100 MB
python python/make_checkpoint.py --n-layer 12 --n-embd 1024 --n-head 16 \
    --block-size 512 --output artifacts/bench-1024.micra  # ~150M params, ~600 MB
```

### 3. Time micra

```bash
./build/micra artifacts/names.micra --greedy --samples 20        # baseline kernels
./build/micra artifacts/names.micra --greedy --samples 20 --flash
./build/micra artifacts/names.micra --prompt "an" --greedy --samples 20
```

Each run prints prefill and decode ms/token. Run `--verify` once
before timing with a prompt.

Decode is limited by memory bandwidth, so the number to look at is weight
bytes divided by time per token. Prefill reuses the weights across the
prompt, so compare its ms/token with decode's.

### 4. Time PyTorch

```bash
python benchmarks/torch_baseline.py artifacts/names.micra --iters 20
```

This prints the median and p95 ms/token after a few warmup runs. There is
a torch.compile version too:

```bash
python benchmarks/torch_compiled.py artifacts/bench-512.micra --iters 20
```
