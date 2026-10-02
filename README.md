# micra

*MicroGPT Inference CUDA Runtime Architecture*

A from-scratch CUDA inference engine for GPT with custom kernels, KV caching, and optimized prefill and decode.

It runs models with the architecture of [Karpathy's microgpt](https://karpathy.github.io/2026/02/12/microgpt/),
a GPT trained in ~200 lines of dependency-free Python. Training stays in
Python, and micra takes over at inference:

```
microgpt.py --train--> weights --export--> model.micra --load--> ./micra (CUDA)
```

I've always been curious how LLMs answer so fast when a single token takes
billions of floating-point operations. Billions, for something that usually
isn't even a full word. I couldn't stop wondering what that looks like
inside a GPU, so I built it myself, one kernel at a time.

## The model

Char-level GPT with a KV cache. It is GPT-2 style with microgpt's
simplifications: RMSNorm without a learned scale, no biases, ReLU, learned
positional embeddings, and untied `wte` and `lm_head`. The default model
trained on the names dataset has 1 layer, n_embd 16, and 4 heads, which is
4,192 parameters. The checkpoint format is in
[docs/format.md](docs/format.md).

## Kernels

Inference has two phases that behave very differently, so each one gets its
own kernels. Prefill pushes the whole prompt through as one batch, so the
projections are matrix-matrix products (GEMM). That is compute bound, and
shared-memory tiling captures the weight reuse across tokens. Decode then
generates one token at a time, so everything is a matrix-vector product
(GEMV). That is memory bandwidth bound, since every weight is read once per
token, so the wins come from coalescing and from cutting kernel launches.

| Kernel | What it does |
|---|---|
| `LinearKernel` | decode GEMV: one block per output row, coalesced float4 loads, tree reduction |
| `GemmKernel` | prefill GEMM: 16x16 shared-memory tiles, with a padded weight tile to avoid bank conflicts |
| `AttentionKernel` | one block per (head, query position): scores, softmax, weighted sum of the cached values |
| `FlashAttentionKernel` (`--flash`) | the same attention in one pass with an online softmax, so the scores are never stored |
| `RmsNormKernel` | one block per row, sum-of-squares reduction |
| `EmbeddingKernel` | one thread per element |

The k/v projections write straight into the KV cache, so appending is free,
and causality needs no mask because each query only reads the rows before
it. The ReLU and the residual adds are fused into the linear kernels, which
drops launches per layer from 12 to 9. That matters because decode on small
models is bound by launch overhead, not math.

## Correctness

Each piece is tested against the one before it:

1. `python/microgpt.py` is Karpathy's scalar original, the ground truth.
2. `python/architecture.py` is the same math in NumPy. Tests hold every
   logit to within 1e-9 of the original.
3. `python/checkpoint.py` saves and loads `.micra` files, with round-trip
   tests.
4. `src/micra.cu` has to greedy-generate the same text as the NumPy version
   from the same checkpoint.
5. `--verify` runs a prompt through prefill and again through token-by-token
   decode, then compares the KV caches and logits on the GPU.

```bash
python -m pytest -q
./build/micra artifacts/names.micra --prompt "an" --verify
```

## Build and run

Needs an NVIDIA GPU, the CUDA toolkit, and CMake. The Python side needs
NumPy 1.17 or newer, plus pytest for the tests.

```bash
# train the tiny model and export a checkpoint (pure Python, a few minutes)
mkdir -p data
curl -o data/names.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt
python python/export_microgpt.py --data data/names.txt --output artifacts/names.micra

# build and run
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/micra artifacts/names.micra --samples 20
./build/micra artifacts/names.micra --prompt "an" --greedy --flash
```

Every run prints prefill and decode ms/token separately.

## Benchmarks

Decode time in ms per token on an RTX 2060 SUPER, fp32, greedy, with the
same weights and the same output text on every side. The bench models use
seeded random weights, which run at the same speed as trained ones.

| Model | Params | micra | eager PyTorch | torch.compile |
|---|---:|---:|---:|---:|
| names | 4K | 0.041 | 0.720 | 0.287 |
| bench-256 | 3M | 0.207 | 4.118 | 0.592 |
| bench-512 | 25M | 0.559 | 7.822 | 1.032 |
| bench-1024 | 152M | 1.836 | 19.967 | 6.941 |

- At 152M params decode sustains 330 GB/s, 74% of the card's 448 GB/s
  memory bandwidth. At 3M it is only 14%, because launch overhead dominates
  until the model gets big.
- micra is 11x to 20x faster than eager PyTorch and 1.85x to 7x faster than
  torch.compile. Eager torch turned out to be bound by Python dispatch, not
  by its kernels.
- Prefill is 6x to 8.5x cheaper per token than decode at 25M params, and
  3.8x at 152M.
- The flash kernel is 2.5x to 2.9x slower than the baseline attention on
  long generations. It gives up parallelism across timesteps to save
  memory, and at a context of 256 that trade doesn't pay off.

The test machine and the full write-ups with logs are in
[benchmarks/](benchmarks/README.md).

## Layout

```
src/micra.cu               the whole runtime
python/microgpt.py         Karpathy's original
python/architecture.py     the forward pass in NumPy
python/checkpoint.py       the .micra checkpoint format
python/export_microgpt.py  train and export the names model
python/make_checkpoint.py  random checkpoints of any size, for benchmarking
tests/                     tests for the NumPy version and the format
benchmarks/                PyTorch baselines and results
docs/format.md             the .micra format, byte by byte
```

## Credits

`python/microgpt.py` is Andrej Karpathy's
[microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95),
included unmodified. The names dataset comes from his
[makemore](https://github.com/karpathy/makemore).
