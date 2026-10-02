# 01: Tiny model, correctness and first timings

The trained names model from export_microgpt.py: 1 layer, n_embd 16,
4 heads, block 16, 4,192 params, 16,960 bytes. That is too small to compare
optimizations, so this run is mainly a correctness check.

## Correctness

| Check | Result |
|---|---|
| Export self-check (NumPy greedy sample) | `'anan'` |
| micra `--greedy`, baseline attention | `'anan'`, matches |
| micra `--greedy --flash` | `'anan'`, matches |
| micra `--prompt "an" --greedy` | `'anan'`, matches |
| `--prompt "an" --verify` (prefill vs decode, on device) | PASS, max diff 2.4e-07 |
| PyTorch baseline greedy sample, same file | `'anan'`, matches |

## Timings

| Timing | Value | Derived |
|---|---|---|
| micra decode | 0.041 ms/token | ~24,000 tok/s; 12 launches + 2 copies per token, so ~3 us per operation |
| micra decode, flash | 0.041 ms/token | identical to baseline at this size, as expected |
| micra prefill, batch of 1 | 0.052 ms/call (0.050 in the flash run) | one forward pass, slightly above decode (2D launch setup) |
| micra prefill, batch of 3 | 0.017 ms/token | same ~0.05 ms per call as batch of 1, so 3x the tokens for free |
| PyTorch decode | 0.720 ms/token median, 0.755 p95 | micra is 17.6x faster here |

At 4,192 parameters the math is ~8 kFLOP per token, which is nanoseconds.
Both runtimes appear to be measuring overhead, not kernels. micra is faster
since it pays ~3 us per kernel launch, while torch pays Python dispatch on
dozens of ops per token.

## Logs

Export (system Python 3.6):

```
$ python python/export_microgpt.py --data data/names.txt --output artifacts/names.micra
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.6497
manifest:
  path: artifacts/names.micra
  config: {'n_layer': 1, 'n_embd': 16, 'n_head': 4, 'block_size': 16, 'vocab_size': 27}
  params: 4192
  dtype: fp32
  bytes: 16960
  sha256: 2cbdbdbebc34150f546cd07f44e9cf9e28513f09ff798158173e747663b3c722
greedy sample from exported checkpoint: 'anan'
```

Build:

```
$ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
-- The CUDA compiler identification is NVIDIA 13.4.92
-- Check for working CUDA compiler: /usr/local/cuda-13/bin/nvcc - skipped
-- Build files have been written to: build
$ cmake --build build -j
[ 50%] Building CUDA object CMakeFiles/micra.dir/src/micra.cu.o
[100%] Linking CUDA executable micra
```

Greedy decode, all three variants (20 samples each, all 20 identical as
greedy requires, showing one):

```
$ ./build/micra artifacts/names.micra --greedy --samples 20
sample  1: anan
  ... (all 20 identical)
prefill: 20 tokens in 1.03 ms (0.052 ms/token)
decode:  80 tokens in 3.26 ms (0.041 ms/token)

$ ./build/micra artifacts/names.micra --greedy --samples 20 --flash
sample  1: anan
  ... (all 20 identical)
prefill: 20 tokens in 1.00 ms (0.050 ms/token)
decode:  80 tokens in 3.24 ms (0.041 ms/token)

$ ./build/micra artifacts/names.micra --prompt "an" --greedy --samples 20
sample  1: anan
  ... (all 20 identical)
prefill: 60 tokens in 1.01 ms (0.017 ms/token)
decode:  40 tokens in 1.65 ms (0.041 ms/token)
```

Prefill vs decode verification:

```
$ ./build/micra artifacts/names.micra --prompt "an" --verify
prefill vs decode over 3 positions: max diff 2.38419e-07 -> PASS
```

PyTorch baseline, same checkpoint:

```
$ python benchmarks/torch_baseline.py artifacts/names.micra --iters 20
greedy sample: 'anan'
device: cuda, torch 2.14.1+cu130
ms/token over 20 runs (100 tokens): median 0.720, p95 0.755
```
