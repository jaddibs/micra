# 03: The torch.compile rematch

[02](02-scaling.md) showed eager PyTorch losing mostly to its own Python
dispatch. This is the fairer comparison: the same machine, build, and
checkpoints against torch.compile, which traces the forward pass once and
then runs it without Python in the loop.

## Setup

The baseline is benchmarks/torch_compiled.py with fullgraph compilation. To
compile into one static graph it had to be restructured:

- a preallocated KV cache written in place, instead of growing lists
- a causal mask, instead of shapes that grow every step
- heads vectorized with einsum, instead of a Python loop

Greedy samples matched micra and the eager baseline at every size as a
correctness check.

## Results

Decode ms/token, greedy, identical weights.

| Model | micra | eager torch | compiled torch | micra lead |
|---|---|---|---|---|
| names (4K) | 0.041 | 0.720 | 0.287 | 7.0x |
| bench-256 (3M) | 0.207 | 4.118 | 0.592 | 2.9x |
| bench-512 (25M) | 0.559 | 7.822 | 1.032 | 1.85x |
| bench-1024 (152M) | 1.836 | 19.967 | 6.941 | 3.8x |

- Compiling sped torch up 2.5x to 7.6x. That is the dispatch overhead going
  away, which backs up what 02 said about the eager numbers.
- micra is still ahead at every size. The closest is bench-512 at 1.85x,
  where both sides are mostly down to runtime and kernel efficiency.
- At 152M compiled torch works out to about 87 GB/s against micra's 330.
  Its own log says why: Inductor printed "Not enough SMs to use
  max_autotune_gemm mode". The card has 34 SMs, which is under the
  compiler's tuning threshold, so its matmul heuristics stay generic.

## Logs

All four compiled runs. The long sample strings for bench-256 and bench-512
are left out.

```
$ python benchmarks/torch_compiled.py artifacts/names.micra --iters 5
greedy sample: 'anan'
device: cuda, torch 2.14.1+cu130, mode default
ms/token over 5 runs (25 tokens): median 0.287, p95 0.288

$ python benchmarks/torch_compiled.py artifacts/bench-256.micra --iters 20
W0930 23:14:37.503000 879291 torch/_inductor/utils.py:1806] [0/0] Not enough SMs to use max_autotune_gemm mode
device: cuda, torch 2.14.1+cu130, mode default
ms/token over 20 runs (5120 tokens): median 0.592, p95 0.612

$ python benchmarks/torch_compiled.py artifacts/bench-512.micra --iters 20
W0930 23:15:02.637000 879700 torch/_inductor/utils.py:1806] [0/0] Not enough SMs to use max_autotune_gemm mode
device: cuda, torch 2.14.1+cu130, mode default
ms/token over 20 runs (5120 tokens): median 1.032, p95 1.053

$ python benchmarks/torch_compiled.py artifacts/bench-1024.micra --iters 10
W0930 23:15:32.955000 882179 torch/_inductor/utils.py:1806] [0/0] Not enough SMs to use max_autotune_gemm mode
greedy sample: 'w'
device: cuda, torch 2.14.1+cu130, mode default
ms/token over 10 runs (20 tokens): median 6.941, p95 6.979
```
