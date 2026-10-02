# 02: Scaling runs, 3M / 25M / 152M parameters

Same machine and build as [01](01-tiny.md), on three checkpoints with
seeded random weights from make_checkpoint.py. The text they generate is
gibberish, but it is deterministic, so greedy output still works as a
parity check across runtimes.

## Checkpoints

| Model | Config | Params | Weights | Verify (prefill vs decode) |
|---|---|---|---|---|
| bench-256 | 4L / 256d / 8h / block 256 | 3,225,088 | 12.9 MB | PASS, relative 9.2e-07 |
| bench-512 | 8L / 512d / 8h / block 256 | 25,324,544 | 101.3 MB | PASS, relative 1.3e-06 |
| bench-1024 | 12L / 1024d / 16h / block 512 | 151,574,528 | 606.3 MB | PASS, relative 3.4e-06 |

## Decode

Greedy, 5 samples, taking the second run of each command (the first pays
warmup). GB/s is weight MB divided by ms per token, compared against the
card's 448 GB/s.

| Model | Baseline ms/tok | GB/s | % of 448 | Flash ms/tok |
|---|---|---|---|---|
| bench-256 | 0.207 | 62 | 14% | 0.595 |
| bench-512 | 0.559 | 181 | 40% | 1.374 |
| bench-1024 | 1.836 | 330 | 74% | 1.822 |

Bandwidth climbs from 14% to 74% of the card's spec as the model grows,
because the fixed launch overhead gets spread over more weight bytes.

The flash kernel is 2.5x to 2.9x slower on the two models that generate
long sequences. Its online softmax walks the timesteps one after another,
while the baseline scores them in parallel. What it saves is the score
array, and at block_size 256 that is only 1 KB.

bench-256 and bench-512 generate all the way to the 255-token limit, so
attention runs over the full cache. bench-1024 samples BOS after one token,
so its number comes from only 5 short-context tokens. The weights read per
token are the same either way, and that is what dominates at this size.

## Prefill vs decode

Same command with a long prompt of repeated `a` (63 or 200 characters, plus
BOS). The decode column is the no-prompt number from the table above, so
every ratio uses the same reference.

| Model | Prompt tokens | Prefill ms/tok | Decode ms/tok | Ratio |
|---|---|---|---|---|
| bench-512 | 64 | 0.066 | 0.559 | 8.5x |
| bench-512 | 201 | 0.093 | 0.559 | 6.0x |
| bench-1024 | 201 | 0.477 | 1.836 | 3.8x |

One prefill pass reads the weights once for the whole batch, and decode
reads them once per token. That reuse is where the ratio comes from. The
tiled GEMM sustains roughly 540 to 770 GFLOP/s on bench-512.

Per-token prefill cost rises a little from 64 to 201 tokens because causal
attention work grows with the square of the prompt length while the GEMM
work grows linearly.

The decode steps that followed the two bench-512 prompts ran at 0.590 and
0.743 ms/token. They only cover the later positions, where the cache is
longer. The bench-1024 prompt run printed no decode line because its first
sampled token was BOS.

## Eager PyTorch

benchmarks/torch_baseline.py on the same checkpoints, 20 timed runs each
(10 for bench-1024). The greedy text matched micra exactly at all three
sizes before timing.

| Model | micra ms/tok | torch median | torch p95 | micra advantage |
|---|---|---|---|---|
| names (01) | 0.041 | 0.720 | 0.755 | 17.6x |
| bench-256 | 0.207 | 4.118 | 4.378 | 19.9x |
| bench-512 | 0.559 | 7.822 | 7.902 | 14.0x |
| bench-1024 | 1.836 | 19.967 | 20.313 | 10.9x |

I expected cuBLAS to win once the model got big. It did not. At 152M torch
moves 606 MB in 19.97 ms, which is 30 GB/s, or 7% of the card. So it is
limited by eager-mode dispatch (hundreds of Python ops per token, a Python
loop over heads, a cache re-stack every step), not by its kernels. This is
shown by the gap narrowing from 20x to 11x as the real math grows.

## Logs

Trimmed to the lines the tables are built from (sample text left out).

Verify:

```
$ ./build/micra artifacts/bench-256.micra --prompt "an" --verify
prefill vs decode over 3 positions: max diff 1.43051e-05 on values up to 15.6329 (relative 9.15064e-07) -> PASS
$ ./build/micra artifacts/bench-512.micra --prompt "an" --verify
prefill vs decode over 3 positions: max diff 9.15527e-05 on values up to 68.6557 (relative 1.3335e-06) -> PASS
$ ./build/micra artifacts/bench-1024.micra --prompt "an" --verify
prefill vs decode over 3 positions: max diff 0.000766754 on values up to 224.957 (relative 3.40845e-06) -> PASS
```

Decode, baseline and flash:

```
$ ./build/micra artifacts/bench-256.micra --greedy --samples 5
prefill: 5 tokens in 2.52 ms (0.505 ms/token)
decode:  1275 tokens in 264.01 ms (0.207 ms/token)
$ ./build/micra artifacts/bench-256.micra --greedy --samples 5 --flash
prefill: 5 tokens in 2.15 ms (0.431 ms/token)
decode:  1275 tokens in 758.83 ms (0.595 ms/token)

$ ./build/micra artifacts/bench-512.micra --greedy --samples 5
prefill: 5 tokens in 9.07 ms (1.815 ms/token)
decode:  1275 tokens in 712.29 ms (0.559 ms/token)
$ ./build/micra artifacts/bench-512.micra --greedy --samples 5 --flash
prefill: 5 tokens in 8.21 ms (1.642 ms/token)
decode:  1275 tokens in 1752.44 ms (1.374 ms/token)

$ ./build/micra artifacts/bench-1024.micra --greedy --samples 5
prefill: 5 tokens in 39.74 ms (7.947 ms/token)
decode:  5 tokens in 9.18 ms (1.836 ms/token)
$ ./build/micra artifacts/bench-1024.micra --greedy --samples 5 --flash
prefill: 5 tokens in 39.63 ms (7.927 ms/token)
decode:  5 tokens in 9.11 ms (1.822 ms/token)
```

Prefill with long prompts:

```
$ ./build/micra artifacts/bench-512.micra --greedy --prompt (63 a's)
prefill: 64 tokens in 4.23 ms (0.066 ms/token)
decode:  192 tokens in 113.36 ms (0.590 ms/token)

$ ./build/micra artifacts/bench-512.micra --greedy --prompt (200 a's)
prefill: 201 tokens in 18.74 ms (0.093 ms/token)
decode:  55 tokens in 40.86 ms (0.743 ms/token)

$ ./build/micra artifacts/bench-1024.micra --greedy --prompt (200 a's)
prefill: 201 tokens in 95.90 ms (0.477 ms/token)
```

Eager PyTorch:

```
$ python benchmarks/torch_baseline.py artifacts/bench-256.micra --iters 20
device: cuda, torch 2.14.1+cu130
ms/token over 20 runs (5120 tokens): median 4.118, p95 4.378

$ python benchmarks/torch_baseline.py artifacts/bench-512.micra --iters 20
device: cuda, torch 2.14.1+cu130
ms/token over 20 runs (5120 tokens): median 7.822, p95 7.902

$ python benchmarks/torch_baseline.py artifacts/bench-1024.micra --iters 10
device: cuda, torch 2.14.1+cu130
ms/token over 10 runs (20 tokens): median 19.967, p95 20.313
```
