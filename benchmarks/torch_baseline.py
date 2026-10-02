"""PyTorch decode baseline for the same .micra checkpoint.

Runs the identical architecture (RMSNorm without scale, no biases, ReLU,
separate q/k/v, learned positions) one token at a time, so the comparison
against ./build/micra is like for like: same weights, same sequence of
matrix-vector products, fp32, greedy decoding.

Usage:
    python benchmarks/torch_baseline.py artifacts/names.micra --iters 20
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import checkpoint


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return x * (x.pow(2).mean() + 1e-5).rsqrt()


def forward(w: dict, cfg: dict, token: int, pos: int,
            keys: list, values: list) -> torch.Tensor:
    x = rmsnorm(w["wte"][token] + w["wpe"][pos])
    head_dim = cfg["n_embd"] // cfg["n_head"]
    for li in range(cfg["n_layer"]):
        residual = x
        x = rmsnorm(x)
        q = w[f"layer{li}.attn_wq"] @ x
        keys[li].append(w[f"layer{li}.attn_wk"] @ x)
        values[li].append(w[f"layer{li}.attn_wv"] @ x)
        K = torch.stack(keys[li])
        V = torch.stack(values[li])
        attn = torch.empty_like(x)
        for h in range(cfg["n_head"]):
            hs = slice(h * head_dim, (h + 1) * head_dim)
            scores = K[:, hs] @ q[hs] / head_dim**0.5
            attn[hs] = torch.softmax(scores, 0) @ V[:, hs]
        x = w[f"layer{li}.attn_wo"] @ attn + residual
        residual = x
        x = rmsnorm(x)
        x = torch.relu(w[f"layer{li}.mlp_fc1"] @ x)
        x = w[f"layer{li}.mlp_fc2"] @ x + residual
    return w["lm_head"] @ x


def generate(w: dict, cfg: dict, uchars: str) -> tuple:
    keys = [[] for _ in range(cfg["n_layer"])]
    values = [[] for _ in range(cfg["n_layer"])]
    bos = cfg["vocab_size"] - 1
    token, out, steps = bos, [], 0
    for pos in range(cfg["block_size"]):
        logits = forward(w, cfg, token, pos, keys, values)
        steps += 1
        token = int(logits.argmax())
        if token == bos:
            break
        out.append(uchars[token])
    return "".join(out), steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config, uchars, weights = checkpoint.read_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    w = {name: torch.tensor(array, dtype=torch.float32, device=device)
         for name, array in weights.items()}

    for _ in range(args.warmup):
        generate(w, config, uchars)

    times_ms = []
    total_steps = 0
    for _ in range(args.iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        sample, steps = generate(w, config, uchars)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) / steps * 1000)
        total_steps += steps

    p95 = sorted(times_ms)[int(0.95 * (len(times_ms) - 1))]
    print(f"greedy sample: {sample!r}")
    print(f"device: {device.type}, torch {torch.__version__}")
    print(f"ms/token over {args.iters} runs ({total_steps} tokens): "
          f"median {statistics.median(times_ms):.3f}, p95 {p95:.3f}")


if __name__ == "__main__":
    main()
