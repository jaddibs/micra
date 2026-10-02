"""torch.compile decode baseline, the rematch after torch_baseline.py.

Same math, restructured so torch.compile can capture one static graph:
preallocated KV cache tensors instead of growing lists, a causal mask
instead of growing shapes, heads vectorized with einsum instead of a
python loop, and token/pos passed as length-1 tensors so the graph does
not re-specialize per step.

Usage:
    python benchmarks/torch_compiled.py artifacts/bench-512.micra --iters 20
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import checkpoint


def rmsnorm(x):
    return x * (x.pow(2).mean() + 1e-5).rsqrt()


def forward(w, cfg, k_cache, v_cache, positions, token, pos):
    n_layer, n_head, head_dim = cfg
    tok_emb = w["wte"].index_select(0, token).squeeze(0)
    pos_emb = w["wpe"].index_select(0, pos).squeeze(0)
    x = rmsnorm(tok_emb + pos_emb)

    for li in range(n_layer):
        residual = x
        x = rmsnorm(x)
        q = w[f"layer{li}.attn_wq"] @ x
        k = w[f"layer{li}.attn_wk"] @ x
        v = w[f"layer{li}.attn_wv"] @ x
        k_cache[li].index_copy_(0, pos, k.unsqueeze(0))
        v_cache[li].index_copy_(0, pos, v.unsqueeze(0))

        K = k_cache[li].view(k_cache.shape[1], n_head, head_dim)
        V = v_cache[li].view(v_cache.shape[1], n_head, head_dim)
        scores = torch.einsum("thd,hd->ht", K, q.view(n_head, head_dim))
        scores = scores / head_dim**0.5
        scores = scores.masked_fill(positions > pos, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        attn = torch.einsum("ht,thd->hd", weights, V).reshape(-1)

        x = w[f"layer{li}.attn_wo"] @ attn + residual
        residual = x
        x = rmsnorm(x)
        x = torch.relu(w[f"layer{li}.mlp_fc1"] @ x)
        x = w[f"layer{li}.mlp_fc2"] @ x + residual
    return w["lm_head"] @ x


def generate(step, config, uchars, device, k_cache, v_cache, positions):
    bos = config["vocab_size"] - 1
    tok, out, steps = bos, [], 0
    for pos in range(config["block_size"]):
        token_t = torch.tensor([tok], device=device)
        pos_t = torch.tensor([pos], device=device)
        logits = step(k_cache, v_cache, positions, token_t, pos_t)
        steps += 1
        tok = int(logits.argmax())
        if tok == bos:
            break
        out.append(uchars[tok])
    return "".join(out), steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    args = parser.parse_args()

    config, uchars, weights = checkpoint.read_checkpoint(args.checkpoint)
    device = torch.device("cuda")
    w = {name: torch.tensor(array, dtype=torch.float32, device=device)
         for name, array in weights.items()}
    cfg = (config["n_layer"], config["n_head"],
           config["n_embd"] // config["n_head"])
    cache_shape = (config["n_layer"], config["block_size"], config["n_embd"])
    k_cache = torch.zeros(cache_shape, device=device)
    v_cache = torch.zeros(cache_shape, device=device)
    positions = torch.arange(config["block_size"], device=device)

    compiled = torch.compile(forward, mode=args.mode, fullgraph=True)

    def step(kc, vc, pos_mask, token_t, pos_t):
        return compiled(w, cfg, kc, vc, pos_mask, token_t, pos_t)

    for _ in range(args.warmup):
        generate(step, config, uchars, device, k_cache, v_cache, positions)

    times_ms = []
    total_steps = 0
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        sample, steps = generate(step, config, uchars, device,
                                 k_cache, v_cache, positions)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) / steps * 1000)
        total_steps += steps

    p95 = sorted(times_ms)[int(0.95 * (len(times_ms) - 1))]
    print(f"greedy sample: {sample!r}")
    print(f"device: cuda, torch {torch.__version__}, mode {args.mode}")
    print(f"ms/token over {args.iters} runs ({total_steps} tokens): "
          f"median {statistics.median(times_ms):.3f}, p95 {p95:.3f}")


if __name__ == "__main__":
    main()
