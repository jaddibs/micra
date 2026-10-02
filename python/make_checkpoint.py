"""Create a .micra checkpoint of any size with seeded random weights.

Same architecture family and the same init as microgpt (gaussian, std 0.08),
just untrained. Random weights run at exactly the same speed as trained
ones, so these checkpoints exist to exercise and benchmark the runtime at
sizes where scalar training is impossible. Generation will be gibberish,
but it is deterministic gibberish: greedy decoding still gives identical
output on every runtime, so parity checks and --verify keep working.

Usage:
    python python/make_checkpoint.py --n-layer 8 --n-embd 512 --n-head 8 \
        --block-size 256 --output artifacts/bench-512.micra
"""

import argparse
from pathlib import Path

import numpy as np

import architecture
import checkpoint

UCHARS = "abcdefghijklmnopqrstuvwxyz"
INIT_STD = 0.08  # microgpt's matrix() init scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    vocab_size = len(UCHARS) + 1
    shapes = checkpoint.expected_shapes(args.n_layer, args.n_embd,
                                        args.block_size, vocab_size)
    weights = {name: rng.normal(0, INIT_STD, size=shape)
               for name, shape in shapes.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = checkpoint.write_checkpoint(args.output, UCHARS, weights,
                                           n_head=args.n_head)
    print("manifest:")
    for key in ("path", "config", "params", "dtype", "bytes", "sha256"):
        print(f"  {key}: {manifest[key]}")

    # sanity check: read it back and run a few greedy steps through the reference
    config, uchars, loaded = checkpoint.read_checkpoint(args.output)
    keys = [[] for _ in range(config["n_layer"])]
    values = [[] for _ in range(config["n_layer"])]
    token, out = config["vocab_size"] - 1, []
    for pos in range(min(config["block_size"], 8)):
        logits = architecture.gpt(token, pos, keys, values, loaded,
                                  config["n_head"])
        token = int(logits.argmax())
        if token == config["vocab_size"] - 1:
            break
        out.append(uchars[token])
    print(f"greedy sample (random weights): {''.join(out)!r}")


if __name__ == "__main__":
    main()
