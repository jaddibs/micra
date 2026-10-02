"""Train the scalar microgpt model verbatim and export a .micra checkpoint.

Runs everything in python/microgpt.py above its sampling loop (dataset,
tokenizer, seeded init, and the full training loop) exactly as Karpathy wrote
it, then writes the trained weights out with checkpoint.py. As a self-check
the exported file is read back and greedy generation is run on it through the
NumPy reference.

Usage:
    python python/export_microgpt.py --data data/names.txt --output artifacts/names.micra
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import architecture
import checkpoint

INFERENCE_MARKER = "# Inference: may the model babble back to us"


def train_microgpt(data_path: Path) -> dict:
    """Execute microgpt.py's setup + training loop verbatim; return its namespace."""
    source = (Path(__file__).parent / "microgpt.py").read_text()
    assert INFERENCE_MARKER in source, "microgpt.py changed; update INFERENCE_MARKER"
    train_source = source.split(INFERENCE_MARKER)[0]
    ns: dict = {}
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as workdir:
        shutil.copy(data_path, Path(workdir) / "input.txt")
        os.chdir(workdir)  # microgpt.py reads ./input.txt
        try:
            exec(compile(train_source, "microgpt.py", "exec"), ns)
        finally:
            os.chdir(cwd)
    print()  # training progress ends with \r
    return ns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="one document per line")
    parser.add_argument("--output", type=Path, required=True, help=".micra output path")
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(f"{args.data} not found (see README.md)")

    ns = train_microgpt(args.data)
    weights = architecture.weights_from_microgpt(ns["state_dict"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = checkpoint.write_checkpoint(
        args.output, "".join(ns["uchars"]), weights, n_head=ns["n_head"])

    print("manifest:")
    for key in ("path", "config", "params", "dtype", "bytes", "sha256"):
        print(f"  {key}: {manifest[key]}")

    # self-check: read the file back and greedy-generate through the NumPy reference
    config, uchars, loaded = checkpoint.read_checkpoint(args.output)
    keys = [[] for _ in range(config["n_layer"])]
    values = [[] for _ in range(config["n_layer"])]
    token, out = config["vocab_size"] - 1, []
    for pos in range(config["block_size"]):
        logits = architecture.gpt(token, pos, keys, values, loaded, config["n_head"])
        token = int(logits.argmax())
        if token == config["vocab_size"] - 1:
            break
        out.append(uchars[token])
    print(f"greedy sample from exported checkpoint: {''.join(out)!r}")


if __name__ == "__main__":
    main()
