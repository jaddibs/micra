"""Write and read .micra checkpoints (format version 1, docs/format.md).

Layout: 76-byte header | vocabulary | zero padding to 32-byte boundary |
float32 little-endian tensors in fixed order (wte, wpe, per-layer, lm_head last).

The Python reader exists for tests and tooling. It does the same validation
as the C++ loader, so the two can be checked against each other.
"""

import hashlib
import struct
from pathlib import Path

import numpy as np

MAGIC = b"MICRA\x00\x00\x00"
VERSION = 1
DTYPE_FP32 = 1
HEADER_FMT = "<8s9I8I"  # magic, 9 header fields, reserved[8]
HEADER_BYTES = struct.calcsize(HEADER_FMT)  # 76
MAX_DIM = 1 << 20  # conservative sanity limit on any single dimension


def align32(offset: int) -> int:
    return (offset + 31) // 32 * 32


def tensor_order(n_layer: int) -> list:
    """Fixed v1 tensor order. lm_head is last (docs/format.md), even though
    microgpt's state_dict inserts it third."""
    names = ["wte", "wpe"]
    for i in range(n_layer):
        names += [f"layer{i}.{t}" for t in
                  ("attn_wq", "attn_wk", "attn_wv", "attn_wo", "mlp_fc1", "mlp_fc2")]
    return names + ["lm_head"]


def expected_shapes(n_layer: int, n_embd: int, block_size: int,
                    vocab_size: int) -> dict:
    shapes = {"wte": (vocab_size, n_embd), "wpe": (block_size, n_embd),
              "lm_head": (vocab_size, n_embd)}
    for i in range(n_layer):
        for t in ("attn_wq", "attn_wk", "attn_wv", "attn_wo"):
            shapes[f"layer{i}.{t}"] = (n_embd, n_embd)
        shapes[f"layer{i}.mlp_fc1"] = (4 * n_embd, n_embd)
        shapes[f"layer{i}.mlp_fc2"] = (n_embd, 4 * n_embd)
    return shapes


def write_checkpoint(path, uchars: str, weights: dict, n_head: int) -> dict:
    """Serialize weights to a .micra file; returns a manifest dict.

    Config is derived from the weight shapes (wte -> vocab_size/n_embd,
    wpe -> block_size, layer count from keys); only n_head must be given.
    """
    n_layer = sum(1 for k in weights if k.endswith(".attn_wq"))
    vocab_size, n_embd = weights["wte"].shape
    block_size = weights["wpe"].shape[0]

    if len(uchars) != vocab_size - 1:
        raise ValueError(f"uchars has {len(uchars)} chars; wte implies "
                         f"vocab_size {vocab_size} (needs {vocab_size - 1})")
    if n_head <= 0 or n_embd % n_head != 0:
        raise ValueError(f"n_embd ({n_embd}) not divisible by n_head ({n_head})")
    shapes = expected_shapes(n_layer, n_embd, block_size, vocab_size)
    if set(weights) != set(shapes):
        raise ValueError(f"tensor names mismatch: {set(weights) ^ set(shapes)}")
    for name, shape in shapes.items():
        if weights[name].shape != shape:
            raise ValueError(f"{name}: shape {weights[name].shape}, expected {shape}")

    order = tensor_order(n_layer)
    vocab_bytes = 4 + 4 * len(uchars)
    header = struct.pack(HEADER_FMT, MAGIC, VERSION, DTYPE_FP32, n_layer, n_embd,
                         n_head, block_size, vocab_size, vocab_bytes, len(order),
                         *([0] * 8))
    vocab = struct.pack(f"<I{len(uchars)}I", len(uchars),
                        *(ord(c) for c in uchars))
    padding = b"\x00" * (align32(HEADER_BYTES + vocab_bytes)
                         - HEADER_BYTES - vocab_bytes)
    tensors = b"".join(np.ascontiguousarray(weights[n], dtype="<f4").tobytes()
                       for n in order)

    blob = header + vocab + padding + tensors
    Path(path).write_bytes(blob)
    return {
        "path": str(path),
        "config": {"n_layer": n_layer, "n_embd": n_embd, "n_head": n_head,
                   "block_size": block_size, "vocab_size": vocab_size},
        "tensors": {n: shapes[n] for n in order},
        "params": sum(r * c for r, c in shapes.values()),
        "dtype": "fp32",
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def read_checkpoint(path) -> tuple:
    """Parse and validate a .micra file -> (config, uchars, weights).

    Same checks as the C++ loader: every claim the header makes is checked
    against the actual bytes before any tensor is trusted.
    """
    raw = Path(path).read_bytes()
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"file size {len(raw)} smaller than header ({HEADER_BYTES})")
    (magic, version, dtype, n_layer, n_embd, n_head, block_size, vocab_size,
     vocab_bytes, tensor_count, *reserved) = struct.unpack_from(HEADER_FMT, raw)

    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version} (reader supports {VERSION})")
    if dtype != DTYPE_FP32:
        raise ValueError(f"unsupported dtype {dtype}")
    if any(reserved):
        raise ValueError("reserved header words must be zero")
    dims = {"n_layer": n_layer, "n_embd": n_embd, "n_head": n_head,
            "block_size": block_size, "vocab_size": vocab_size}
    for name, value in dims.items():
        if not 0 < value <= MAX_DIM:
            raise ValueError(f"{name} = {value} out of range")
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd ({n_embd}) not divisible by n_head ({n_head})")
    if vocab_bytes != 4 + 4 * (vocab_size - 1):
        raise ValueError(f"vocab_bytes {vocab_bytes} inconsistent with "
                         f"vocab_size {vocab_size}")
    if tensor_count != 3 + 6 * n_layer:
        raise ValueError(f"tensor_count {tensor_count}, expected {3 + 6 * n_layer}")

    count = struct.unpack_from("<I", raw, HEADER_BYTES)[0]
    if count != vocab_size - 1:
        raise ValueError(f"vocab count {count} != vocab_size - 1")
    points = struct.unpack_from(f"<{count}I", raw, HEADER_BYTES + 4)
    uchars = "".join(chr(p) for p in points)

    shapes = expected_shapes(n_layer, n_embd, block_size, vocab_size)
    offset = align32(HEADER_BYTES + vocab_bytes)
    total = sum(r * c for r, c in shapes.values())
    if len(raw) != offset + 4 * total:
        raise ValueError(f"file size {len(raw)} != expected {offset + 4 * total}")

    weights = {}
    for name in tensor_order(n_layer):
        rows, cols = shapes[name]
        weights[name] = np.frombuffer(raw, dtype="<f4", count=rows * cols,
                                      offset=offset).reshape(rows, cols)
        offset += 4 * rows * cols
    return {"version": version, "dtype": dtype, **dims}, uchars, weights
