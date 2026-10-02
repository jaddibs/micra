"""Checkpoint tests: round-trip, corruption rejection, determinism, and
byte-layout guarantees of the .micra format v1 (docs/format.md)."""

import struct
from pathlib import Path

import numpy as np
import pytest

import checkpoint as ckpt

# Small config, exercises n_layer > 1 and non-square shapes.
N_LAYER, N_EMBD, N_HEAD, BLOCK_SIZE = 2, 8, 2, 4
UCHARS = "abcd"  # vocab_size = 5 with BOS


@pytest.fixture
def weights() -> dict:
    rng = np.random.default_rng(0)
    vocab_size = len(UCHARS) + 1
    w = {
        "wte": rng.normal(size=(vocab_size, N_EMBD)),
        "wpe": rng.normal(size=(BLOCK_SIZE, N_EMBD)),
        "lm_head": rng.normal(size=(vocab_size, N_EMBD)),
    }
    for i in range(N_LAYER):
        for name in ("attn_wq", "attn_wk", "attn_wv", "attn_wo"):
            w[f"layer{i}.{name}"] = rng.normal(size=(N_EMBD, N_EMBD))
        w[f"layer{i}.mlp_fc1"] = rng.normal(size=(4 * N_EMBD, N_EMBD))
        w[f"layer{i}.mlp_fc2"] = rng.normal(size=(N_EMBD, 4 * N_EMBD))
    return w


@pytest.fixture
def path(tmp_path, weights) -> Path:
    p = tmp_path / "model.micra"
    ckpt.write_checkpoint(p, UCHARS, weights, n_head=N_HEAD)
    return p


def corrupt(path: Path, offset: int, data: bytes) -> Path:
    raw = bytearray(path.read_bytes())
    raw[offset:offset + len(data)] = data
    bad = path.with_name("bad.micra")
    bad.write_bytes(bytes(raw))
    return bad


def test_round_trip(path, weights):
    config, uchars, loaded = ckpt.read_checkpoint(path)
    assert uchars == UCHARS
    assert config == {"version": 1, "dtype": 1, "n_layer": N_LAYER, "n_embd": N_EMBD,
                      "n_head": N_HEAD, "block_size": BLOCK_SIZE,
                      "vocab_size": len(UCHARS) + 1}
    assert loaded.keys() == weights.keys()
    for name, w in weights.items():
        np.testing.assert_array_equal(loaded[name], w.astype(np.float32))


def test_deterministic_bytes(tmp_path, weights, path):
    other = tmp_path / "again.micra"
    ckpt.write_checkpoint(other, UCHARS, weights, n_head=N_HEAD)
    assert other.read_bytes() == path.read_bytes()


def test_tensor_payload_aligned_and_lm_head_last(path, weights):
    raw = path.read_bytes()
    vocab_bytes = 4 + 4 * len(UCHARS)
    offset = ckpt.align32(ckpt.HEADER_BYTES + vocab_bytes)
    assert offset % 32 == 0
    # padding between vocab and tensors is zero-filled
    assert raw[ckpt.HEADER_BYTES + vocab_bytes:offset] == b"\x00" * (
        offset - ckpt.HEADER_BYTES - vocab_bytes)
    # first tensor is wte, last is lm_head, both byte-exact (docs/format.md order)
    wte = weights["wte"].astype("<f4").tobytes()
    lm = weights["lm_head"].astype("<f4").tobytes()
    assert raw[offset:offset + len(wte)] == wte
    assert raw[-len(lm):] == lm


def test_rejects_bad_magic(path):
    with pytest.raises(ValueError, match="magic"):
        ckpt.read_checkpoint(corrupt(path, 0, b"XXXXX"))


def test_rejects_unsupported_version(path):
    with pytest.raises(ValueError, match="version"):
        ckpt.read_checkpoint(corrupt(path, 8, struct.pack("<I", 2)))


def test_rejects_nonzero_reserved(path):
    with pytest.raises(ValueError, match="reserved"):
        ckpt.read_checkpoint(corrupt(path, 44, struct.pack("<I", 7)))


def test_rejects_truncated_file(path):
    truncated = path.with_name("short.micra")
    truncated.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="size"):
        ckpt.read_checkpoint(truncated)


def test_rejects_trailing_garbage(path):
    padded = path.with_name("long.micra")
    padded.write_bytes(path.read_bytes() + b"\x00" * 4)
    with pytest.raises(ValueError, match="size"):
        ckpt.read_checkpoint(padded)


def test_write_rejects_wrong_shape(tmp_path, weights):
    weights["wpe"] = weights["wpe"][:, :-1]  # wrong n_embd
    with pytest.raises(ValueError, match="wpe"):
        ckpt.write_checkpoint(tmp_path / "x.micra", UCHARS, weights, n_head=N_HEAD)


def test_write_rejects_bad_head_split(tmp_path, weights):
    with pytest.raises(ValueError, match="n_head"):
        ckpt.write_checkpoint(tmp_path / "x.micra", UCHARS, weights, n_head=3)
