# The .micra checkpoint format (version 1)

A `.micra` file is a flat binary snapshot of one trained model. Everything is
little-endian. The Python writer/reader lives in `python/checkpoint.py`, and
the C++ loader in `src/micra.cu` performs the same validation.

## Layout

```
offset 0    header        76 bytes
offset 76   vocabulary    4 + 4 * (vocab_size - 1) bytes
            zero padding  up to the next 32-byte boundary
            tensors       float32, row-major, fixed order
```

The tensors start at a 32-byte-aligned offset,
`align32(header_bytes + vocab_bytes)`, so a reader that maps the file can
point a `float*` straight at them.

## Header (76 bytes)

| Offset | Size | Field | Value |
|---:|---:|---|---|
| 0 | 8 | magic | `"MICRA\0\0\0"` |
| 8 | 4 | version | 1 |
| 12 | 4 | dtype | 1 = float32 |
| 16 | 4 | n_layer | > 0 |
| 20 | 4 | n_embd | divisible by n_head |
| 24 | 4 | n_head | > 0 |
| 28 | 4 | block_size | > 0 |
| 32 | 4 | vocab_size | includes BOS |
| 36 | 4 | vocab_bytes | 4 + 4 * (vocab_size - 1) |
| 40 | 4 | tensor_count | 3 + 6 * n_layer |
| 44 | 32 | reserved | eight uint32 words, all zero |

Every field after the magic is a uint32. The reserved block is eight uint32
words to match, which leaves room for future versions.

## Vocabulary

A uint32 count (= vocab_size - 1) followed by that many uint32 Unicode code
points, one per token, in token-id order. BOS is implicit: its id is always
`vocab_size - 1` and it has no code point. Multi-character tokens would not
fit this fixed-width layout and would need a new format version.

The format allows any code point, but the CUDA runtime only loads ASCII
vocabularies for now.

## Tensors

Row-major float32, no per-tensor header, in this fixed order:

```
wte        [vocab_size, n_embd]
wpe        [block_size, n_embd]
layer{i}.attn_wq   [n_embd, n_embd]      for i = 0 .. n_layer-1
layer{i}.attn_wk   [n_embd, n_embd]
layer{i}.attn_wv   [n_embd, n_embd]
layer{i}.attn_wo   [n_embd, n_embd]
layer{i}.mlp_fc1   [4*n_embd, n_embd]
layer{i}.mlp_fc2   [n_embd, 4*n_embd]
lm_head    [vocab_size, n_embd]          always last
```

Every weight matrix `W [n_out, n_in]` is applied as `y = W @ x`.
`wte` and `lm_head` have the same shape but are separate, untied tensors.
