"""
microgpt.py's inference math, translated scalar-for-scalar into NumPy.

Same function names, same operation order, same state_dict keys as microgpt.py.
The only change is representation: list[list[Value]] becomes np.ndarray, and each
list comprehension becomes the array expression it was secretly computing.
This file is the project's numerical oracle (see tests/test_reference.py).
"""

import numpy as np


def linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    # microgpt: [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    # w is [n_out, n_in] row-major, so that IS a matrix-vector product.
    return w @ x


def softmax(logits: np.ndarray) -> np.ndarray:
    exps = np.exp(logits - logits.max())  # max-shift prevents overflow
    return exps / exps.sum()


def rmsnorm(x: np.ndarray) -> np.ndarray:
    ms = (x * x).mean()
    return x * (ms + 1e-5) ** -0.5


def gpt(token_id: int, pos_id: int, keys: list, values: list,
        state_dict: dict, n_head: int) -> np.ndarray:
    """One decode step -> logits. Mirrors microgpt.py's gpt() line for line.

    keys/values: one python list per layer; k/v vectors are appended per
    position, exactly like microgpt's KV cache.
    """
    tok_emb = state_dict['wte'][token_id]       # token embedding
    pos_emb = state_dict['wpe'][pos_id]         # position embedding
    x = tok_emb + pos_emb                       # joint token and position embedding
    x = rmsnorm(x)

    n_layer = len(keys)
    n_embd = x.shape[0]
    head_dim = n_embd // n_head

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        K = np.stack(keys[li])                  # [t, n_embd], cached positions 0..pos_id
        V = np.stack(values[li])
        x_attn = np.empty(n_embd)
        for h in range(n_head):
            hs = slice(h * head_dim, (h + 1) * head_dim)
            attn_logits = K[:, hs] @ q[hs] / head_dim**0.5
            attn_weights = softmax(attn_logits)
            x_attn[hs] = attn_weights @ V[:, hs]  # writing the slice = concat heads
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = x + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = np.maximum(x, 0)                    # relu
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    logits = linear(x, state_dict['lm_head'])
    return logits


def weights_from_microgpt(state_dict: dict) -> dict:
    """Convert microgpt's list[list[Value]] grids into float64 arrays."""
    return {name: np.array([[p.data for p in row] for row in grid], dtype=np.float64)
            for name, grid in state_dict.items()}
