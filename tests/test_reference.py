"""Differential tests: python/architecture.py (NumPy) vs python/microgpt.py (scalar).

Ground truth is Karpathy's code executed verbatim, everything above the training
loop (tokenizer, autograd, seeded random init, and gpt() itself). Training is
skipped, since identical weights on both sides are all a computation test needs.

Both sides run float64, so they should agree to ~machine precision. The 1e-9
tolerance flags real bugs: an architecture mistake shifts logits by ~1e0.
"""

import contextlib
import io
import os
from pathlib import Path

import numpy as np
import pytest

import architecture as arch

MICROGPT = Path(__file__).resolve().parents[1] / "python" / "microgpt.py"
TRAIN_MARKER = "# Let there be Adam"


@pytest.fixture(scope="module")
def ref(tmp_path_factory) -> dict:
    """Run microgpt.py's setup (no training) in a temp dir on a tiny corpus.
    random.seed(42) at its top makes the initial weights deterministic."""
    workdir = tmp_path_factory.mktemp("microgpt")
    (workdir / "input.txt").write_text("anna\nben\ncara\ndan\nelle\nfinn\n")
    setup = MICROGPT.read_text().split(TRAIN_MARKER)[0]
    ns: dict = {}
    cwd = os.getcwd()
    os.chdir(workdir)  # so microgpt.py finds input.txt instead of downloading
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(setup, str(MICROGPT), "exec"), ns)
    finally:
        os.chdir(cwd)
    ns["weights"] = arch.weights_from_microgpt(ns["state_dict"])
    return ns


def test_logits_match_at_every_position(ref):
    """Same tokens through both implementations -> same logits everywhere.
    Any wrong intermediate step would contaminate the logits, so this
    transitively validates the whole forward pass."""
    n_layer, n_head = ref["n_layer"], ref["n_head"]
    tokens = [ref["BOS"]] + [ref["uchars"].index(c) for c in "anna"] + [ref["BOS"]]

    s_keys = [[] for _ in range(n_layer)]
    s_vals = [[] for _ in range(n_layer)]
    a_keys = [[] for _ in range(n_layer)]
    a_vals = [[] for _ in range(n_layer)]
    for pos, tok in enumerate(tokens):
        expected = np.array([v.data for v in ref["gpt"](tok, pos, s_keys, s_vals)])
        got = arch.gpt(tok, pos, a_keys, a_vals, ref["weights"], n_head)
        diff = float(np.max(np.abs(got - expected)))
        assert diff < 1e-9, f"logits diverge at pos {pos}: max abs diff {diff:.3e}"


def test_greedy_generation_matches(ref):
    """End-to-end: greedy argmax must pick the identical token at every step."""
    n_layer, n_head, block_size = ref["n_layer"], ref["n_head"], ref["block_size"]

    def run(gpt_fn, to_scalar):
        keys = [[] for _ in range(n_layer)]
        vals = [[] for _ in range(n_layer)]
        token, out = ref["BOS"], []
        for pos in range(block_size):
            logits = gpt_fn(token, pos, keys, vals)
            token = max(range(ref["vocab_size"]), key=lambda i: to_scalar(logits[i]))
            if token == ref["BOS"]:
                break
            out.append(token)
        return out

    scalar = run(ref["gpt"], lambda v: v.data)
    tensor = run(lambda t, p, k, v: arch.gpt(t, p, k, v, ref["weights"], ref["n_head"]),
                 float)
    assert tensor == scalar


def test_softmax_is_stable():
    """exp(1000) overflows float64; the max-subtraction must prevent that."""
    p = arch.softmax(np.array([1000.0, 0.0]))
    assert np.isfinite(p).all() and abs(float(p.sum()) - 1.0) < 1e-12
