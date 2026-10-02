"""
Check that the fast WAN builder (data_factory/wan.py) gives the same matrix as the
original implementation in vis/utils.py.

Run:  uv run python -m tests.test_wan
"""
import contextlib
import io
import os
import time

import numpy as np

from data_factory.wan import build_wan, load_function_words
from vis.utils import construct_wan_from_text

FW_FILE = os.environ.get("FW_FILE", "data/function_words.txt")

SAMPLES = [
    "The cat and the dog walked in the garden, but the bird flew to the roof and waited in silence for the dog.",
    "I would wish not to be hasty in censuring any one; but I always speak what I think.",
    "Short.",
    "",
    "of of of of of of of the the",
]


def _original(text, window):
    with contextlib.redirect_stdout(io.StringIO()):   # the original prints *a lot*
        return np.array(construct_wan_from_text(text, windowWidth=window, functionWordsFile=FW_FILE), dtype=float)


def test_matches_original(extra_texts=()):
    fwords = load_function_words(FW_FILE)
    for text in list(SAMPLES) + list(extra_texts):
        for window in (1, 3, 5):
            W_fast, _, _ = build_wan(text, fwords, window=window)
            W_orig = _original(text, window)
            assert W_fast.shape == W_orig.shape
            assert np.allclose(W_fast, W_orig, atol=1e-12), f"mismatch (window={window}) on: {text[:60]!r}"


if __name__ == "__main__":
    extra = []
    longer = "test_wan/test_longer.txt"
    if os.path.exists(longer):
        extra.append(open(longer).read())
    test_matches_original(extra)
    print("OK: fast WAN == original WAN on all samples")

    if extra:
        fwords = load_function_words(FW_FILE)
        t0 = time.perf_counter(); _original(extra[0], 5); t_orig = time.perf_counter() - t0
        t0 = time.perf_counter(); build_wan(extra[0], fwords, 5); t_fast = time.perf_counter() - t0
        print(f"~1000-word text: original {t_orig:.2f}s, fast {t_fast * 1000:.1f}ms ({t_orig / t_fast:.0f}x faster)")
