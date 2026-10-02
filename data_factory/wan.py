"""
Fast, print-free Word Adjacency Network (WAN) construction.

This reproduces *exactly* the matrix produced by `vis.utils.construct_wan_from_text`
(the gabrielegan.com WAN script Christina adapted), with three differences:
  1. no console logging, so it is orders of magnitude faster on real essays;
  2. it also returns the raw function-word counts (needed as node features);
  3. window width / decay / function-word list are plain arguments, so they can
     be swept in the hyper-parameter sensitivity study.

Construction (same as the original script):
  - strip , . ! ? " : ; ) ( [ ]   -> lowercase -> split on whitespace
  - for every token i that is a function word (except the very last token),
    look at the next `window` tokens; every function word found at offset k
    (k = 0..window-1) adds decay**k to edge (word_i -> word_{i+1+k})
  - row-normalise, so W[i, j] ~ P(next function word is j | current is i)

`tests/test_wan.py` checks this against the original implementation.
"""
from __future__ import annotations

import numpy as np

PUNCTUATION = [',', '.', '!', '?', '"', ':', ';', ')', '(', '[', ']']


def load_function_words(path: str) -> list[str]:
    """Same ordering as the original script: alphabetically sorted, de-duplicated."""
    with open(path, "r") as handle:
        words = handle.read().split()
    return sorted(set(words))


def tokenize(text: str) -> list[str]:
    for mark in PUNCTUATION:
        text = text.replace(mark, "")
    return text.lower().split()


def build_wan(text: str, fwords: list[str], window: int = 5, decay: float = 0.75,
              normalize: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns
        W      : (n, n) float array, row-normalised WAN (raw scores if normalize=False)
        counts : (n,)   float array, how often each function word occurs
                 (like the original, the very last token is not counted)
        n_tok  : number of tokens in the cleaned text
    """
    n = len(fwords)
    index = {w: i for i, w in enumerate(fwords)}
    tokens = tokenize(text)
    L = len(tokens)
    ids = np.fromiter((index.get(t, -1) for t in tokens), dtype=np.int64, count=L)

    scores = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    if L < 2:
        return scores, counts, L

    src_all = ids[: L - 1]
    counts += np.bincount(src_all[src_all >= 0], minlength=n)

    for k in range(window):
        if 1 + k >= L:
            break
        src = ids[: L - 1 - k]
        dst = ids[1 + k:]
        m = (src >= 0) & (dst >= 0)
        if m.any():
            np.add.at(scores, (src[m], dst[m]), decay ** k)

    if normalize:
        row = scores.sum(axis=1, keepdims=True)
        scores = np.divide(scores, row, out=np.zeros_like(scores), where=row != 0)
    return scores, counts, L
