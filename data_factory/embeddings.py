import numpy as np


def load_vectors(name: str):
    import gensim.downloader
    return gensim.downloader.load(name)


def vectors_for(words, wv):
    """One row per word (same order as `words`), plus the list of words that had no vector."""
    out = np.zeros((len(words), wv.vector_size))
    missing = []
    for i, w in enumerate(words):
        if w in wv:
            out[i] = wv[w]
        else:
            missing.append(w)
    return out, missing