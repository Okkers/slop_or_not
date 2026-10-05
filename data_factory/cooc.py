import argparse
import json
import time
from collections import Counter, defaultdict

import pandas as pd

from vis import utils as wan_utils   # only for Christina's cleanUp (lowercase, strip punctuation, split); ALL words are kept

wan_utils.VERBOSE = False


def window_scores(words: list, window: int = 5) -> dict:
    """{(a, b): score} before normalising: + 0.75**k when b is k+1 words after a (same as countFWs)."""
    scores = defaultdict(float)
    for i, a in enumerate(words):
        for k in range(min(window, len(words) - 1 - i)):
            scores[(a, words[i + 1 + k])] += 0.75 ** k
    return scores


def build_cooc(text: str, window: int = 5) -> dict:
    words = wan_utils.cleanUp(text)
    nodes = sorted(set(words))
    index = {w: i for i, w in enumerate(nodes)}
    word_counts = Counter(words)

    scores = window_scores(words, window)
    row_sum = defaultdict(float)
    for (a, _), s in scores.items():
        row_sum[a] += s
    edges = sorted([index[a], index[b], round(s / row_sum[a], 6)] for (a, b), s in scores.items())

    return {"nodes": nodes, "counts": [word_counts[w] for w in nodes],
            "n_tokens": len(words), "window": window, "edges": edges}


def main():
    p = argparse.ArgumentParser(description="Add a co-occurrence graph column (COOC) to a dataset CSV")
    p.add_argument("--data_path", required=True, help="CSV with an output_text column")
    p.add_argument("--out", default="", help="where to save (default: overwrite --data_path)")
    p.add_argument("--window", type=int, default=5, help="how many words ahead count as co-occurring (same as the WAN)")
    args = p.parse_args()

    df = pd.read_csv(args.data_path)
    t0 = time.time()
    graphs = [build_cooc(str(t), args.window) for t in df["output_text"]]
    df["COOC"] = [json.dumps(g, separators=(",", ":")) for g in graphs]

    out = args.out or args.data_path
    df.to_csv(out, index=False)
    n_nodes = sum(len(g["nodes"]) for g in graphs) / len(graphs)
    n_edges = sum(len(g["edges"]) for g in graphs) / len(graphs)
    print(f"added 'COOC' to {len(df)} rows in {time.time() - t0:.1f}s "
          f"(avg {n_nodes:.0f} words/nodes and {n_edges:.0f} edges per text) -> {out}")


if __name__ == "__main__":
    main()