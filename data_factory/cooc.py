"""
Word co-occurrence graph per text built the same way as the WAN but over ALL words.

    uv run python -m data_factory.cooc --data_path data/persuade_w_wan.csv

adds a column `COOC` to the CSV (written back to the same file unless you give --out).
The cell is JSON:  {"nodes": [...words...], "counts": [...], "n_tokens": T_words,
                    "edges": [[a, b, count, pmi], ...]}      (a, b = positions in "nodes")

"""
import argparse
import json
import math
import time
from collections import Counter

import pandas as pd

from vis import utils as wan_utils   # using this for the cleanUp fn

wan_utils.VERBOSE = False


def build_cooc(text: str, window: int = 5) -> dict:
    words = wan_utils.cleanUp(text)
    nodes = sorted(set(words))
    index = {w: i for i, w in enumerate(nodes)}
    word_counts = Counter(words)

    pairs = Counter()
    for i, a in enumerate(words):
        for k in range(1, window + 1):
            if i + k >= len(words):
                break
            pairs[(index[a], index[words[i + k]])] += 1

    total = sum(pairs.values())
    out_tot, in_tot = Counter(), Counter()
    for (a, b), c in pairs.items():
        out_tot[a] += c
        in_tot[b] += c
    edges = [[a, b, c, round(math.log(c * total / (out_tot[a] * in_tot[b])), 4)]
             for (a, b), c in sorted(pairs.items())]

    return {"nodes": nodes, "counts": [word_counts[w] for w in nodes],
            "n_tokens": len(words), "edges": edges}


def main():
    p = argparse.ArgumentParser(description="Add a co-occurrence graph column (COOC) to a dataset CSV")
    p.add_argument("--data_path", required=True, help="CSV with an output_text column")
    p.add_argument("--out", default="", help="where to save (default: overwrite --data_path)")
    p.add_argument("--window", type=int, default=5, help="how many words ahead count as co-occurring")
    p.add_argument("--column", default="COOC")
    args = p.parse_args()

    df = pd.read_csv(args.data_path)
    t0 = time.time()
    graphs = [build_cooc(str(t), args.window) for t in df["output_text"]]
    df[args.column] = [json.dumps(g, separators=(",", ":")) for g in graphs]

    out = args.out or args.data_path
    df.to_csv(out, index=False)
    n_nodes = sum(len(g["nodes"]) for g in graphs) / len(graphs)
    n_edges = sum(len(g["edges"]) for g in graphs) / len(graphs)
    print(f"added '{args.column}' to {len(df)} rows in {time.time() - t0:.1f}s "
          f"(avg {n_nodes:.0f} words/nodes and {n_edges:.0f} edges per text) -> {out}")


if __name__ == "__main__":
    main()