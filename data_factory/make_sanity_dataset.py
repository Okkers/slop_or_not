"""
Builds a small SANITY-CHECK dataset in exactly the same CSV format as data_generator.py
(task_prompt, output_text, WAN, label, generator), without needing an LLM.

    label 0  = a real excerpt of human text
    label 1  = an excerpt with its words shuffled inside small blocks

Each excerpt is used ONCE, as either the original or the shuffled version (coin flip), so
both classes have the same distribution of function-word COUNTS and only the word ORDER
(= the WAN edges) differs. A model that only looks at node features (--model MLP) should
therefore stay near 50% accuracy, while the GNNs should beat it. That makes it a quick
end-to-end test that the training pipeline really uses the graph structure.
It is NOT a stand-in for real machine-generated text.

(--paired puts both versions of every excerpt in the data. Then memorising methods can score
BELOW chance, because a test excerpt's most similar training graph is its own twin with the
opposite label.)

Each shuffled excerpt uses one of --shuffle_blocks (stored in the `generator` column,
e.g. "shuffle6"), so --split generator can be tested too.

Usage:
    uv run python -m data_factory.make_sanity_dataset --source data/some_folder_of_txt_files
    uv run python -m data_factory.make_sanity_dataset --source data/persuade_2.0_human_scores_demo_id_github.csv --text_col full_text --group_col assignment --max_docs 2000
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from vis import utils as wan_utils   # Christina's WAN code

wan_utils.VERBOSE = False


def load_sources(args):
    if args.source.endswith(".csv"):
        df = pd.read_csv(args.source)
        if args.max_docs:
            df = df.sample(n=min(args.max_docs, len(df)), random_state=args.seed)
        group = df[args.group_col] if args.group_col in df.columns else pd.Series(["csv"] * len(df))
        return list(zip(group.astype(str), df[args.text_col].astype(str)))
    files = sorted(glob.glob(os.path.join(args.source, "*.txt")))
    if not files:
        raise SystemExit(f"no .txt files in {args.source}")
    return [(os.path.splitext(os.path.basename(f))[0], open(f, encoding="latin-1").read()) for f in files]


def block_shuffle(tokens, block, rng):
    out = []
    for s in range(0, len(tokens), block):
        chunk = tokens[s:s + block]
        rng.shuffle(chunk)
        out.extend(chunk)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="folder of .txt files, or a .csv")
    p.add_argument("--text_col", default="full_text")
    p.add_argument("--group_col", default="assignment", help="CSV column used as task_prompt")
    p.add_argument("--out", default="data/sanity_wan.csv")
    p.add_argument("--function_words", default="data/function_words.txt")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--excerpt_words", type=int, default=300)
    p.add_argument("--max_per_source", type=int, default=120)
    p.add_argument("--max_docs", type=int, default=0)
    p.add_argument("--shuffle_blocks", default="6,12")
    p.add_argument("--paired", action="store_true", help="keep both versions of every excerpt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    blocks = [int(b) for b in args.shuffle_blocks.split(",")]

    rows = []

    def add(name, toks, block):
        versions = [(0, toks, "human"), (1, block_shuffle(list(toks), block, rng), f"shuffle{block}")]
        if not args.paired:
            versions = [versions[int(rng.integers(2))]]
        for lab, toks_out, gen in versions:
            txt = " ".join(toks_out)
            W = wan_utils.construct_wan_from_text(txt, windowWidth=args.window, functionWordsFile=args.function_words)
            rows.append({"task_prompt": name, "output_text": txt, "WAN": json.dumps(W),
                         "label": lab, "generator": gen})

    for name, text in load_sources(args):
        toks = wan_utils.cleanUp(text)   # same cleaning the WAN uses: strip punctuation, lowercase, split
        if len(toks) < args.excerpt_words:           # short documents (e.g. essays): use them whole
            add(name, toks, blocks[len(rows) % len(blocks)])
            continue
        starts = range(0, len(toks) - args.excerpt_words + 1, args.excerpt_words)
        for n_ex, s in enumerate(starts):
            if n_ex >= args.max_per_source:
                break
            add(name, toks[s:s + args.excerpt_words], blocks[n_ex % len(blocks)])

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"saved {len(df)} rows ({int(df.label.sum())} shuffled) from {df.task_prompt.nunique()} sources to {args.out}")


if __name__ == "__main__":
    main()
    