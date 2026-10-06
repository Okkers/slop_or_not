"""
Summarise results/results.csv into a mean ± std table (over seeds).

    uv run python -m vis.summarize                       # everything
    uv run python -m vis.summarize --model_id E3         # one experiment
    uv run python -m vis.summarize --data qwen.csv       # one dataset
    uv run python -m vis.summarize --by model,edge_weight --markdown
"""
import argparse
import os

import pandas as pd

METRICS = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc"]


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    by = [c for c in by if c in df.columns]
    g = df.groupby(by, dropna=False)
    out = g[METRICS].agg(["mean", "std"])
    table = pd.DataFrame(index=out.index)
    for m in METRICS:
        mean, std = out[(m, "mean")], out[(m, "std")].fillna(0)
        table[m.replace("test_", "")] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean, std)]
    table["runs"] = g.size()
    order = out[("test_f1", "mean")].sort_values(ascending=False).index
    return table.loc[order].reset_index()


def to_markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/results.csv")
    p.add_argument("--model_id", default="", help="only rows with this model_id")
    p.add_argument("--data", default="", help="only rows for this CSV file name")
    p.add_argument("--by", default="model_id,model,graph_type,data,split", help="columns to group by")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.results):
        raise SystemExit(f"{args.results} not found - train something first")
    df = pd.read_csv(args.results)
    if args.model_id:
        df = df[df.model_id == args.model_id]
    if args.data:
        df = df[df.data == args.data]
    table = summarize(df, args.by.split(","))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(to_markdown(table) if args.markdown else table.to_string(index=False))


if __name__ == "__main__":
    main()
    