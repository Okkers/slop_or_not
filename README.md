# Slop or not?
GML machine-text detection task: classify an excerpt as human-written (0) or machine-generated (1)
from a graph of the text: its function-word Word Adjacency Network (WAN) or its all-word co-occurrence graph (COOC).

### Setup
create a folder for holding data (preferably call it data/, or add the name to the .gitignore)

To run the repo, install uv => https://docs.astral.sh/uv/getting-started/installation/:
```
uv sync
uv run main.py
```

To add packages:
```
uv add PACKAGE_NAME
```

### Orientation
- data/ => self-explanatory (git-ignored)
- data_factory/ => Preprocessing, loading and formatting data
- exp/ => Model training, validation and testing here
- layers/ => model layers
- models/ => model variants (GNNs + non-GNN baselines)
- scripts/ => drivers for the experiments
- splits/ => saved train/val/test splits (commit these, so everyone uses the same rows)
- vis/ => utils for visualisation, metrics, results tables

---

## 0. The flow, per dataset (one CSV per LLM)

```
data_generator.py ──► data/<llm>.csv        task_prompt, output_text, WAN, label
        │
cooc.py ────────────► same CSV + COOC       co-occurrence graph of every text
        │
main.py --graph_type wan   ──┐
main.py --graph_type cooc  ──┼──► results/results.csv ──► vis.summarize (table)
run_baselines.py  (SVM)    ──┘
```

1. **Make the dataset for one LLM.** At the bottom of `data_factory/data_generator.py` set
   `model_name` (the LLM, e.g. `"Qwen/Qwen2.5-1.5B-Instruct"`), `path_out` (one file per LLM,
   e.g. `"data/qwen2.5-1.5b.csv"`) and the number of essays in `df.sample(n=20)`. Then:
   ```
   uv run python -m data_factory.data_generator
   ```
   Tip: `df.sample(n=..., random_state=2026)` gives every LLM the same human essays and prompts,
   so the datasets only differ in the machine text.
2. **Add the co-occurrence graphs** (adds a `COOC` column to the same file):
   ```
   uv run python -m data_factory.cooc --data_path data/qwen2.5-1.5b.csv
   ```
3. **Train + test on the WANs:**
   ```
   uv run main.py --data_path data/qwen2.5-1.5b.csv --graph_type wan --model GCN --itr 5
   ```
4. **Train + test on the co-occurrence graphs** (same command, other graph type):
   ```
   uv run main.py --data_path data/qwen2.5-1.5b.csv --graph_type cooc --model GCN --itr 5
   ```
5. **SVM baseline:** `uv run run_baselines.py --data_path data/qwen2.5-1.5b.csv`
6. **Results table:** `uv run python -m vis.summarize --data qwen2.5-1.5b.csv`

Steps 2-6 for all four GNNs (GCN, GAT, SAGE, MLP) in one go:
```
bash scripts/main_comparison.sh data/qwen2.5-1.5b.csv
```
Next LLM: back to step 1 with a new `model_name` and `path_out`. Every dataset gets its own split
file in `splits/`, and its rows in `results/results.csv` are told apart by the `data` column.

Within one dataset, WAN and COOC runs use the same split (same train/val/test rows), the same node
features (count, frequency, GloVe vector) and the same edge weights (0.75^k, row-normalised).
The only difference is which words are nodes: the 117 function words, or every word of the text.

## 1. Data

Put in `data/`:
- `function_words.txt` (from the `wan-creator` branch, `test_wan/function_words.txt`)
- the dataset CSVs made by `data_factory/data_generator.py`
  (columns `task_prompt, output_text, WAN, label`; label 0 = human, 1 = machine), plus `COOC` once
  `data_factory/cooc.py` has run on them

`COOC` holds one JSON object per row: `{"nodes": [words], "counts": [...], "n_tokens": N, "window": 5,
"edges": [[a, b, weight], ...]}`, where `a` and `b` are positions in `nodes`.

**No LLM output yet?** Build a sanity-check set in the same format from any human text:
```
uv run python -m data_factory.make_sanity_dataset --source data/persuade_2.0_human_scores_demo_id_github.csv --text_col full_text --group_col assignment --max_docs 2000
```
Half of the excerpts get their words shuffled in small blocks (label 1). Word counts are the same
in both classes, only word order differs, so the no-graph `MLP` should stay at ~50% while the
GNNs beat it. Use it to check the pipeline works, not as a result.

## 2. Train the GNNs (`main.py`)

```
uv run main.py --data_path data/persuade_w_wan.csv --model GCN
uv run main.py --data_path data/persuade_w_wan.csv --model GAT --itr 5                  # 5 seeds, prints mean ± std
uv run main.py --data_path data/persuade_w_wan.csv --model GAT --itr 5 --graph_type cooc
```
Models (`--model`): `GCN`, `GAT` (`--gat_v2 true` for GATv2), `SAGE` (GraphSAGE), and `MLP`,
a control that ignores the edges (does the graph structure help at all?).

What happens:
1. every CSV row becomes a graph (`data_factory/graph_builder.py`):
   - `--graph_type wan` (default): 117 nodes, node i is always function word i, so graphs are aligned
     across excerpts (handy for GNNExplainer). The WAN is rebuilt from `output_text` with Christina's code
     in `vis/utils.py` (printing switched off with `VERBOSE = False`, ~20 ms per essay); it is the same as
     the WAN column, and `--wan_source csv` reads that column instead.
   - `--graph_type cooc`: read from the `COOC` column; one node per unique word of that essay, so every
     graph has its own size (~170 nodes on the practice data).

   Graphs are cached in `data/cache/` after the first run.
2. rows are split 70/15/15, stratified by label; the split is saved in `splits/` and reused.
3. the model trains with Adam + cross-entropy; the epoch with the lowest validation loss is kept.
4. the test set is scored once: accuracy, precision, recall, F1 (primary metric), ROC-AUC.

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--graph_type` | `wan` | `cooc` = all-word co-occurrence graph from the `COOC` column |
| `--node_features` | `count,freq,emb` | log count, frequency per 100 tokens, pre-trained word vector (`onehot` = old word-identity feature, WAN only) |
| `--embeddings` | `glove-wiki-gigaword-100` | which pre-trained word vectors (downloaded once to `~/gensim-data`) |
| `--extra_node_cols` | | WAN only: CSV columns holding a JSON list of 117 values each (e.g. per-word perplexity) |
| `--edge_weight` | `wan` | `binary` = unweighted ablation |
| `--undirected` | `false` | symmetrise the graph |
| `--window` | `5` | window width; for `cooc` it must match the `--window` given to `cooc.py` |
| `--split` | `random` | `group` (hold out whole `--group_col` values, e.g. prompts) or `generator` |
| `--num_layers / --hidden_dim / --dropout` | `2 / 64 / 0.3` | |
| `--readout` | `sum` | `mean`, `max`, `meanmax` |
| `--root_weight` | `true` | separate weight for a node's own features in GCN/GAT (see note below) |
| `--es_metric` | `loss` | validation quantity used to choose the best epoch |
| `--is_training 0` | | only re-test a saved checkpoint |

`uv run main.py --help` lists everything.

Outputs:
- `checkpoints/<setting>/checkpoint.pth`, `history.csv` (per-epoch losses)
- `results/<setting>/metrics.json`, `predictions.csv` (per row: true label, P(machine), prompt, generator)
- `results/results.csv`: one row per run, shared with the baselines

`<setting>` contains the graph type (`_wan-w5_` or `_cooc-w5_`), so WAN and COOC runs never overwrite each other.

## 3. SVM and RF baseline (`run_baselines.py`)

The course requires an SVM baseline and RF. It uses the same CSV, split and results table as the GNNs:
```
uv run run_baselines.py --data_path data/persuade_w_wan.csv
```
- `SVM-<features>`: SVM on `freq` (function-word frequencies), `freq+graphstats` (plus 16 graph-level
  WAN statistics) and `freq+wan` (plus the flattened WAN). Kernel and C are picked on the validation set.
- It always uses the WAN: an SVM needs the same features for every essay, and co-occurrence graphs have
  different words per essay.

## 4. Run everything at once

`bash scripts/main_comparison.sh data/qwen2.5-1.5b.csv` runs steps 2-6 of the flow for one dataset:
adds the `COOC` column if it is missing, trains all 4 GNNs on WAN and on COOC graphs (5 seeds each),
runs the SVM baseline, and prints the table. `ITR=3` changes the number of seeds,
`EXTRA="--train_epochs 3"` passes extra flags to `main.py`.

Summarise results at any time: `uv run python -m vis.summarize` (one row per model, graph type, dataset
and split; `--data qwen2.5-1.5b.csv` for one dataset, `--markdown` for a report-ready table).

## 5. Notes

- **Root weight.** In plain GCN a node's own features and its neighbours' features go through the same
  weight matrix, so "of -> the" and "the -> of" look almost the same and word order is lost. On the sanity
  set (with one-hot features), GCN reached 57% without the separate root term and ~90% with it. Turn it off
  with `--root_weight false` if you need the textbook GCN.
- **Model selection.** Early stopping uses validation *loss*: on balanced data, predicting "machine" for
  everything already gives F1 = 0.67, so picking epochs by F1 can keep a useless model.
- **Different graph sizes.** COOC graphs have different numbers of nodes. The layers work on one node at a
  time and the readout adds each graph's nodes into one vector, so any size works. Words GloVe doesn't know
  (names, typos, numbers; ~5% of nodes on the practice data) get a vector of zeros.