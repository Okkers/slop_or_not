# Slop or not?
GML machine-text detection task: classify an excerpt as human-written (0) or machine-generated (1)
from its function-word Word Adjacency Network (WAN).

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

## 1. Data

Put in `data/`:
- `function_words.txt` (from the `wan-creator` branch, `test_wan/function_words.txt`)
- the dataset CSV made by `uv run python -m data_factory.data_generator`
  (columns `task_prompt, output_text, WAN, label`; label 0 = human, 1 = machine)

Optional but needed for the cross-LLM experiment (E2): a `generator` column naming who wrote
each row (`human`, `Qwen/Qwen2.5-1.5B-Instruct`, ...).

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
uv run main.py --data_path data/persuade_w_wan.csv --model GAT --itr 5      # 5 seeds, prints mean ± std
```
Models (`--model`): `GCN`, `GAT` (`--gat_v2 true` for GATv2), `SAGE` (GraphSAGE), and `MLP`,
a control that ignores the edges (does the graph structure help at all?).

What happens:
1. every CSV row becomes a 117-node graph (`data_factory/graph_builder.py`). Node i is always
   function word i, so graphs are aligned across excerpts (handy for GNNExplainer).
   The WAN is rebuilt from `output_text` with Christina's code in `vis/utils.py` (printing switched off
   with `VERBOSE = False`, ~20 ms per essay; graphs are cached in `data/cache/` after the first run).
   `--wan_source csv` uses the CSV's WAN column instead.
2. rows are split 70/15/15, stratified by label; the split is saved in `splits/` and reused.
3. the model trains with Adam + cross-entropy; the epoch with the lowest validation loss is kept.
4. the test set is scored once: accuracy, precision, recall, F1 (primary metric), ROC-AUC.

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--node_features` | `count,freq,emb` | log count, frequency per 100 tokens, pre-trained word vector (`onehot` = old word-identity feature) |
| `--embeddings` | `glove-wiki-gigaword-100` | which pre-trained word vectors (downloaded once to `~/gensim-data`) |
| `--extra_node_cols` | | CSV columns holding a JSON list of 117 values each (e.g. per-word perplexity) |
| `--edge_weight` | `wan` | `binary` = unweighted ablation |
| `--undirected` | `false` | symmetrise the WAN |
| `--window` | `5` | WAN window width |
| `--split` | `random` | `group` (hold out whole `--group_col` values, e.g. prompts) or `generator` (E2) |
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

## 3. SVM Baseline (`run_baselines.py`)

The course requires an SVM baseline. They use the same CSV, split and results table:
```
uv run run_baselines.py --data_path data/persuade_w_wan.csv
```
- `SVM-<features>`: SVM on `freq` (function-word frequencies), `freq+graphstats` (plus 16 graph-level
  WAN statistics) and `freq+wan` (plus the flattened WAN). Kernel and C are picked on the validation set.

## 4. Run everything at once

`bash scripts/main_comparison.sh data/persuade_w_wan.csv` trains all 4 GNNs (5 seeds each) and SVM baseline
on the same split, then prints the summary table.
Summarise results at any time: `uv run python -m vis.summarize` (add `--markdown` for a report-ready table).

Other graph types (e.g. the content-word graph for E1) can be added in `data_factory/data_loader.py::build_all_graphs`.

## 5. Notes

- **Root weight.** In plain GCN a node's own one-hot and its neighbours' one-hots go through the same
  weight matrix, so "of -> the" and "the -> of" look almost the same and word order is lost. On the sanity
  set, GCN reached 57% without the separate root term and ~90% with it. Turn it off with `--root_weight false`
  if you need the textbook GCN.
- **Model selection.** Early stopping uses validation *loss*: on balanced data, predicting "machine" for
  everything already gives F1 = 0.67, so picking epochs by F1 can keep a useless model.