"""
Train / val / test splits, saved to disk so EVERY model (GNNs, the SVM baseline, ...)
is evaluated on exactly the same rows (the proposal requires identical splits).

--split random     stratified by label (default 70/15/15)                 -> main experiments
--split group      hold out whole values of --group_col (e.g. task_prompt)
                   for testing; --test_groups picks them, otherwise random -> cross-topic robustness
--split generator  E2 (cross-LLM): machine rows whose --gen_col value is in
                   --test_generators go to test, other machine rows to
                   train/val; human rows are split randomly so every split
                   has both classes                                         -> generalisation to unseen LLMs

The split file is data/splits/<csv name>__<settings>__seed<split_seed>.json and
contains CSV row indices. If the file exists it is reused, never regenerated.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _random_split(idx, y, val_ratio, test_ratio, seed):
    idx = np.asarray(idx)
    strat = y if len(np.unique(y)) > 1 else None
    rest, test = train_test_split(idx, test_size=test_ratio, random_state=seed, stratify=strat)
    y_rest = pd.Series(y, index=idx).loc[rest].values
    strat = y_rest if len(np.unique(y_rest)) > 1 else None
    train, val = train_test_split(rest, test_size=val_ratio / (1 - test_ratio), random_state=seed, stratify=strat)
    return train, val, test


def make_splits(df: pd.DataFrame, args) -> dict[str, list[int]]:
    y = df["label"].values
    idx = np.arange(len(df))
    rng = np.random.default_rng(args.split_seed)

    if args.split == "random":
        train, val, test = _random_split(idx, y, args.val_ratio, args.test_ratio, args.split_seed)

    elif args.split == "group":
        groups = df[args.group_col].astype(str).values
        uniq = np.unique(groups)
        if args.test_groups:
            test_groups = [g.strip() for g in args.test_groups.split("|")]
            missing = set(test_groups) - set(uniq)
            if missing:
                raise ValueError(f"--test_groups not found in column '{args.group_col}': {missing}")
        else:
            n_test = max(1, int(round(args.test_ratio * len(uniq))))
            test_groups = list(rng.choice(uniq, size=n_test, replace=False))
        is_test = np.isin(groups, test_groups)
        test = idx[is_test]
        rest = idx[~is_test]
        train, val = train_test_split(rest, test_size=args.val_ratio / (1 - args.test_ratio),
                                      random_state=args.split_seed, stratify=y[rest])
        print(f"[split] held-out {args.group_col} values: {sorted(map(str, test_groups))}")

    elif args.split == "generator":
        if args.gen_col not in df.columns:
            raise ValueError(f"--split generator needs a '{args.gen_col}' column in the CSV "
                             "(e.g. 'human', 'Qwen/Qwen2.5-1.5B-Instruct', ...)")
        if not args.test_generators:
            raise ValueError("--split generator needs --test_generators (separate several with |)")
        test_gens = [g.strip() for g in args.test_generators.split("|")]
        gens = df[args.gen_col].astype(str).values
        human = idx[y == 0]
        machine_test = idx[(y == 1) & np.isin(gens, test_gens)]
        machine_rest = idx[(y == 1) & ~np.isin(gens, test_gens)]
        if len(machine_test) == 0:
            raise ValueError(f"no machine rows with {args.gen_col} in {test_gens}")
        h_rest, h_test = train_test_split(human, test_size=args.test_ratio, random_state=args.split_seed)
        rest = np.concatenate([h_rest, machine_rest])
        train, val = train_test_split(rest, test_size=args.val_ratio / (1 - args.test_ratio),
                                      random_state=args.split_seed, stratify=y[rest])
        test = np.concatenate([h_test, machine_test])
        print(f"[split] test generators: {test_gens}")
    else:
        raise ValueError(f"unknown --split '{args.split}'")

    return {"train": sorted(map(int, train)), "val": sorted(map(int, val)), "test": sorted(map(int, test))}


def split_file_path(args) -> str:
    stem = os.path.splitext(os.path.basename(args.data_path))[0]
    tag = args.split
    if args.split == "group":
        tag += f"-{args.group_col}" + (f"-{_slug(args.test_groups)}" if args.test_groups else "")
    elif args.split == "generator":
        tag += f"-{_slug(args.test_generators)}"
    return os.path.join(args.split_dir, f"{stem}__{tag}__v{args.val_ratio}-t{args.test_ratio}__seed{args.split_seed}.json")


def _slug(s: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in s)
    return keep[:40]


def get_splits(df: pd.DataFrame, args) -> dict[str, list[int]]:
    path = args.split_file or split_file_path(args)
    if os.path.exists(path):
        with open(path) as f:
            splits = json.load(f)
        if max(max(v) for v in splits.values() if v) >= len(df):
            raise ValueError(f"split file {path} refers to rows that are not in {args.data_path}; "
                             "the CSV changed - delete the split file to regenerate it")
        return splits
    splits = make_splits(df, args)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f)
    print(f"[split] saved new split to {path}")
    return splits
