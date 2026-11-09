"""Fonctions pures : jeux de caractéristiques, entraînement, métriques."""

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

GRAPH = ["in_degree", "out_degree", "pagerank", "betweenness", "community_size"]

# features.csv : colonnes 2 à 94 = 93 locales, 95 à 166 = 72 agrégées sur les voisins à 1 saut.
FEATURE_SETS = {
    "local": ["local"],
    "local_graph": ["local", "graph"],
    "local_agg": ["local", "agg"],
    "local_agg_graph": ["local", "agg", "graph"],
}


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    return {"local": columns[2:95], "agg": columns[95:167], "graph": GRAPH}


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Précision, rappel et F1 sur la classe illicite (1)."""
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "n_illicit": int(y.sum())}


def evaluate(df: pl.DataFrame, cols: list[str], train_max: int, n_trees: int, seed: int) -> list:
    """Entraîne sur step <= train_max, teste sur step > train_max. Une ligne par pas de test
    (ceux qui contiennent des illicites), plus une ligne 'all' pour tout le test."""
    train, test = df.filter(pl.col("step") <= train_max), df.filter(pl.col("step") > train_max)
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    rf.fit(train.select(cols).to_numpy(), train["y"].to_numpy())
    test = test.with_columns(pred=pl.Series(rf.predict(test.select(cols).to_numpy())))
    rows = [{"step": "all", **metrics(test["y"].to_numpy(), test["pred"].to_numpy())}]
    for step, t in test.group_by("step", maintain_order=True):
        if t["y"].sum() > 0:
            rows.append({"step": step[0], **metrics(t["y"].to_numpy(), t["pred"].to_numpy())})
    return rows
