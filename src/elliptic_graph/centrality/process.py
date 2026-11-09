"""Fonctions pures : centralités d'un pas de temps."""

import networkx as nx
import numpy as np
import polars as pl

from elliptic_graph.permutation import p_value, shuffle_within_steps, step_bounds

METRICS = ["in_degree", "out_degree", "pagerank", "betweenness"]


def centralities(g: nx.DiGraph, k: int, seed: int) -> pl.DataFrame:
    """Degrés, PageRank et betweenness approchée (k pivots) de chaque nœud."""
    pagerank = nx.pagerank(g)
    betweenness = nx.betweenness_centrality(g, k=min(k, len(g)), seed=seed)
    return pl.DataFrame(
        {
            "tx": list(g),
            "label": [g.nodes[n]["label"] for n in g],
            "in_degree": [g.in_degree(n) for n in g],
            "out_degree": [g.out_degree(n) for n in g],
            "pagerank": [pagerank[n] for n in g],
            "betweenness": [betweenness[n] for n in g],
        }
    )


def by_label(df: pl.DataFrame) -> pl.DataFrame:
    """Moyenne et médiane de chaque centralité selon la classe."""
    cols = ["in_degree", "out_degree", "pagerank", "betweenness"]
    return df.group_by("label").agg(
        [pl.col(c).mean().alias(f"{c}_mean") for c in cols]
        + [pl.col(c).median().alias(f"{c}_median") for c in cols]
    )


def permutation_tests(df: pl.DataFrame, n_perm: int, seed: int) -> pl.DataFrame:
    """Écart de moyenne illicite moins licite pour chaque centralité, face à un modèle nul :
    étiquettes mélangées à l'intérieur de chaque pas (contrôle la taille des pas de temps).

    `df` : colonnes step, label et les centralités."""
    df = df.filter(pl.col("label") != "unknown").sort("step")
    values = df.select(METRICS).to_numpy()
    y = (df["label"] == "illicit").to_numpy().astype(float)
    n_illicit = y.sum()

    def gap(y):
        return values.T @ y / n_illicit - values.T @ (1 - y) / (len(y) - n_illicit)

    rng, bounds = np.random.default_rng(seed), step_bounds(df["step"].to_numpy())
    null = np.array([gap(shuffle_within_steps(y, bounds, rng)) for _ in range(n_perm)])
    observed = gap(y)
    return pl.DataFrame(
        {
            "metric": METRICS,
            "illicit_minus_licit": observed,
            "null_mean": null.mean(axis=0),
            "null_std": null.std(axis=0),
            "p_value": [p_value(observed[i], null[:, i]) for i in range(len(METRICS))],
            "n_permutations": n_perm,
        }
    )
