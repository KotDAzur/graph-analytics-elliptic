"""Fonctions pures : centralités d'un pas de temps."""

import networkx as nx
import polars as pl


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
