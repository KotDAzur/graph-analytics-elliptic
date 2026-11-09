"""Fonctions pures : Louvain, part d'illicites par communauté, homophilie."""

import random

import networkx as nx
import numpy as np
import polars as pl

from elliptic_graph.permutation import p_value, shuffle_within_steps, step_bounds


def louvain(g: nx.DiGraph, resolution: float, seed: int) -> tuple[list[set], float]:
    """Louvain sur la version non orientée. Renvoie les communautés et la modularité."""
    u = g.to_undirected()
    comms = nx.community.louvain_communities(u, resolution=resolution, seed=seed)
    return comms, nx.community.modularity(u, comms, resolution=resolution)


def community_rows(g: nx.DiGraph, comms: list[set]) -> list[dict]:
    """Une ligne par communauté : taille, effectifs par classe, arêtes internes, degré maximal."""
    rows = []
    for i, c in enumerate(comms):
        sub = g.subgraph(c)
        labels = [g.nodes[n]["label"] for n in c]
        rows.append(
            {
                "community": i,
                "size": len(c),
                "illicit": labels.count("illicit"),
                "licit": labels.count("licit"),
                "n_edges": sub.number_of_edges(),
                "max_degree": max(d for _, d in sub.degree()),
            }
        )
    return rows


def with_shares(cm: pl.DataFrame) -> pl.DataFrame:
    """Ajoute illicit_share (null si aucun nœud étiqueté), star_share et edges_per_tree.

    star_share = degré maximal / (taille - 1) : 1 pour une étoile.
    edges_per_tree = n_edges / (taille - 1) : 1 pour un arbre."""
    labeled = pl.col("illicit") + pl.col("licit")
    return cm.with_columns(
        illicit_share=pl.when(labeled > 0).then(pl.col("illicit") / labeled),
        star_share=pl.col("max_degree") / (pl.col("size") - 1),
        edges_per_tree=pl.col("n_edges") / (pl.col("size") - 1),
    )


def concentration(cm: pl.DataFrame) -> dict:
    """Répartition des illicites dans les communautés, et forme des communautés riches en illicites.

    `cm` : une ligne par communauté (step, size, illicit, licit, n_edges, max_degree)."""
    cm = with_shares(cm)
    total = cm["illicit"].sum()
    top = cm.sort("illicit", descending=True).head(int(0.05 * cm.height))
    majority = cm.filter(pl.col("illicit_share") > 0.5)
    # Formes comparées sur les communautés avec au moins 5 nœuds étiquetés.
    big = cm.filter(pl.col("illicit") + pl.col("licit") >= 5)
    rich = big.filter(pl.col("illicit_share") > 0.5)
    other = big.filter(pl.col("illicit_share") <= 0.5)
    shape = lambda d: {
        "n": d.height,
        "median_size": d["size"].median(),
        "median_star_share": d["star_share"].median(),
        "median_edges_per_tree": d["edges_per_tree"].median(),
    }
    return {
        "n_communities": cm.height,
        "n_with_illicit": cm.filter(pl.col("illicit") > 0).height,
        "top5pct_n": top.height,
        "top5pct_illicit_share": top["illicit"].sum() / total,
        "n_majority_illicit": majority.height,
        "majority_illicit_holds": majority["illicit"].sum() / total,
        "n_pure_illicit_min5": big.filter(pl.col("illicit_share") == 1).height,
        "shape_illicit_rich": shape(rich),
        "shape_other": shape(other),
    }


def edge_homophily(g: nx.DiGraph) -> dict:
    """Arêtes illicite-illicite, illicite-licite, licite-licite (arêtes dont les deux bouts sont
    étiquetés), part d'illicites parmi les voisins étiquetés, et assortativité de la classe."""
    labeled = g.subgraph(n for n, lab in g.nodes(data="label") if lab != "unknown")
    pairs = [(labeled.nodes[a]["label"], labeled.nodes[b]["label"]) for a, b in labeled.edges]
    ii = sum(p == ("illicit", "illicit") for p in pairs)
    ll = sum(p == ("licit", "licit") for p in pairs)
    il = len(pairs) - ii - ll
    n_illicit = sum(lab == "illicit" for _, lab in labeled.nodes(data="label"))
    return {
        "edges_illicit_illicit": ii,
        "edges_illicit_licit": il,
        "edges_licit_licit": ll,
        "base_rate_illicit": n_illicit / len(labeled),
        "illicit_neighbors_of_illicit": 2 * ii / (2 * ii + il),
        "illicit_neighbors_of_licit": il / (il + 2 * ll),
        "assortativity": nx.attribute_assortativity_coefficient(labeled, "label"),
    }


def mean_distances(g: nx.DiGraph, n_sources: int, seed: int) -> pl.DataFrame:
    """Depuis n_sources transactions illicites : distances (non orienté) vers les autres
    illicites et vers les licites du même pas de temps."""
    u = g.to_undirected()
    illicit = [n for n, lab in g.nodes(data="label") if lab == "illicit"]
    sources = random.Random(seed).sample(illicit, min(n_sources, len(illicit)))
    rows = []
    for s in sources:
        for n, d in nx.single_source_shortest_path_length(u, s).items():
            if n != s and g.nodes[n]["label"] != "unknown":
                rows.append({"target": g.nodes[n]["label"], "distance": d})
    return pl.DataFrame(rows, schema={"target": pl.String, "distance": pl.Int64})


def assortativity_null(g: nx.DiGraph, n_perm: int, seed: int) -> dict:
    """Assortativité de la classe (arêtes entre nœuds étiquetés) face à un modèle nul : étiquettes
    mélangées au hasard à l'intérieur de chaque pas de temps (on garde les illicites par pas)."""
    nodes = sorted(
        (n for n, lab in g.nodes(data="label") if lab != "unknown"),
        key=lambda n: g.nodes[n]["step"],
    )
    pos = {n: i for i, n in enumerate(nodes)}
    y = np.array([g.nodes[n]["label"] == "illicit" for n in nodes], dtype=int)
    steps = np.array([g.nodes[n]["step"] for n in nodes])
    edges = np.array([(pos[a], pos[b]) for a, b in g.edges if a in pos and b in pos])

    def assortativity(y):
        e = np.bincount(2 * y[edges[:, 0]] + y[edges[:, 1]], minlength=4).reshape(2, 2) / len(edges)
        ab = e.sum(1) @ e.sum(0)
        return (np.trace(e) - ab) / (1 - ab)

    rng, bounds = np.random.default_rng(seed), step_bounds(steps)
    null = np.array([assortativity(shuffle_within_steps(y, bounds, rng)) for _ in range(n_perm)])
    observed = assortativity(y)
    return {
        "observed": observed,
        "null_mean": null.mean(),
        "null_std": null.std(),
        "p_value": p_value(observed, null),
        "n_permutations": n_perm,
    }
