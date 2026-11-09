"""Fonctions pures : Louvain, part d'illicites par communauté, homophilie."""

import random

import networkx as nx
import polars as pl


def louvain(g: nx.DiGraph, resolution: float, seed: int) -> tuple[list[set], float]:
    """Louvain sur la version non orientée. Renvoie les communautés et la modularité."""
    u = g.to_undirected()
    comms = nx.community.louvain_communities(u, resolution=resolution, seed=seed)
    return comms, nx.community.modularity(u, comms, resolution=resolution)


def community_rows(g: nx.DiGraph, comms: list[set]) -> list[dict]:
    """Une ligne par communauté : taille et effectifs par classe."""
    rows = []
    for i, c in enumerate(comms):
        labels = [g.nodes[n]["label"] for n in c]
        illicit, licit = labels.count("illicit"), labels.count("licit")
        rows.append({"community": i, "size": len(c), "illicit": illicit, "licit": licit})
    return rows


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
