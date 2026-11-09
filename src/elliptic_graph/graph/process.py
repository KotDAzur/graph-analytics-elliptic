"""Fonctions pures : construction du DiGraph et résumé."""

import networkx as nx
import polars as pl


def build_graph(nodes: pl.DataFrame, edges: pl.DataFrame) -> nx.DiGraph:
    """DiGraph simple et non pondéré. Attributs des nœuds : step et label."""
    g = nx.DiGraph()
    g.add_nodes_from((tx, {"step": s, "label": lab}) for tx, s, lab in nodes.iter_rows())
    g.add_edges_from(edges.select("src", "dst").iter_rows())
    return g


def step_graphs(g: nx.DiGraph) -> dict[int, nx.DiGraph]:
    """Sous-graphe de chaque pas de temps (vues, pas des copies)."""
    groups: dict[int, list] = {}
    for tx, step in g.nodes(data="step"):
        groups.setdefault(step, []).append(tx)
    return {step: g.subgraph(groups[step]) for step in sorted(groups)}


def summary(g: nx.DiGraph, n_edges_in_file: int) -> dict:
    weak = [len(c) for c in nx.weakly_connected_components(g)]
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "directed": g.is_directed(),
        "multigraph": g.is_multigraph(),
        "duplicate_edges": n_edges_in_file - g.number_of_edges(),
        "self_loops": nx.number_of_selfloops(g),
        "is_dag": nx.is_directed_acyclic_graph(g),
        "weak_components": len(weak),
        "largest_weak_component": max(weak),
        "strong_components": nx.number_strongly_connected_components(g),
    }
