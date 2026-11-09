"""Fonctions pures : connexité, diamètre, rayon, degrés."""

from collections import Counter

import networkx as nx
import polars as pl


def components_row(g: nx.DiGraph) -> dict:
    """Nombre de composantes faibles et fortes, taille de la plus grande composante faible."""
    return {
        "n_nodes": len(g),
        "n_edges": g.number_of_edges(),
        "weak": nx.number_weakly_connected_components(g),
        "strong": nx.number_strongly_connected_components(g),
        "largest_weak": max(len(c) for c in nx.weakly_connected_components(g)),
    }


def diameter_radius(g: nx.DiGraph) -> dict:
    """Diamètre et rayon du graphe non orienté sous-jacent, sur sa plus grande composante.

    Dans un DAG, la plupart des distances orientées sont infinies : on ignore le sens des arêtes.
    """
    u = g.to_undirected()
    biggest = u.subgraph(max(nx.connected_components(u), key=len))
    ecc = nx.eccentricity(biggest)
    return {"n_nodes": len(biggest), "diameter": max(ecc.values()), "radius": min(ecc.values())}


def degree_table(g: nx.DiGraph) -> pl.DataFrame:
    """Nombre de nœuds pour chaque degré entrant et sortant."""
    n_in = Counter(d for _, d in g.in_degree())
    n_out = Counter(d for _, d in g.out_degree())
    degrees = sorted(n_in | n_out)
    return pl.DataFrame(
        {
            "degree": degrees,
            "n_in": [n_in[d] for d in degrees],
            "n_out": [n_out[d] for d in degrees],
        }
    )
