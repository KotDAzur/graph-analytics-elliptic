"""Étape graph : construit le DiGraph, vérifie le DAG, écrit graph_summary.json."""

import json

import networkx as nx

from elliptic_graph.data.task import read_data
from elliptic_graph.graph import process
from elliptic_graph.utils import Context


def load_graph(ctx: Context) -> nx.DiGraph:
    """Même graphe pour toutes les étapes."""
    return process.build_graph(*read_data(ctx))


def run(ctx: Context) -> None:
    nodes, edges = read_data(ctx)
    g = process.build_graph(nodes, edges)
    stats = process.summary(g, edges.height)
    (ctx.dir("metrics") / "graph_summary.json").write_text(json.dumps(stats, indent=2))
    ctx.log.info("graph : %s", stats)
