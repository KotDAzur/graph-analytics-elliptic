"""Petits graphes jouets dont on connaît les réponses."""

import networkx as nx
import polars as pl

from elliptic_graph.centrality.process import centralities
from elliptic_graph.communities.process import community_rows, edge_homophily
from elliptic_graph.data.process import build_edges, build_nodes
from elliptic_graph.graph.process import build_graph, step_graphs
from elliptic_graph.structure.process import components_row, degree_table, diameter_radius


def labeled(g: nx.DiGraph, labels: list[str]) -> nx.DiGraph:
    nx.set_node_attributes(g, dict(zip(g, labels)), "label")
    return g


def test_labels_recoded_and_steps_filtered():
    times = pl.DataFrame({"tx": [1, 2, 3], "step": [1, 1, 3]})
    classes = pl.DataFrame({"txId": [1, 2, 3], "class": ["1", "2", "unknown"]})
    nodes = build_nodes(times, classes, [1, 2])
    assert nodes["label"].to_list() == ["illicit", "licit"]
    edges = pl.DataFrame({"txId1": [1, 2], "txId2": [2, 3]})
    assert build_edges(edges, nodes).height == 1  # l'arête vers le nœud 3 est écartée


def test_build_graph_and_step_graphs():
    nodes = pl.DataFrame({"tx": [1, 2, 3], "step": [1, 1, 2], "label": ["licit"] * 3})
    edges = pl.DataFrame({"src": [1], "dst": [2]})
    g = build_graph(nodes, edges)
    assert g.number_of_edges() == 1
    assert {s: len(sg) for s, sg in step_graphs(g).items()} == {1: 2, 2: 1}


def test_path_graph_diameter_radius():
    g = nx.path_graph(5, create_using=nx.DiGraph)  # 0->1->2->3->4
    assert diameter_radius(g) == {"n_nodes": 5, "diameter": 4, "radius": 2}


def test_components():
    g = nx.DiGraph([(0, 1), (1, 2), (3, 4)])
    row = components_row(g)
    assert (row["weak"], row["strong"], row["largest_weak"]) == (2, 5, 3)  # DAG : forts isolés


def test_cycle_is_one_strong_component():
    assert components_row(nx.DiGraph([(0, 1), (1, 2), (2, 0)]))["strong"] == 1


def test_degree_table():
    t = degree_table(nx.DiGraph([(0, 1), (0, 2)]))  # 0 : sortant 2 ; 1 et 2 : entrant 1
    assert t.filter(pl.col("degree") == 2)["n_out"].to_list() == [1]
    assert t.filter(pl.col("degree") == 1)["n_in"].to_list() == [2]


def test_star_centralities():
    g = labeled(nx.DiGraph([(0, 1), (0, 2), (0, 3)]), ["illicit", "licit", "licit", "licit"])
    df = centralities(g, k=4, seed=0)
    assert df.filter(pl.col("tx") == 0)["out_degree"].item() == 3
    assert df["pagerank"].sum() > 0.99


def test_community_rows_and_homophily():
    g = labeled(nx.DiGraph([(0, 1), (2, 3)]), ["illicit", "illicit", "licit", "licit"])
    rows = community_rows(g, [{0, 1}, {2, 3}])
    assert [(r["illicit"], r["licit"]) for r in rows] == [(2, 0), (0, 2)]
    h = edge_homophily(g)
    assert (h["edges_illicit_illicit"], h["edges_licit_licit"], h["edges_illicit_licit"]) == (
        1,
        1,
        0,
    )
    assert h["illicit_neighbors_of_illicit"] == 1.0
