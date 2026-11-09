"""Petits graphes jouets dont on connaît les réponses."""

import networkx as nx
import numpy as np
import polars as pl

from elliptic_graph.centrality.process import centralities, permutation_tests
from elliptic_graph.communities.process import (
    assortativity_null,
    community_rows,
    concentration,
    edge_homophily,
)
from elliptic_graph.data.process import build_edges, build_nodes
from elliptic_graph.graph.process import build_graph, step_graphs
from elliptic_graph.permutation import p_value, shuffle_within_steps, step_bounds
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
    assert [(r["n_edges"], r["max_degree"]) for r in rows] == [(1, 1), (1, 1)]
    h = edge_homophily(g)
    assert (h["edges_illicit_illicit"], h["edges_licit_licit"], h["edges_illicit_licit"]) == (
        1,
        1,
        0,
    )
    assert h["illicit_neighbors_of_illicit"] == 1.0


def test_shuffle_within_steps_keeps_illicit_per_step():
    steps = np.array([1, 1, 1, 2, 2])
    y = np.array([1, 0, 0, 1, 1])
    out = shuffle_within_steps(y, step_bounds(steps), np.random.default_rng(0))
    assert out[:3].sum() == 1 and out[3:].sum() == 2


def test_p_value_extreme_and_central():
    null = np.linspace(-1, 1, 99)
    assert p_value(5.0, null) == 1 / 100  # plus extrême que tout le modèle nul
    assert p_value(0.0, null) > 0.95


def test_assortativity_null_matches_networkx_and_is_significant():
    g = nx.DiGraph()
    for step in (1, 2):  # dans chaque pas : un amas illicite et un amas licite, sans lien entre eux
        base = step * 100
        illicit = [(base + i, base + i + 1) for i in range(0, 10, 2)]
        licit = [(base + 50 + i, base + 50 + i + 1) for i in range(0, 10, 2)]
        g.add_edges_from(illicit + licit)
        for a, b in illicit:
            g.nodes[a].update(label="illicit", step=step)
            g.nodes[b].update(label="illicit", step=step)
        for a, b in licit:
            g.nodes[a].update(label="licit", step=step)
            g.nodes[b].update(label="licit", step=step)
    res = assortativity_null(g, n_perm=200, seed=0)
    assert abs(res["observed"] - edge_homophily(g)["assortativity"]) < 1e-9
    assert res["observed"] == 1.0 and res["p_value"] < 0.05


def test_permutation_tests_detect_a_gap():
    df = pl.DataFrame(
        {
            "step": [1] * 20,
            "label": ["illicit"] * 10 + ["licit"] * 10,
            "in_degree": [5.0] * 10 + [1.0] * 10,
            "out_degree": [1.0] * 20,
            "pagerank": [1.0] * 20,
            "betweenness": [1.0] * 20,
        }
    )
    t = permutation_tests(df, n_perm=200, seed=0).filter(pl.col("metric") == "in_degree")
    assert t["illicit_minus_licit"].item() == 4.0 and t["p_value"].item() < 0.05


def test_concentration():
    cm = pl.DataFrame(  # la 4e communauté n'a aucun nœud étiqueté : elle ne doit rien compter
        {
            "size": [6, 6, 6, 6],
            "illicit": [5, 0, 1, 0],
            "licit": [0, 5, 4, 0],
            "n_edges": [5, 5, 5, 5],
            "max_degree": [5, 2, 2, 2],
        }
    )
    c = concentration(cm)
    assert c["n_majority_illicit"] == 1 and c["n_pure_illicit_min5"] == 1
    assert c["shape_illicit_rich"]["median_star_share"] == 1.0  # étoile
