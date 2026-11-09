"""Étape communities : Louvain par pas de temps, part d'illicites, homophilie."""

import json

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl

from elliptic_graph.communities import process
from elliptic_graph.graph.process import step_graphs
from elliptic_graph.graph.task import load_graph
from elliptic_graph.utils import Context


def run(ctx: Context) -> None:
    g = load_graph(ctx)
    res, seed = ctx.cfg["communities"]["resolution"], ctx.seed
    rows, modularity, node_size, distances = [], [], [], []
    for step, sg in step_graphs(g).items():
        comms, q = process.louvain(sg, res, seed)
        rows += [{"step": step, **r} for r in process.community_rows(sg, comms)]
        modularity.append({"step": step, "n_communities": len(comms), "modularity": q})
        node_size += [{"tx": n, "community_size": len(c)} for c in comms for n in c]
        distances.append(process.mean_distances(sg, ctx.cfg["communities"]["n_sources"], seed))
        ctx.log.info("communities : pas %d, %d communautés, Q=%.3f", step, len(comms), q)

    out = ctx.dir("metrics")
    pl.DataFrame(rows).with_columns(
        illicit_share=pl.col("illicit") / (pl.col("illicit") + pl.col("licit"))
    ).write_csv(out / "communities.csv")
    pl.DataFrame(modularity).write_csv(out / "modularity.csv")
    pl.DataFrame(node_size).write_parquet(ctx.dir("data") / "community_size.parquet")

    d = pl.concat(distances).group_by("target").agg(pl.col("distance").mean())
    homophily = process.edge_homophily(g) | {
        f"mean_distance_from_illicit_to_{t}": v for t, v in d.iter_rows()
    }
    (out / "homophily.json").write_text(json.dumps(homophily, indent=2))

    draw_illicit_subgraph(g, ctx)


def draw_illicit_subgraph(g: nx.DiGraph, ctx: Context) -> None:
    sgs = step_graphs(g)
    step = max(sgs, key=lambda s: sum(lab == "illicit" for _, lab in sgs[s].nodes(data="label")))
    sg = sgs[step]
    illicit = [n for n, lab in sg.nodes(data="label") if lab == "illicit"]
    keep = set(illicit) | {m for n in illicit for m in sg.predecessors(n)}
    keep |= {m for n in illicit for m in sg.successors(n)}
    sub = sg.subgraph(keep)
    colors = {"illicit": "tab:red", "licit": "tab:green", "unknown": "lightgray"}
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(
        sub,
        pos=nx.spring_layout(sub, seed=ctx.seed),
        node_color=[colors[sub.nodes[n]["label"]] for n in sub],
        node_size=15,
        arrows=False,
        with_labels=False,
        width=0.4,
        ax=ax,
    )
    ax.set_title(f"Pas {step} : transactions illicites (rouge) et leurs voisins")
    ax.axis("off")
    fig.savefig(ctx.dir("figures") / "illicit_subgraph.png", dpi=150)
