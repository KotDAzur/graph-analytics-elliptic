"""Étape centrality : degrés, PageRank, betweenness approchée, par pas de temps."""

import polars as pl

from elliptic_graph.centrality import process
from elliptic_graph.graph.process import step_graphs
from elliptic_graph.graph.task import load_graph
from elliptic_graph.utils import Context


def run(ctx: Context) -> None:
    g = load_graph(ctx)
    k, seed = ctx.cfg["centrality"]["betweenness_k"], ctx.seed
    parts = []
    for step, sg in step_graphs(g).items():
        parts.append(process.centralities(sg, k, seed).with_columns(step=pl.lit(step)))
        ctx.log.info("centrality : pas %d", step)
    df = pl.concat(parts)
    out = ctx.dir("metrics")
    df.write_csv(out / "centrality.csv")
    process.by_label(df).sort("label").write_csv(out / "centrality_by_label.csv")
    tests = process.permutation_tests(df, ctx.cfg["permutations"], ctx.seed)
    tests.write_csv(out / "centrality_tests.csv")
