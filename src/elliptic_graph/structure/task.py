"""Étape structure : composantes, diamètre et rayon par pas de temps, degrés."""

import matplotlib.pyplot as plt
import polars as pl

from elliptic_graph.graph.process import step_graphs
from elliptic_graph.graph.task import load_graph
from elliptic_graph.structure import process
from elliptic_graph.utils import Context


def run(ctx: Context) -> None:
    g = load_graph(ctx)
    components, distances = [], []
    for step, sg in step_graphs(g).items():
        components.append({"step": step, **process.components_row(sg)})
        distances.append({"step": step, **process.diameter_radius(sg)})
        ctx.log.info("structure : pas %d", step)
    components, distances = pl.DataFrame(components), pl.DataFrame(distances)
    degrees = process.degree_table(g)

    out = ctx.dir("metrics")
    components.write_csv(out / "components.csv")
    distances.write_csv(out / "distances.csv")
    degrees.write_csv(out / "degrees.csv")

    fig, ax = plt.subplots()
    d = degrees.filter(pl.col("degree") > 0)
    ax.loglog(d["degree"], d["n_in"], "o", label="degré entrant")
    ax.loglog(d["degree"], d["n_out"], "s", label="degré sortant")
    ax.set(xlabel="degré", ylabel="nombre de nœuds")
    ax.legend()
    fig.savefig(ctx.dir("figures") / "degrees.png", dpi=150)

    fig, ax = plt.subplots()
    ax.plot(distances["step"], distances["diameter"], label="diamètre")
    ax.plot(distances["step"], distances["radius"], label="rayon")
    ax.set(xlabel="pas de temps", ylabel="distance (graphe non orienté)")
    ax.legend()
    fig.savefig(ctx.dir("figures") / "diameter_by_step.png", dpi=150)
