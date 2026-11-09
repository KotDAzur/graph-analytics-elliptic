"""Étape detection : random forest avec et sans caractéristiques de graphe."""

import matplotlib.pyplot as plt
import polars as pl

from elliptic_graph.data.task import FILES, read_data
from elliptic_graph.detection import process
from elliptic_graph.utils import Context


def run(ctx: Context) -> None:
    cfg = ctx.cfg["detection"]
    nodes, _ = read_data(ctx)
    metrics = ctx.dir("metrics")
    # Seules les transactions étiquetées servent à entraîner et évaluer.
    labeled = nodes.filter(pl.col("label") != "unknown").select(
        "tx", y=(pl.col("label") == "illicit").cast(pl.Int8)
    )
    features = pl.read_csv(ctx.raw / FILES[0], has_header=False)
    groups = process.feature_groups(features.columns)
    features = features.rename({"column_0": "tx", "column_1": "step"})
    df = (
        labeled.join(features, on="tx")
        .join(pl.read_csv(metrics / "centrality.csv").select("tx", *process.GRAPH[:4]), on="tx")
        .join(pl.read_parquet(ctx.dir("data") / "community_size.parquet"), on="tx")
    )

    rows = []
    for name, parts in process.FEATURE_SETS.items():
        cols = [c for p in parts for c in groups[p]]
        res = process.evaluate(df, cols, cfg["train_max_step"], cfg["n_estimators"], ctx.seed)
        rows += [{"feature_set": name, **r} for r in res]
        ctx.log.info("detection : %s, F1 test = %.3f", name, res[0]["f1"])
    out = pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col("step").cast(pl.String))
    out.filter(pl.col("step") == "all").write_csv(metrics / "detection_overall.csv")
    per_step = out.filter(pl.col("step") != "all").with_columns(pl.col("step").cast(pl.Int64))
    per_step.write_csv(metrics / "detection.csv")

    fig, ax = plt.subplots()
    for name, d in per_step.group_by("feature_set", maintain_order=True):
        d = d.sort("step")
        ax.plot(d["step"], d["f1"], marker="o", label=name[0])
    ax.set(xlabel="time step (test)", ylabel="F1 (illicit class)")
    ax.legend()
    fig.savefig(ctx.dir("figures") / "f1_by_step.png", dpi=150)
