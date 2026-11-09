"""Fonctions pures : fusion des étiquettes, filtrage par pas de temps, statistiques."""

import polars as pl

# Dans le CSV : "1" = illicite, "2" = licite. On recode tout de suite pour éviter l'inversion.
LABELS = {"1": "illicit", "2": "licit", "unknown": "unknown"}


def build_nodes(times: pl.DataFrame, classes: pl.DataFrame, steps: list[int]) -> pl.DataFrame:
    """Colonnes tx, step, label, pour les pas de temps dans [steps[0], steps[1]]."""
    labels = classes.select(
        pl.col("txId").alias("tx"), pl.col("class").replace_strict(LABELS).alias("label")
    )
    return times.join(labels, on="tx").filter(pl.col("step").is_between(*steps))


def build_edges(edges: pl.DataFrame, nodes: pl.DataFrame) -> pl.DataFrame:
    """Colonnes src, dst, src_step, dst_step, pour les arêtes dont les deux bouts sont gardés."""
    steps = nodes.select("tx", "step")
    return (
        edges.rename({"txId1": "src", "txId2": "dst"})
        .join(steps.rename({"tx": "src", "step": "src_step"}), on="src")
        .join(steps.rename({"tx": "dst", "step": "dst_step"}), on="dst")
    )


def summary(nodes: pl.DataFrame, edges: pl.DataFrame) -> dict:
    return {
        "n_nodes": nodes.height,
        "n_edges": edges.height,
        "labels": dict(nodes["label"].value_counts().iter_rows()),
        "edges_across_steps": edges.filter(pl.col("src_step") != pl.col("dst_step")).height,
    }


def labels_by_step(nodes: pl.DataFrame) -> pl.DataFrame:
    """Une ligne par pas de temps : effectifs illicit / licit / unknown."""
    return nodes.pivot(on="label", index="step", values="tx", aggregate_function="len").sort("step")
