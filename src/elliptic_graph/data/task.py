"""Étape data : télécharge les CSV Kaggle, fusionne les étiquettes, écrit nodes et edges."""

import json
import shutil
from pathlib import Path

import kagglehub
import polars as pl

from elliptic_graph.data import process
from elliptic_graph.utils import Context

FILES = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]


def download(ctx: Context) -> None:
    """Copie les CSV dans data/raw/ s'ils manquent (kagglehub, sans clé API)."""
    ctx.raw.mkdir(parents=True, exist_ok=True)
    if all((ctx.raw / f).exists() for f in FILES):
        return
    src = Path(kagglehub.dataset_download("ellipticco/elliptic-data-set"))
    for f in FILES:
        shutil.copy(next(src.rglob(f)), ctx.raw / f)


def read_data(ctx: Context) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Relit nodes et edges écrits par l'étape data."""
    d = ctx.dir("data")
    return pl.read_parquet(d / "nodes.parquet"), pl.read_parquet(d / "edges.parquet")


def run(ctx: Context) -> None:
    download(ctx)
    # features.csv n'a pas d'en-tête : colonne 0 = txId, colonne 1 = pas de temps.
    times = pl.read_csv(ctx.raw / FILES[0], has_header=False, columns=[0, 1])
    times = times.rename({"column_0": "tx", "column_1": "step"})
    classes = pl.read_csv(ctx.raw / FILES[1], schema_overrides={"class": pl.String})
    nodes = process.build_nodes(times, classes, ctx.cfg["time_steps"])
    edges = process.build_edges(pl.read_csv(ctx.raw / FILES[2]), nodes)

    nodes.write_parquet(ctx.dir("data") / "nodes.parquet")
    edges.write_parquet(ctx.dir("data") / "edges.parquet")
    out = ctx.dir("metrics")
    (out / "data_summary.json").write_text(json.dumps(process.summary(nodes, edges), indent=2))
    process.labels_by_step(nodes).write_csv(out / "labels_by_step.csv")
    ctx.log.info("data : %d nœuds, %d arêtes", nodes.height, edges.height)
