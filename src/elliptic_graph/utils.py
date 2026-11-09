"""Contexte d'exécution, logs, dossiers de run."""

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Context:
    cfg: dict
    run_dir: Path
    log: logging.Logger

    def dir(self, name: str) -> Path:
        """Sous-dossier du run (metrics, figures...), créé si besoin."""
        d = self.run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def seed(self) -> int:
        return self.cfg["seed"]

    @property
    def raw(self) -> Path:
        return ROOT / self.cfg["paths"]["raw"]


def _merge(a: dict, b: dict) -> dict:
    return a | {k: _merge(a[k], v) if isinstance(v, dict) and k in a else v for k, v in b.items()}


def make_context(exp: str) -> Context:
    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    cfg = _merge(base, yaml.safe_load((ROOT / f"configs/experiments/{exp}.yaml").read_text()))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run_dir = ROOT / cfg["paths"]["results"] / exp
    run_dir.mkdir(parents=True, exist_ok=True)
    return Context(cfg, run_dir, logging.getLogger(exp))


def write_run_info(ctx: Context, durations: dict[str, float]) -> None:
    """Écrit config.yaml et run_info.json (seed, durées, hash git)."""
    (ctx.run_dir / "config.yaml").write_text(yaml.safe_dump(ctx.cfg, sort_keys=False))
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "HEAD"],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "seed": ctx.cfg["seed"],
        "git_commit": commit or None,
        "durations_s": durations,
    }
    (ctx.run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
