"""CLI : uv run main.py run --exp full [--steps data graph ...]"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from elliptic_graph import centrality, communities, data, detection, graph, structure
from elliptic_graph.utils import make_context, write_run_info

STEPS = {
    "data": data.run,
    "graph": graph.run,
    "structure": structure.run,
    "centrality": centrality.run,
    "communities": communities.run,
    "detection": detection.run,
}


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="exécuter le pipeline")
    r.add_argument("--exp", default="full", help="nom dans configs/experiments/")
    r.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    args = p.parse_args()

    ctx = make_context(args.exp)
    durations = {}
    for name in args.steps:
        ctx.log.info("étape %s", name)
        t = time.perf_counter()
        STEPS[name](ctx)
        durations[name] = round(time.perf_counter() - t, 2)
    write_run_info(ctx, durations)


if __name__ == "__main__":
    main()
