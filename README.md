# Graph analytics of the Elliptic Bitcoin transaction network

Graph analytics 2026 project (Albert School). **Authors:** Mael ZAPATA, Thomas FOUSSARD.
The written report is submitted separately on Inside; this repository holds the code, the results of the run used in the report, and the instructions to reproduce them.

## Question
Does the structure of the transaction graph help to identify illicit Bitcoin transactions, and does this advantage hold over time?

We use the Elliptic data set (203,769 transactions, 234,355 directed flows, 49 time steps) with **NetworkX**.

## Data
Elliptic Data Set, Kaggle: <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>, introduced by Weber et al. (2019), *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*. License CC BY-NC-ND 4.0; see [DATA_CARD.md](DATA_CARD.md).

The data (690 MB) are **not stored in the repository**. The `data` step downloads the three CSV files into `data/raw/` automatically (no Kaggle account or API key needed) and skips the download if they are already there. You can also place `elliptic_txs_features.csv`, `elliptic_txs_classes.csv` and `elliptic_txs_edgelist.csv` there by hand.

## Installation and usage
Requires [uv](https://docs.astral.sh/uv/) (Python 3.13).

```bash
uv sync
uv run main.py run --exp smoke_test      # 2 time steps, a few minutes
uv run main.py run --exp full            # 49 time steps, about 20 minutes
uv run main.py run --exp full --steps graph structure   # only some steps
uv run pytest                            # tests on toy graphs
```

The full run takes about 20 minutes on a laptop, almost all of it for the exact diameter and radius of each time step. Results are written to `results/<exp>/`.

## Pipeline
`data` → `graph` → `structure` → `centrality` → `communities` → `detection`

| Step | What it does | Outputs (`results/full/`) |
|---|---|---|
| `data` | download, label recoding, node and edge tables | `metrics/data_summary.json`, `labels_by_step.csv` |
| `graph` | simple unweighted `DiGraph`, DAG check | `metrics/graph_summary.json` |
| `structure` | weak and strong components, exact diameter and radius per time step (undirected), degrees | `components.csv`, `distances.csv`, `degrees.csv`, `figures/degrees.png`, `figures/diameter_by_step.png` |
| `centrality` | in/out-degree, PageRank, betweenness (k = 500 pivots) per time step; permutation tests of the illicit vs licit gaps | `centrality.csv` (per node, git-ignored: it contains the labels, see DATA_CARD), `centrality_by_label.csv`, `centrality_tests.csv` |
| `communities` | Louvain per time step, illicit share and shape of each community, homophily with a permutation null model | `communities.csv`, `communities_summary.json`, `top_communities.csv`, `modularity.csv`, `homophily.json`, `figures/illicit_subgraph.png` |
| `detection` | random forest, temporal split (train steps 1 to 34, test 35 to 49), 4 feature sets, 5 seeds | `detection_overall.csv` (mean and sd), `detection_seeds.csv`, `detection.csv` (per step), `figures/f1_by_step.png` |

Each module has `process.py` (pure functions) and `task.py` (`run(ctx)`: read, compute, write). Parameters (seed 42, forest seeds, number of permutations, betweenness `k`, Louvain resolution, forest size, split) are in `configs/base.yaml` and `configs/experiments/`. Each run stores its `config.yaml` and `run_info.json` (seed, durations, git commit). Intermediate tables go to `results/<exp>/data/` (git-ignored).

## Key results
- The graph is a **simple, unweighted DAG**; it has **49 weakly connected components** (one per time step, no edge between steps) and 203,769 trivial strongly connected components.
- Diameter and radius are computed per time step on the undirected graph: diameter 41 to 786 (median 94), radius 21 to 393 (median 47).
- **Homophily:** 54% of the labelled neighbours of an illicit transaction are illicit (2.4% for a licit one; base rate 9.8%); assortativity 0.52, against 0.09 when labels are shuffled within time steps (1,000 permutations, p = 0.001).
- Illicit transactions are **less connected** than licit ones (mean in-degree 1.27 vs 1.91; the four centrality gaps are significant under within-step permutation tests, p = 0.001). Louvain: 57 communities per time step (modularity 0.90 on average); 5% of the communities hold 62% of the illicit nodes. The communities richest in illicit nodes are either long chains or hub-and-spoke clusters.
- **Detection (illicit-class F1, steps 35 to 49, mean ± sd over 5 seeds):** local 0.794 ± 0.003, local + graph 0.797 ± 0.002, local + aggregated 0.827 ± 0.002, local + aggregated + graph 0.828 ± 0.001. Graph features change F1 by less than the seed noise, and F1 collapses after time step 42 for every feature set.

## Repository layout
```
main.py                  CLI
configs/                 base.yaml, experiments/{full,smoke_test}.yaml
src/elliptic_graph/      utils.py and one folder per step (process.py, task.py)
tests/test_process.py    toy graphs with known diameter, radius, components...
results/full/            metrics/, figures/, config.yaml, run_info.json
data/raw/                CSV files (git-ignored)
notebooks/               empty placeholder
```

## Code reuse and use of AI tools
- **Reused code:** none. No existing code for this data set (Kaggle notebook, code of the paper's authors) was reused. The layout (process/task modules, YAML configs) follows an earlier project of one of the authors, `contract-review-nlp`. Libraries are used through their documented interfaces: NetworkX 3.7, Polars 2.1, scikit-learn 1.9, SciPy, NumPy, matplotlib, PyYAML, kagglehub.
- **AI tools:** AI was used as an aid for writing the code and the report. The authors defined the question and provided the skeleton of the project (repository structure and pipeline steps); Claude (Anthropic) helped to write the code within this skeleton (Claude Code) and to draft the report, and was also used in a chat to discuss the choice of data set and the analysis plan. All figures and numbers come from running this code.

## References
- Weber et al., *Anti-Money Laundering in Bitcoin...*, KDD'19 Workshop on Anomaly Detection in Finance, 2019.
- Elliptic Data Set, Kaggle, <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>.
- Hagberg, Schult, Swart, *Exploring network structure, dynamics, and function using NetworkX*, SciPy 2008. Blondel et al., *Fast unfolding of communities in large networks*, 2008 (Louvain). Brandes and Pich, *Centrality estimation in large networks*, 2007 (sampled betweenness).
