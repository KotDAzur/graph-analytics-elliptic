# graph-analytics-elliptic

Analyse de graphe du jeu de données Elliptic (transactions Bitcoin, Weber et al. 2019).

## Question
_À compléter._

## Installation
```bash
uv sync
```
Les 3 CSV du [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) sont téléchargés automatiquement dans `data/raw/` par l'étape `data` (sans clé API) :
`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`.

## Commandes
```bash
uv run main.py run --exp smoke_test          # test rapide (2 pas de temps)
uv run main.py run --exp full                # 49 pas de temps
uv run main.py run --exp full --steps graph structure
uv run pytest
```

## Résultats clés
_À compléter._

## Usage de l'IA
_À compléter._
