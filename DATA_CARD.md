# Data card : Elliptic Data Set

- **Source :** Kaggle, Elliptic Data Set (ellipticco/elliptic-data-set).
- **Article :** Weber et al. (2019), *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*, KDD Workshop.
- **Licence :** à vérifier sur la page Kaggle.
- **Contenu :** 203 769 transactions (nœuds), 234 355 flux de bitcoins (arêtes orientées), 49 pas de temps d'environ deux semaines.
- **Étiquettes :** 4 545 illicites (environ 2 %), 42 019 licites, 157 205 inconnues. Dans le CSV : "1" = illicite, "2" = licite, "unknown".
- **Fichiers :** `elliptic_txs_features.csv` (sans en-tête : txId, time_step, 165 caractéristiques), `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`.
- **Limites :** étiquettes partielles (77 % inconnues), caractéristiques anonymisées, pas de montants ni de poids, dérive temporelle, aucune arête entre pas de temps.
