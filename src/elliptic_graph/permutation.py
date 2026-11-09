"""Outils pour les tests par permutation des étiquettes (mélange à l'intérieur de chaque pas)."""

from itertools import pairwise

import numpy as np


def step_bounds(steps: np.ndarray) -> np.ndarray:
    """Indices de début de chaque pas dans un tableau trié par pas, plus l'indice de fin."""
    return np.r_[0, np.flatnonzero(np.diff(steps)) + 1, len(steps)]


def shuffle_within_steps(y: np.ndarray, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mélange y à l'intérieur de chaque pas de temps : garde le nombre d'illicites par pas."""
    out = y.copy()
    for a, b in pairwise(bounds):
        out[a:b] = rng.permutation(y[a:b])
    return out


def p_value(observed: float, null: np.ndarray) -> float:
    """p-valeur bilatérale : part des tirages au moins aussi éloignés de la moyenne nulle."""
    far = np.abs(null - null.mean()) >= abs(observed - null.mean())
    return float((1 + far.sum()) / (len(null) + 1))
