"""Intervalle de Wilson (proportions) et bootstrap par grappes.

Le bootstrap rééchantillonne les COPIES entières, pas les items : les 83 items
d'une copie ne sont pas indépendants, un bootstrap par item sous-estimerait
l'incertitude.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConfidenceInterval:
    """Un intervalle de confiance avec sa valeur ponctuelle."""

    estimate: float
    lower: float
    upper: float
    level: float = 0.95


def wilson_interval(successes: int, n: int, level: float = 0.95) -> ConfidenceInterval:
    """Intervalle de Wilson pour une proportion.

    Args:
        successes: nombre de succès observés.
        n: nombre total d'observations.
        level: niveau de confiance (0.90, 0.95 ou 0.99 ; sinon 0.95 par défaut).

    Returns:
        L'intervalle de confiance avec sa proportion ponctuelle, borné à [0, 1].

    Raises:
        ValueError: si n vaut 0.
    """
    if n == 0:
        raise ValueError("n doit être > 0.")
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(level, 1.96)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ConfidenceInterval(
        estimate=p, lower=max(0.0, centre - half), upper=min(1.0, centre + half), level=level
    )


def cluster_bootstrap(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    cluster_col: str = "copy_id",
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """Bootstrap par grappes : rééchantillonne les copies, pas les items.

    Args:
        df: données à l'item, avec une colonne identifiant la grappe.
        metric_fn: fonction calculant la métrique scalaire sur un DataFrame.
        cluster_col: colonne servant de grappe (copie) pour le rééchantillonnage.
        n_boot: nombre de tirages bootstrap.
        level: niveau de confiance de l'intervalle.
        seed: graine du générateur aléatoire (reproductibilité).

    Returns:
        L'intervalle de confiance : estimation sur df complet et bornes issues
        des quantiles de la distribution bootstrap.
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    groups = {c: g for c, g in df.groupby(cluster_col)}

    estimate = float(metric_fn(df))
    samples = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        boot_df = pd.concat([groups[c] for c in drawn], ignore_index=True)
        samples[b] = metric_fn(boot_df)

    alpha = (1 - level) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])
    return ConfidenceInterval(
        estimate=estimate, lower=float(lower), upper=float(upper), level=level
    )
