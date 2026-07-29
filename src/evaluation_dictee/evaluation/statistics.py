"""Intervalles de confiance tenant compte du regroupement des items par copie.

Fil directeur : les 83 items d'une copie ne sont pas indépendants (un élève faible
se trompe partout), donc N items portent moins d'information que N observations
indépendantes. Ignorer ce fait produit des intervalles trop étroits.

Deux façons d'en tenir compte, selon la statistique :

- **une proportion** (accord, rappel, précision…) : `wilson_interval` avec le
  `design_effect` de l'indicatrice concernée. Instantané. Attention, le design
  effect appartient à une indicatrice donnée, pas au jeu de données : celui de
  l'erreur et celui de l'accord diffèrent nettement sur le même corpus ;
- **le kappa**, qui n'est la moyenne d'aucune indicatrice : `kappa_interval_clustered`,
  un bootstrap par grappes exact et rapide. Le corriger par le design effect d'une
  proportion voisine le sur-corrige de moitié.

Cas où AUCUNE correction n'est due : une statistique calculée à raison d'une seule
observation par copie (la prévalence d'un item, vue une fois par élève) — les
observations sont alors indépendantes. `wilson_interval` sans `deff` convient.

`kappa_interval` (delta method) et `cluster_bootstrap` (générique, plus lent)
restent disponibles pour les cas hors grappes ou les vérifications ponctuelles.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

_Z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


@dataclass
class ConfidenceInterval:
    """Un intervalle de confiance avec sa valeur ponctuelle."""

    estimate: float
    lower: float
    upper: float
    level: float = 0.95


def wilson_interval(
    successes: int, n: int, level: float = 0.95, deff: float = 1.0
) -> ConfidenceInterval:
    """Intervalle de Wilson pour une proportion, éventuellement corrigé du clustering.

    Args:
        successes: nombre de succès observés.
        n: nombre total d'observations.
        level: niveau de confiance (0.90, 0.95 ou 0.99 ; sinon 0.95 par défaut).
        deff: design effect (voir `design_effect`). L'intervalle est alors calculé
            sur l'effectif effectif `n / deff`, ce qui l'élargit d'un facteur
            √deff. Laisser à 1.0 quand les observations sont indépendantes.

    Returns:
        L'intervalle de confiance avec sa proportion ponctuelle, borné à [0, 1].
        La proportion ponctuelle reste celle des données brutes : seul l'intervalle
        est affecté par `deff`.

    Raises:
        ValueError: si n vaut 0 ou si deff est inférieur à 1.
    """
    if n == 0:
        raise ValueError("n doit être > 0.")
    if deff < 1.0:
        raise ValueError("deff doit être >= 1 (1.0 = observations indépendantes).")
    z = _Z.get(level, 1.96)
    p = successes / n
    n_eff = n / deff  # effectif effectif : ce que les données pèsent réellement
    denom = 1 + z**2 / n_eff
    centre = (p + z**2 / (2 * n_eff)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n_eff + z**2 / (4 * n_eff**2))
    return ConfidenceInterval(
        estimate=p, lower=max(0.0, centre - half), upper=min(1.0, centre + half), level=level
    )


def kappa_interval(
    y_true: Sequence[str] | pd.Series | np.ndarray,
    y_pred: Sequence[str] | pd.Series | np.ndarray,
    level: float = 0.95,
) -> ConfidenceInterval:
    """Kappa de Cohen et son IC asymptotique (delta method de Fleiss).

    Alternative instantanée au bootstrap pour les grands effectifs : la variance
    est calculée analytiquement sur la matrice des proportions conjointes.
    L'approximation normale suppose un effectif suffisant (quelques centaines
    d'observations au moins) — en dessous, préférer `cluster_bootstrap`.

    Args:
        y_true: codes de référence (annotateur expert).
        y_pred: codes prédits (modèle).
        level: niveau de confiance de l'intervalle.

    Returns:
        Le kappa et ses bornes. L'intervalle est dégénéré (bornes = estimation)
        quand l'accord attendu par hasard vaut 1, cas où la variance n'est pas
        définie — avec une seule catégorie observée, le kappa lui-même est NaN.

    Raises:
        ValueError: si les deux séries sont vides ou de longueurs différentes.
    """
    from sklearn.metrics import cohen_kappa_score

    true_arr = np.asarray(y_true, dtype=object)
    pred_arr = np.asarray(y_pred, dtype=object)
    if len(true_arr) != len(pred_arr):
        raise ValueError("y_true et y_pred doivent avoir la même longueur.")
    n = len(true_arr)
    if n == 0:
        raise ValueError("Séries vides : le kappa n'est pas défini.")

    kappa = float(cohen_kappa_score(true_arr, pred_arr))

    # Matrice des proportions conjointes p_ij (expert en lignes, modèle en colonnes).
    cats = sorted(set(true_arr) | set(pred_arr))
    idx = {c: i for i, c in enumerate(cats)}
    joint = np.zeros((len(cats), len(cats)))
    for t, p in zip(true_arr, pred_arr, strict=True):
        joint[idx[t], idx[p]] += 1
    joint /= n

    marge_pred = joint.sum(axis=0)  # marges du modèle
    marge_true = joint.sum(axis=1)  # marges de l'expert
    p_e = float((marge_pred * marge_true).sum())
    if p_e >= 1.0:
        return ConfidenceInterval(estimate=kappa, lower=kappa, upper=kappa, level=level)

    somme_marges = marge_pred + marge_true  # (p_.i + p_i.) indexé par catégorie
    terme_diag = np.diag(joint) * (1 - somme_marges * (1 - kappa))
    carres = joint * (marge_pred[np.newaxis, :] + marge_true[:, np.newaxis]) ** 2
    terme_hors_diag = (1 - kappa) ** 2 * (carres.sum() - np.trace(joint * somme_marges**2))
    variance = (terme_diag.sum() + terme_hors_diag - (kappa - p_e * (1 - kappa)) ** 2) / (
        n * (1 - p_e) ** 2
    )

    z = _Z.get(level, 1.96)
    demi = z * math.sqrt(max(0.0, variance))
    return ConfidenceInterval(estimate=kappa, lower=kappa - demi, upper=kappa + demi, level=level)


def _kappa_depuis_confusion(matrice: np.ndarray) -> float:
    """Kappa de Cohen calculé directement sur une matrice de confusion (comptes)."""
    total = matrice.sum()
    if total == 0:
        return float("nan")
    p_o = np.trace(matrice) / total
    p_e = float((matrice.sum(axis=0) * matrice.sum(axis=1)).sum()) / total**2
    if p_e >= 1.0:
        return float("nan")
    return float((p_o - p_e) / (1 - p_e))


def kappa_interval_clustered(
    y_true: Sequence[str] | pd.Series | np.ndarray,
    y_pred: Sequence[str] | pd.Series | np.ndarray,
    clusters: Sequence[str] | pd.Series | np.ndarray,
    level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 42,
) -> ConfidenceInterval:
    """Kappa de Cohen et son IC par bootstrap par grappes (copies rééchantillonnées).

    Pourquoi cette fonction plutôt que `kappa_interval` corrigé d'un design effect :
    le design effect se dérive d'une **indicatrice** précise (erreur, accord…) et
    diffère de l'une à l'autre. Le kappa n'est pas la moyenne d'une indicatrice, et
    gonfler sa variance par le design effect d'une proportion voisine le
    sur-corrige nettement. Le bootstrap par grappes, lui, ne suppose rien.

    L'implémentation reste rapide malgré les milliers de tirages : la matrice de
    confusion de chaque copie est calculée une fois, et un tirage se réduit à
    sommer les matrices des copies tirées. C'est exactement équivalent à
    rééchantillonner les lignes par grappes, sans reconstruire de DataFrame.

    Args:
        y_true: codes de référence (annotateur expert).
        y_pred: codes prédits (modèle).
        clusters: identifiant de grappe de chaque observation (la copie).
        level: niveau de confiance de l'intervalle.
        n_boot: nombre de tirages bootstrap.
        seed: graine du générateur aléatoire (reproductibilité).

    Returns:
        Le kappa observé et les bornes issues des quantiles de la distribution
        bootstrap.

    Raises:
        ValueError: si les séries sont vides ou de longueurs différentes.
    """
    true_arr = np.asarray(y_true, dtype=object)
    pred_arr = np.asarray(y_pred, dtype=object)
    grappes = np.asarray(clusters, dtype=object)
    if not (len(true_arr) == len(pred_arr) == len(grappes)):
        raise ValueError("y_true, y_pred et clusters doivent avoir la même longueur.")
    if len(true_arr) == 0:
        raise ValueError("Séries vides : le kappa n'est pas défini.")

    cats = sorted(set(true_arr) | set(pred_arr))
    idx = {c: i for i, c in enumerate(cats)}
    codes_true = np.array([idx[v] for v in true_arr])
    codes_pred = np.array([idx[v] for v in pred_arr])

    # Une matrice de confusion (k × k) par grappe, empilée : (n_grappes, k, k).
    grappes_uniques, position = np.unique(grappes, return_inverse=True)
    k = len(cats)
    par_grappe = np.zeros((len(grappes_uniques), k, k))
    np.add.at(par_grappe, (position, codes_true, codes_pred), 1)

    observe = _kappa_depuis_confusion(par_grappe.sum(axis=0))

    rng = np.random.default_rng(seed)
    tirages = np.empty(n_boot)
    for b in range(n_boot):
        choix = rng.integers(0, len(grappes_uniques), len(grappes_uniques))
        tirages[b] = _kappa_depuis_confusion(par_grappe[choix].sum(axis=0))

    alpha = (1 - level) / 2
    bas, haut = np.nanquantile(tirages, [alpha, 1 - alpha])
    return ConfidenceInterval(estimate=observe, lower=float(bas), upper=float(haut), level=level)


def design_effect(indicator: pd.Series | np.ndarray, clusters: pd.Series | np.ndarray) -> float:
    """Design effect de Kish d'une indicatrice binaire, dû au regroupement en grappes.

    Les items d'une même copie ne sont pas indépendants (un élève faible se trompe
    partout) : la moyenne d'une indicatrice sur N items porte donc moins
    d'information que N observations indépendantes. Le design effect chiffre cette
    perte — `deff = 1 + (m - 1) x ICC`, où m est la taille moyenne de grappe et
    l'ICC la part de variance qui vient des différences *entre* copies. Diviser N
    par ce facteur donne l'effectif effectif, ce qui élargit l'IC de √deff.

    **Le design effect appartient à une indicatrice, pas à un jeu de données** : sur
    ce corpus, celle de l'erreur donne ≈ 10, celle de l'accord ≈ 7,5, celle du
    rappel ≈ 4. Passer l'indicatrice concernée est donc obligatoire, sous peine de
    corriger avec le mauvais facteur. Corollaire : une statistique calculée à raison
    d'**une observation par grappe** (la prévalence d'un item, mesurée une fois par
    copie) n'a aucune corrélation intra-grappe à corriger — son deff vaut 1 et cette
    fonction n'a pas lieu d'être appelée.

    Args:
        indicator: indicatrice binaire (ou booléenne) dont on estime la moyenne.
        clusters: identifiant de grappe de chaque observation (la copie).

    Returns:
        Le design effect, dans [1, taille moyenne de grappe]. Les bornes sont
        atteintes quand l'ICC estimé sort de [0, 1], ce qui n'a pas de sens ici :
        la variance intra étant obtenue par différence, elle peut devenir négative
        sur des grappes dégénérées (parfaitement homogènes ou très peu nombreuses).
    """
    ind = pd.Series(np.asarray(indicator, dtype=float)).reset_index(drop=True)
    grappes = pd.Series(np.asarray(clusters, dtype=object)).reset_index(drop=True)
    if len(ind) != len(grappes):
        raise ValueError("indicator et clusters doivent avoir la même longueur.")

    taille_moyenne = float(ind.groupby(grappes).size().mean()) if len(ind) else 0.0
    var_totale = float(ind.var(ddof=1)) if len(ind) > 1 else 0.0
    if var_totale == 0 or taille_moyenne <= 1 or len(ind) < 2:
        return 1.0

    # Décomposition ANOVA à un facteur : variance inter-copies vs intra-copie.
    var_inter = float(ind.groupby(grappes).mean().var(ddof=1))
    var_intra = (var_totale - var_inter) / (taille_moyenne - 1)
    denom = var_inter + (taille_moyenne - 1) * var_intra
    if denom <= 0:
        return 1.0
    icc = min(1.0, max(0.0, (var_inter - var_intra) / denom))
    return max(1.0, 1 + (taille_moyenne - 1) * icc)


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
