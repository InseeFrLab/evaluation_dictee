"""Calibration de la confiance et courbe de renvoi humain (livrable décisionnel, CLAUDE.md §4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReferralPoint:
    """Un point de la courbe renvoi-humain / erreur-résiduelle."""

    threshold: float
    human_referral_rate: float
    residual_error_rate: float
    n_auto_validated: int


def referral_curve(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float | None],
    thresholds: list[float] | None = None,
) -> list[ReferralPoint]:
    """Courbe « taux de renvoi humain vs taux d'erreur résiduel ».

    Un item est renvoyé si sa confiance est < seuil ou absente (None).

    Args:
        y_true: codes experts.
        y_pred: codes prédits, alignés sur y_true.
        confidences: confiance de chaque prédiction (None = pas de confiance).
        thresholds: seuils à balayer ; par défaut 0.0 à 1.0 par pas de 0.1.

    Returns:
        Un point par seuil : taux de renvoi humain, erreur résiduelle sur les
        items auto-validés et nombre d'items auto-validés.
    """
    if thresholds is None:
        thresholds = [i / 10 for i in range(11)]

    n_total = len(y_true)
    points: list[ReferralPoint] = []

    for threshold in thresholds:
        kept_true: list[str] = []
        kept_pred: list[str] = []
        for true_code, pred_code, conf in zip(y_true, y_pred, confidences, strict=True):
            if conf is not None and conf >= threshold:
                kept_true.append(true_code)
                kept_pred.append(pred_code)

        n_kept = len(kept_true)
        n_referred = n_total - n_kept
        if n_kept > 0:
            errors = sum(1 for a, b in zip(kept_true, kept_pred, strict=True) if a != b)
            residual = errors / n_kept
        else:
            residual = 0.0

        points.append(
            ReferralPoint(
                threshold=threshold,
                human_referral_rate=n_referred / n_total,
                residual_error_rate=residual,
                n_auto_validated=n_kept,
            )
        )
    return points


@dataclass
class ReliabilityBin:
    """Une tranche de confiance pour le diagramme de fiabilité."""

    bin_lower: float
    bin_upper: float
    mean_confidence: float
    accuracy: float
    n: int


def reliability_bins(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float | None],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Regroupe les items par tranche de confiance et mesure l'accord réel.

    Un modèle bien calibré a accuracy ≈ mean_confidence dans chaque tranche.

    Args:
        y_true: codes experts.
        y_pred: codes prédits, alignés sur y_true.
        confidences: confiance de chaque prédiction (None = item ignoré).
        n_bins: nombre de tranches de confiance sur [0, 1].

    Returns:
        Les tranches non vides, chacune avec ses bornes, la confiance moyenne,
        l'accord observé et l'effectif.
    """
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        confs, corrects = [], []
        for t, p, c in zip(y_true, y_pred, confidences, strict=True):
            if c is None:
                continue
            # Dernière tranche inclusive à droite pour capter c == 1.0
            if (lo <= c < hi) or (b == n_bins - 1 and c == hi):
                confs.append(c)
                corrects.append(t == p)
        if confs:
            bins.append(
                ReliabilityBin(
                    bin_lower=lo,
                    bin_upper=hi,
                    mean_confidence=sum(confs) / len(confs),
                    accuracy=sum(corrects) / len(corrects),
                    n=len(confs),
                )
            )
    return bins


def expected_calibration_error(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float | None],
    n_bins: int = 10,
) -> float:
    """ECE : écart moyen pondéré entre confiance annoncée et accord observé.

    0 = parfaitement calibré. Au-delà de ~0.1, la confiance doit être recalibrée
    avant usage pour le seuil de renvoi humain.

    Args:
        y_true: codes experts.
        y_pred: codes prédits, alignés sur y_true.
        confidences: confiance de chaque prédiction (None = item ignoré).
        n_bins: nombre de tranches de confiance sur [0, 1].

    Returns:
        L'ECE (écart absolu accord/confiance pondéré par les effectifs), ou NaN
        si aucun item n'a de confiance.
    """
    bins = reliability_bins(y_true, y_pred, confidences, n_bins)
    n_total = sum(b.n for b in bins)
    if n_total == 0:
        return float("nan")
    return sum(b.n * abs(b.accuracy - b.mean_confidence) for b in bins) / n_total
