"""Synthèses prêtes à afficher pour l'onglet Résultats du site Quarto.

Chaque fonction condense un DataFrame de prédictions (chargé par
`report.load_predictions`, local ou S3) en une poignée de chiffres agrégés,
correspondant aux lignes des tableaux comparatifs de `website/resultats.qmd`.

Convention de codage (grille simplifiée) : un code vaut "1" si l'item est
correct, autre chose sinon (« erreur »). Voir CLAUDE.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from evaluation_dictee.evaluation.calibration import expected_calibration_error

CORRECT_CODE = "1"


@dataclass
class ScoringSummary:
    """Chiffres agrégés d'un run de codage (une colonne du tableau end-to-end vs two-stage)."""

    run: str
    n_items: int
    n_copies: int
    raw_agreement: float
    cohen_kappa: float
    recall_errors: float  # rappel sur la classe « erreur »
    precision_errors: float  # précision sur la classe « erreur »
    # part d'items « corrigés » silencieusement (expert=erreur, modèle=correct)
    overcorrection_rate: float
    ece: float  # Expected Calibration Error


@dataclass
class HTRSummary:
    """Chiffres agrégés d'un run de transcription HTR (une ligne du tableau HTR)."""

    run: str
    n_samples: int
    cer: float
    wer: float
    cer_normalise: float
    wer_normalise: float


def scoring_summary(df: pd.DataFrame, run: str = "") -> ScoringSummary:
    """Condense les prédictions de codage d'un run en métriques de tableau.

    Args:
        df: prédictions à l'item (colonnes y_true, y_pred, confidence, copy_id).
        run: nom du run, repris tel quel dans la synthèse.

    Returns:
        La synthèse agrégée du run.

    Raises:
        ValueError: si le DataFrame est vide ou manque des colonnes attendues.
    """
    required = {"y_true", "y_pred"}
    if df.empty or not required.issubset(df.columns):
        raise ValueError("DataFrame de codage vide ou colonnes y_true/y_pred manquantes.")

    y_true = df["y_true"].astype(str)
    y_pred = df["y_pred"].astype(str)
    n = len(df)

    exp_err = y_true != CORRECT_CODE
    mod_err = y_pred != CORRECT_CODE
    vrais_pos = int((exp_err & mod_err).sum())
    n_exp_err = int(exp_err.sum())
    n_mod_err = int(mod_err.sum())

    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except ValueError:
        kappa = float("nan")

    confidences = df["confidence"].tolist() if "confidence" in df.columns else [None] * n

    return ScoringSummary(
        run=run,
        n_items=n,
        n_copies=int(df["copy_id"].nunique()) if "copy_id" in df.columns else 0,
        raw_agreement=float((y_true == y_pred).mean()),
        cohen_kappa=kappa,
        recall_errors=vrais_pos / n_exp_err if n_exp_err else float("nan"),
        precision_errors=vrais_pos / n_mod_err if n_mod_err else float("nan"),
        overcorrection_rate=int((exp_err & ~mod_err).sum()) / n if n else float("nan"),
        ece=expected_calibration_error(y_true.tolist(), y_pred.tolist(), confidences),
    )


def _micro_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    """Moyenne micro-pondérée d'une colonne par une colonne de poids (longueur de référence)."""
    weights = df[weight_col]
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float((df[value_col] * weights).sum() / total)


def htr_summary(df: pd.DataFrame, run: str = "") -> HTRSummary:
    """Condense les prédictions HTR d'un run en CER/WER micro-pondérés.

    Args:
        df: prédictions HTR (colonnes cer, wer, cer_normalise, wer_normalise,
            n_char_ref, n_mots_ref, transcrit).
        run: nom du run, repris tel quel dans la synthèse.

    Returns:
        La synthèse agrégée du run.

    Raises:
        ValueError: si le DataFrame est vide ou manque des colonnes attendues.
    """
    required = {"cer", "wer", "cer_normalise", "wer_normalise", "n_char_ref", "n_mots_ref"}
    if df.empty or not required.issubset(df.columns):
        raise ValueError("DataFrame HTR vide ou colonnes CER/WER manquantes.")

    # Micro-pondération sur les seuls échantillons transcrits (mêmes règles que
    # transcription/htr_benchmark.py).
    ok = df[df["transcrit"]] if "transcrit" in df.columns else df

    return HTRSummary(
        run=run,
        n_samples=len(df),
        cer=_micro_mean(ok, "cer", "n_char_ref"),
        wer=_micro_mean(ok, "wer", "n_mots_ref"),
        cer_normalise=_micro_mean(ok, "cer_normalise", "n_char_ref"),
        wer_normalise=_micro_mean(ok, "wer_normalise", "n_mots_ref"),
    )
