"""Tests des synthèses de résultats (evaluation/results_summary)."""

import math

import pandas as pd
import pytest

from evaluation_dictee.evaluation.results_summary import htr_summary, scoring_summary


def _scoring_df() -> pd.DataFrame:
    # 4 items, 2 copies. Codes "1" = correct, autre = erreur.
    #   c1/i1 : expert=1 pred=1  → accord, pas d'erreur
    #   c1/i2 : expert=9 pred=9  → accord, vrai positif erreur
    #   c2/i1 : expert=9 pred=1  → désaccord, SUR-CORRECTION
    #   c2/i2 : expert=1 pred=9  → désaccord, sur-détection
    return pd.DataFrame(
        [
            {"copy_id": "c1", "item_id": "i1", "y_true": "1", "y_pred": "1", "confidence": 0.9},
            {"copy_id": "c1", "item_id": "i2", "y_true": "9", "y_pred": "9", "confidence": 0.8},
            {"copy_id": "c2", "item_id": "i1", "y_true": "9", "y_pred": "1", "confidence": 0.7},
            {"copy_id": "c2", "item_id": "i2", "y_true": "1", "y_pred": "9", "confidence": 0.6},
        ]
    )


def test_scoring_summary_core_metrics() -> None:
    """Accord, rappel, précision et sur-correction calculés depuis les codes bruts."""
    s = scoring_summary(_scoring_df(), run="r")
    assert s.run == "r"
    assert s.n_items == 4
    assert s.n_copies == 2
    assert s.raw_agreement == 0.5  # 2 accords / 4
    # erreurs expert = {i2, c2/i1} = 2 ; vrais positifs = {c1/i2} = 1
    assert s.recall_errors == pytest.approx(0.5)
    # erreurs modèle = {c1/i2, c2/i2} = 2 ; vrais positifs = 1
    assert s.precision_errors == pytest.approx(0.5)
    # sur-correction = expert erreur & modèle correct = {c2/i1} = 1 sur 4 items
    assert s.overcorrection_rate == pytest.approx(0.25)


def test_scoring_summary_types_coerced_and_ece_bounded() -> None:
    """Les codes numériques sont coercés en str ; l'ECE reste dans [0, 1]."""
    df = _scoring_df()
    df["y_true"] = df["y_true"].astype(int)  # simule un JSONL avec codes entiers
    s = scoring_summary(df, run="r")
    assert s.n_items == 4
    assert 0.0 <= s.ece <= 1.0


def test_scoring_summary_empty_raises() -> None:
    with pytest.raises(ValueError):
        scoring_summary(pd.DataFrame(), run="r")


def _htr_df() -> pd.DataFrame:
    # Un échantillon non transcrit (transcrit=False) doit être exclu des moyennes.
    return pd.DataFrame(
        [
            {
                "cer": 0.10,
                "wer": 0.20,
                "cer_normalise": 0.05,
                "wer_normalise": 0.10,
                "n_char_ref": 100,
                "n_mots_ref": 20,
                "transcrit": True,
            },
            {
                "cer": 0.30,
                "wer": 0.40,
                "cer_normalise": 0.15,
                "wer_normalise": 0.20,
                "n_char_ref": 100,
                "n_mots_ref": 20,
                "transcrit": True,
            },
            {
                "cer": 1.00,
                "wer": 1.00,
                "cer_normalise": 1.00,
                "wer_normalise": 1.00,
                "n_char_ref": 100,
                "n_mots_ref": 20,
                "transcrit": False,
            },
        ]
    )


def test_htr_summary_micro_weighted_excludes_untranscribed() -> None:
    """CER/WER micro-pondérés sur les seuls échantillons transcrits."""
    s = htr_summary(_htr_df(), run="htr")
    assert s.n_samples == 3
    # (0.10*100 + 0.30*100) / 200 = 0.20, l'échantillon non transcrit est ignoré.
    assert s.cer == pytest.approx(0.20)
    assert s.wer == pytest.approx(0.30)
    assert s.cer_normalise == pytest.approx(0.10)


def test_htr_summary_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        htr_summary(pd.DataFrame([{"cer": 0.1}]), run="htr")


def test_htr_summary_zero_weight_is_nan() -> None:
    """Poids total nul → NaN plutôt qu'une division par zéro."""
    df = pd.DataFrame(
        [
            {
                "cer": 0.1,
                "wer": 0.2,
                "cer_normalise": 0.0,
                "wer_normalise": 0.0,
                "n_char_ref": 0,
                "n_mots_ref": 0,
                "transcrit": True,
            }
        ]
    )
    s = htr_summary(df, run="htr")
    assert math.isnan(s.cer)
