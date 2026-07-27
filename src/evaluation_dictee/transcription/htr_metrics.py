"""Métriques HTR (CER, WER) : fidélité de LECTURE, pas correction orthographique.

Une variante « normalisée » (minuscules, sans accents ni ponctuation) isole les
erreurs de lecture substantielles des simples différences de casse/accents.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


def _levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    """Distance d'édition entre a et b (chaînes = niveau caractère, listes = niveau mot).

    Args:
        a: Première séquence (chaîne ou liste de mots).
        b: Seconde séquence (chaîne ou liste de mots).

    Returns:
        Nombre minimal d'insertions, suppressions et substitutions.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _normalize_text(s: str) -> str:
    """Minuscule, sans accents, sans ponctuation (pour la variante normalisée).

    Args:
        s: Texte à normaliser.

    Returns:
        Texte en minuscules, sans accents ni ponctuation, espaces réduits.
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower())
    return " ".join(s.split())


@dataclass
class TranscriptionMetrics:
    """Métriques de transcription pour un échantillon ou un corpus."""

    cer: float
    wer: float
    cer_normalise: float
    wer_normalise: float
    n_char_ref: int
    n_mots_ref: int


def compute_transcription_metrics(reference: str, hypothesis: str) -> TranscriptionMetrics:
    """Calcule CER et WER (bruts et normalisés) entre référence et hypothèse.

    Args:
        reference: Transcription de référence.
        hypothesis: Transcription produite par le modèle.

    Returns:
        Métriques CER/WER bruts et normalisés, et longueurs de la référence.
    """
    ref_c, hyp_c = reference, hypothesis
    ref_w, hyp_w = reference.split(), hypothesis.split()

    n_char = max(len(ref_c), 1)
    n_mots = max(len(ref_w), 1)
    cer = _levenshtein(ref_c, hyp_c) / n_char
    wer = _levenshtein(ref_w, hyp_w) / n_mots

    rn, hn = _normalize_text(reference), _normalize_text(hypothesis)
    rn_w, hn_w = rn.split(), hn.split()
    cer_n = _levenshtein(rn, hn) / max(len(rn), 1)
    wer_n = _levenshtein(rn_w, hn_w) / max(len(rn_w), 1)

    return TranscriptionMetrics(
        cer=cer,
        wer=wer,
        cer_normalise=cer_n,
        wer_normalise=wer_n,
        n_char_ref=len(ref_c),
        n_mots_ref=len(ref_w),
    )
