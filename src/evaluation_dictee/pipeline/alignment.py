"""Ré-alignement des prédictions sur les items attendus (filet contre décalages mot scindé/collé).

Alignement de séquences (type Needleman-Wunsch) entre mots attendus et transcriptions du modèle,
appliqué seulement si un décalage est détecté ; sinon les prédictions positionnelles restent.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass
class AlignedPrediction:
    """Prédiction d'un item après ré-alignement."""

    code: str
    transcription: str | None
    confidence: float | None
    realigned: bool = False  # True si l'item a été déplacé par le ré-alignement


def _norm(s: str | None) -> str:
    """Normalise une chaîne pour la comparaison (minuscule, sans accents/espaces).

    Args:
        s: chaîne à normaliser (None traité comme chaîne vide).

    Returns:
        La chaîne en minuscules, sans accents ni espaces.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(s.lower().split())


def _similar(attendu: str, lu: str | None) -> float:
    """Score de similarité simple entre mot attendu et transcription (0..1).

    Args:
        attendu: mot attendu de l'item.
        lu: transcription proposée par le modèle (None si absente).

    Returns:
        Un score dans [0, 1] : 1.0 si identiques après normalisation, 0.8 si l'un
        inclut l'autre, sinon la proportion de préfixe commun.
    """
    a, b = _norm(attendu), _norm(lu)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Similarité par préfixe commun + inclusion (capte « re trouver » vs « retrouver »)
    if a in b or b in a:
        return 0.8
    commun = 0
    for ca, cb in zip(a, b, strict=False):
        if ca == cb:
            commun += 1
        else:
            break
    return commun / max(len(a), len(b))


def needs_realignment(
    expected_words: list[str],
    transcriptions: list[str | None],
    seuil: float = 0.5,
) -> bool:
    """Détecte un décalage : similarité moyenne transcription/mot attendu sous `seuil`.

    Args:
        expected_words: mots attendus, dans l'ordre des items.
        transcriptions: transcriptions positionnelles du modèle (None ignorées).
        seuil: seuil de similarité moyenne sous lequel un décalage est signalé.

    Returns:
        True si un ré-alignement est nécessaire, False sinon (ou si aucune paire comparable).
    """
    paires = [
        _similar(a, t)
        for a, t in zip(expected_words, transcriptions, strict=False)
        if t is not None
    ]
    if not paires:
        return False
    return (sum(paires) / len(paires)) < seuil


def realign(
    expected_words: list[str],
    codes: list[str],
    transcriptions: list[str | None],
    confidences: list[float | None],
    gap_penalty: float = -0.3,
) -> list[AlignedPrediction]:
    """Ré-aligne les prédictions sur les mots attendus (Needleman-Wunsch) ; mot absent codé "0".

    Args:
        expected_words: mots attendus, un par item.
        codes: codes prédits par le modèle, dans l'ordre de ses tokens.
        transcriptions: transcriptions correspondant aux codes.
        confidences: scores de confiance correspondant aux codes.
        gap_penalty: pénalité appliquée à un saut (mot non lu ou token en trop).

    Returns:
        Une prédiction alignée par mot attendu ; "0" si le mot n'a pas été lu, "?"
        en repli si aucune correspondance n'a pu être établie.
    """
    n, m = len(expected_words), len(codes)
    # score[i][j] = meilleur alignement des i premiers mots attendus et j premières prédictions.
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap_penalty
    for j in range(1, m + 1):
        score[0][j] = j * gap_penalty
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i - 1][j - 1] + _similar(expected_words[i - 1], transcriptions[j - 1])
            delete = score[i - 1][j] + gap_penalty  # mot attendu sans correspondance
            insert = score[i][j - 1] + gap_penalty  # token modèle en trop
            score[i][j] = max(match, delete, insert)

    # Backtrack
    aligned: list[AlignedPrediction | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        sim = _similar(expected_words[i - 1], transcriptions[j - 1])
        if score[i][j] == score[i - 1][j - 1] + sim:
            aligned[i - 1] = AlignedPrediction(
                code=codes[j - 1],
                transcription=transcriptions[j - 1],
                confidence=confidences[j - 1],
                realigned=(i - 1) != (j - 1),
            )
            i, j = i - 1, j - 1
        elif score[i][j] == score[i - 1][j] + gap_penalty:
            # mot attendu non lu → absent
            aligned[i - 1] = AlignedPrediction("0", None, 0.0, realigned=True)
            i -= 1
        else:
            j -= 1  # token modèle en trop : ignoré
    while i > 0:
        aligned[i - 1] = AlignedPrediction("0", None, 0.0, realigned=True)
        i -= 1

    return [a if a is not None else AlignedPrediction("?", None, 0.0, True) for a in aligned]


def _alignment_quality(expected_words: list[str], aligned: list[AlignedPrediction]) -> float:
    """Qualité d'un alignement : similarité moyenne mot attendu / transcription dans [0, 1].

    Args:
        expected_words: mots attendus, un par item.
        aligned: prédictions alignées à évaluer.

    Returns:
        La similarité moyenne dans [0, 1] (0.0 si la liste est vide).
    """
    sims = [_similar(w, a.transcription) for w, a in zip(expected_words, aligned, strict=False)]
    return sum(sims) / len(sims) if sims else 0.0


def realign_anchored(
    expected_words: list[str],
    codes: list[str],
    transcriptions: list[str | None],
    confidences: list[float | None],
) -> list[AlignedPrediction]:
    """Alignement par ancrage sur les correspondances exactes uniques, réparti entre les ancres.

    Complémentaire de `realign` ; robuste quand des mots distinctifs jalonnent la dictée.

    Args:
        expected_words: mots attendus, un par item.
        codes: codes prédits par le modèle, dans l'ordre de ses tokens.
        transcriptions: transcriptions correspondant aux codes.
        confidences: scores de confiance correspondant aux codes.

    Returns:
        Une prédiction alignée par mot attendu ; "0" si le mot n'a pas été lu, "?"
        en repli si aucune correspondance n'a pu être établie.
    """
    n = len(expected_words)
    m = len(codes)
    norm_exp = [_norm(w) for w in expected_words]
    norm_tr = [_norm(t) for t in transcriptions]

    # Ancres : mot attendu unique == transcription unique
    from collections import Counter

    cnt_exp, cnt_tr = Counter(norm_exp), Counter(norm_tr)
    anchors: list[tuple[int, int]] = []  # (index attendu, index modèle)
    for i, w in enumerate(norm_exp):
        if not w or cnt_exp[w] != 1 or cnt_tr.get(w, 0) != 1:
            continue
        j = norm_tr.index(w)
        anchors.append((i, j))
    anchors.sort()

    result: list[AlignedPrediction | None] = [None] * n
    # Segments délimités par les ancres ; alignement positionnel à l'intérieur
    bornes = [(-1, -1), *anchors, (n, m)]
    for (i0, j0), (i1, j1) in zip(bornes, bornes[1:], strict=False):
        # placer l'ancre i1 elle-même
        if 0 <= i1 < n and 0 <= j1 < m:
            result[i1] = AlignedPrediction(
                codes[j1], transcriptions[j1], confidences[j1], realigned=(i1 != j1)
            )
        # répartir le segment ouvert (i0, i1) ↔ (j0, j1) positionnellement
        seg_exp = list(range(i0 + 1, i1))
        seg_mod = list(range(j0 + 1, j1))
        for k, ie in enumerate(seg_exp):
            if k < len(seg_mod):
                jm = seg_mod[k]
                result[ie] = AlignedPrediction(
                    codes[jm], transcriptions[jm], confidences[jm], realigned=(ie != jm)
                )
            else:
                result[ie] = AlignedPrediction("0", None, 0.0, realigned=True)

    return [a if a is not None else AlignedPrediction("?", None, 0.0, True) for a in result]


def best_realignment(
    expected_words: list[str],
    codes: list[str],
    transcriptions: list[str | None],
    confidences: list[float | None],
) -> list[AlignedPrediction]:
    """Applique Needleman-Wunsch et ancrage, garde l'alignement le plus proche des mots attendus.

    Args:
        expected_words: mots attendus, un par item.
        codes: codes prédits par le modèle, dans l'ordre de ses tokens.
        transcriptions: transcriptions correspondant aux codes.
        confidences: scores de confiance correspondant aux codes.

    Returns:
        L'alignement (parmi `realign` et `realign_anchored`) de meilleure qualité.
    """
    candidats = [
        realign(expected_words, codes, transcriptions, confidences),
        realign_anchored(expected_words, codes, transcriptions, confidences),
    ]
    return max(candidats, key=lambda a: _alignment_quality(expected_words, a))
