"""Interface commune `Scorer` à tous les modèles d'évaluation (méthodes A/B/C/D comparables)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from evaluation_dictee.data.loaders import Copy

#: Code posé quand la réponse du modèle n'a pu être ni parsée ni alignée sur l'item
#: (réponse vide, JSON cassé, item absent de la réponse). Ce n'est PAS un code de la
#: grille : il marque un échec technique, à distinguer d'un désaccord de jugement.
#: Défini ici parce que c'est l'interface `Scorer` qui l'émet — les deux scorers, le
#: ré-alignement et le pipeline s'y réfèrent.
CODE_NON_PARSE = "?"


@dataclass
class ItemPrediction:
    """Prédiction du modèle pour un item."""

    item_id: str
    code: str
    confidence: float | None = None
    transcription: str | None = None
    comparaison: str | None = None  # différence lue-attendue, renseignée en mode chain-of-thought
    # True si le modèle a bien écrit « comparaison » AVANT « code » dans son JSON.
    # Le décodage contraint de vLLM n'impose PAS l'ordre des clés (mesuré) : une
    # comparaison écrite après le code est une justification a posteriori, pas un
    # raisonnement — la distinction doit rester visible dans les résultats.
    comparaison_avant_code: bool | None = None


@dataclass
class CopyPrediction:
    """Ensemble des prédictions du modèle pour une copie."""

    copy_id: str
    items: list[ItemPrediction]
    transcribed: bool = True  # False = aucune transcription exploitable, copie exclue des métriques
    n_attempts: int = 1
    # Transcription brute de l'étape 1 (two_stage) ; None en end_to_end.
    raw_transcription: str | None = None
    # Raisonnement natif du modèle (champ `reasoning_content` de l'API), quand le mode
    # thinking est actif. C'est du niveau COPIE, pas de l'item : le modèle produit un
    # seul bloc pour les 83 items. Stocké à part (`<run>_reasoning.jsonl`) et jamais
    # recopié sur chaque ligne d'item — 83 x 13 000 caractères par copie sinon.
    reasoning: str | None = None
    # Nombre d'items que le modèle déclare avoir LUS sur la copie (option count_items).
    # Comparable à ce que déclare l'expert : 83 moins ses codes "0". None si l'option
    # est inactive ou si le champ n'a pas été renvoyé.
    n_items_lus: int | None = None


class Scorer(ABC):
    """Contrat que doit respecter tout modèle d'évaluation de copie (implémenter `score_copy`)."""

    @abstractmethod
    def score_copy(self, copy: Copy, reference_text: str | None) -> CopyPrediction:
        """Évalue une copie et renvoie un code (+ confiance) par item.

        Args:
            copy: Copie à évaluer (image et identifiants d'items).
            reference_text: Texte de référence de la dictée, ou None si inconnu.

        Returns:
            Prédiction de la copie : un `ItemPrediction` par item.
        """
        raise NotImplementedError
