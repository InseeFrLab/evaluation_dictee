"""Scorer en deux étapes (APPROCHE 1) : étape 1 HTR (lecture image), étape 2 codage textuel.

Découple lecture et jugement ; même interface `Scorer` que `VLMScorer` pour une comparaison juste.
"""

from __future__ import annotations

import json
import re
from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from evaluation_dictee.config import ModelConfig, PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.data.loaders import Copy, load_image
from evaluation_dictee.models.base import CopyPrediction, ItemPrediction, Scorer
from evaluation_dictee.models.vlm import _image_to_data_url
from evaluation_dictee.pipeline.alignment import best_realignment, needs_realignment
from evaluation_dictee.pipeline.prompts import (
    attach_image,
    build_text_coding_prompt,
    build_transcription_prompt,
)
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_items_from_content(content: str) -> list[dict]:
    """Extrait la liste d'items d'une réponse modèle, du parsing strict au plus permissif.

    Args:
        content: Contenu textuel brut renvoyé par le modèle.

    Returns:
        Liste des items (dictionnaires bruts), vide si rien n'est extractible.
    """
    if not content:
        return []

    # Tentative 1 : JSON complet
    cleaned = content.strip().removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
        items = data.get("items")
        if isinstance(items, list) and items:
            return items
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    # Tentative 2 : trouver le premier objet racine avec une clé "items"
    m = re.search(r'\{[^{}]*"items"\s*:\s*\[', content)
    if m:
        start = m.start()
        # Aller jusqu'au } équilibré correspondant (approximation)
        depth = 0
        end = None
        for i in range(start, len(content)):
            c = content[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end:
            try:
                data = json.loads(content[start:end])
                items = data.get("items")
                if isinstance(items, list) and items:
                    return items
            except json.JSONDecodeError:
                pass

    # Tentative 3 : récupérer les sous-objets d'item isolés (utile si la liste est tronquée).
    items = []
    for match in re.finditer(
        r'\{\s*"item_id"\s*:\s*"[^"]+"[^{}]*"code"\s*:\s*"[^"]+"[^{}]*\}',
        content,
    ):
        try:
            items.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return items


class TwoStageScorer(Scorer):
    """Évalue une copie en deux étapes : transcription HTR puis codage du texte."""

    def __init__(
        self,
        model_config: ModelConfig,
        prompt_config: PromptConfig,
        base_url: str,
        api_key: str,
        grid_items: list[GridItem],
        scheme: str = "simplifiee",
        model_config_stage2: ModelConfig | None = None,
    ) -> None:
        """Initialise le client, l'index des items et les configs des deux étapes.

        Args:
            model_config: Configuration du modèle de l'étape 1 (transcription HTR).
            prompt_config: Configuration du prompt (lecture de l'état final, etc.).
            base_url: URL de base de l'API compatible OpenAI.
            api_key: Clé d'API.
            grid_items: Items de la grille de codage.
            scheme: Grille de codage utilisée (par défaut « simplifiee »).
            model_config_stage2: Configuration du modèle de l'étape 2 (codage textuel) ;
                reprend `model_config` si None.
        """
        self.model_config = model_config
        self.model_config_stage2 = model_config_stage2 or model_config
        self.prompt_config = prompt_config
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.grid_items = grid_items
        self.scheme = scheme
        self._items_by_id = {it.item_id: it for it in grid_items}

    # ── Étape 1 : transcription HTR ──────────────────────────────────────────
    def _transcribe(self, copy: Copy) -> tuple[str, int]:
        """Lit l'image et renvoie (transcription, nb_tentatives) ; vide si échec total.

        Args:
            copy: Copie dont l'image est à transcrire.

        Returns:
            Couple (transcription, nombre de tentatives) ; transcription vide si aucun
            essai n'a produit de texte exploitable.
        """
        image = load_image(copy.image_path)
        messages = cast(
            "list[ChatCompletionMessageParam]",
            attach_image(
                build_transcription_prompt(read_final_state=self.prompt_config.read_final_state),
                _image_to_data_url(image),
            ),
        )
        for attempt in range(self.model_config.max_retries + 1):
            temp = self.model_config.temperature + (0.3 if attempt > 0 else 0.0)
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                temperature=temp,
                max_tokens=self.model_config.max_tokens,
                messages=messages,
            )
            content = response.choices[0].message.content or "{}"
            transcription = self._parse_transcription(content)
            if transcription.strip():
                return transcription, attempt + 1
        return "", self.model_config.max_retries + 1

    @staticmethod
    def _parse_transcription(content: str) -> str:
        """Extrait le champ 'transcription' du JSON renvoyé à l'étape 1.

        Args:
            content: Contenu textuel brut renvoyé par le modèle.

        Returns:
            Transcription extraite ; repli sur le texte brut si le JSON est illisible.
        """
        try:
            cleaned = content.strip().removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
            data = json.loads(cleaned)
            return str(data.get("transcription", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            # Repli : réponse en texte brut prise telle quelle.
            return content.strip()

    # ── Étape 2 : codage du texte transcrit ──────────────────────────────────
    def _code_text(self, copy: Copy, reference_text: str, transcription: str) -> CopyPrediction:
        """Code chaque item à partir de la transcription (sans image).

        Le retry évite qu'un échec ponctuel de génération JSON code toute la copie en '?'.

        Args:
            copy: Copie évaluée (fournit la liste d'items à coder).
            reference_text: Texte de référence de la dictée.
            transcription: Transcription produite à l'étape 1.

        Returns:
            Prédiction de la copie ; items codés « ? » si aucun essai n'est exploitable.
        """
        items_a_coder = [self._items_by_id[i] for i in copy.item_ids if i in self._items_by_id]
        messages = cast(
            "list[ChatCompletionMessageParam]",
            build_text_coding_prompt(
                reference_text=reference_text,
                transcription=transcription,
                items=items_a_coder,
                scheme=self.scheme,
            ),
        )
        for attempt in range(self.model_config_stage2.max_retries + 1):
            temp = self.model_config_stage2.temperature + (0.3 if attempt > 0 else 0.0)
            response = self.client.chat.completions.create(
                model=self.model_config_stage2.name,
                temperature=temp,
                max_tokens=self.model_config_stage2.max_tokens,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            if _extract_items_from_content(content):
                return self._parse_coding(copy, content)
            logger.warning(
                "Étape 2 : JSON non extractible à l'essai %d/%d pour %s. Nouvel essai.",
                attempt + 1,
                self.model_config_stage2.max_retries + 1,
                copy.copy_id,
            )
        # Échec total : on parse quand même pour laisser _parse_coding logger et remonter des '?'.
        return self._parse_coding(copy, content)

    def _parse_coding(self, copy: Copy, content: str) -> CopyPrediction:
        """Parse le JSON de codage de l'étape 2, avec ré-alignement de sécurité (cf. VLMScorer).

        Args:
            copy: Copie évaluée (fournit la liste d'items attendus).
            content: Contenu textuel brut renvoyé par le modèle à l'étape 2.

        Returns:
            Prédiction de la copie ; items codés « ? » pour ceux absents de la réponse.
        """
        raw_items = _extract_items_from_content(content)
        if not raw_items:
            logger.warning(
                "Étape 2 (codage textuel) : aucun item extractible du JSON pour la "
                "copie %s. Réponse tronquée ou format cassé (longueur brute : %d). "
                "Tous les items seront codés '?'. Extrait : %r",
                copy.copy_id,
                len(content),
                content[:200],
            )

        codes_seq = [str(it.get("code", "?")).strip() for it in raw_items]
        trans_seq = [it.get("transcription") for it in raw_items]
        conf_seq = [it.get("confidence") for it in raw_items]

        expected_words = [
            self._items_by_id[i].attendu for i in copy.item_ids if i in self._items_by_id
        ]

        if codes_seq and needs_realignment(expected_words, trans_seq):
            aligned = best_realignment(expected_words, codes_seq, trans_seq, conf_seq)
            items = [
                ItemPrediction(
                    item_id=item_id,
                    code=a.code,
                    confidence=a.confidence,
                    transcription=a.transcription,
                )
                for item_id, a in zip(copy.item_ids, aligned, strict=False)
            ]
            return CopyPrediction(copy_id=copy.copy_id, items=items)

        by_id = {it.get("item_id"): it for it in raw_items}
        items = []
        for item_id in copy.item_ids:
            entry = by_id.get(item_id)
            if entry is None:
                items.append(ItemPrediction(item_id=item_id, code="?", confidence=0.0))
            else:
                items.append(
                    ItemPrediction(
                        item_id=item_id,
                        code=str(entry.get("code", "?")).strip(),
                        confidence=entry.get("confidence"),
                        transcription=entry.get("transcription"),
                        comparaison=entry.get("comparaison"),
                    )
                )
        return CopyPrediction(copy_id=copy.copy_id, items=items)

    # ── Interface Scorer ─────────────────────────────────────────────────────
    def score_copy(self, copy: Copy, reference_text: str | None) -> CopyPrediction:
        """Évalue une copie : transcription puis codage (non transcrite si l'étape 1 échoue).

        Args:
            copy: Copie à évaluer.
            reference_text: Texte de référence de la dictée, ou None.

        Returns:
            Prédiction de la copie ; `transcribed=False` si l'étape 1 (HTR) n'a produit
            aucune transcription. La transcription brute est conservée dans
            `raw_transcription`.
        """
        transcription, n_attempts = self._transcribe(copy)

        if not transcription.strip():
            items_vides = [
                ItemPrediction(item_id=i, code="?", confidence=0.0) for i in copy.item_ids
            ]
            return CopyPrediction(
                copy_id=copy.copy_id,
                items=items_vides,
                transcribed=False,
                n_attempts=n_attempts,
            )

        prediction = self._code_text(copy, reference_text or "", transcription)
        prediction.n_attempts = n_attempts
        prediction.raw_transcription = transcription
        return prediction
