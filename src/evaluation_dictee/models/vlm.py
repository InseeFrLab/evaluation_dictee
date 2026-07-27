"""VLM (vLLM, API OpenAI) pour la méthode C end-to-end : image + référence -> code JSON par item."""

from __future__ import annotations

import base64
import io
import json
from typing import Any, cast

from langfuse.openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from evaluation_dictee.config import ModelConfig, PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.data.loaders import Copy, load_image
from evaluation_dictee.models.base import CopyPrediction, ItemPrediction, Scorer
from evaluation_dictee.pipeline.alignment import best_realignment, needs_realignment
from evaluation_dictee.pipeline.prompts import (
    PROMPT_DICTATION,
    attach_image,
    build_dictation_prompt,
    fetch_prompt,
)


def _image_to_data_url(image: Image.Image) -> str:
    """Encode une image PIL en data URL base64 (format attendu par l'API).

    Args:
        image: Image PIL à encoder.

    Returns:
        Data URL `data:image/png;base64,...` de l'image encodée en PNG.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _items_json_schema(chain_of_thought: bool) -> dict[str, Any]:
    """Schéma JSON de la réponse attendue (méthode C) pour le décodage contraint vLLM.

    Contraint la structure de chaque item, pas leur nombre (variable selon la copie).

    Args:
        chain_of_thought: Si True, ajoute le champ obligatoire `comparaison` à chaque item.

    Returns:
        Schéma JSON de la réponse attendue (objet avec une liste `items`).
    """
    properties: dict[str, Any] = {
        "item_id": {"type": "string"},
        "transcription": {"type": "string"},
        "code": {"type": "string"},
        "confidence": {"type": "number"},
    }
    required = ["item_id", "transcription", "code", "confidence"]
    if chain_of_thought:
        properties["comparaison"] = {"type": "string"}
        required.append("comparaison")
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": properties, "required": required},
            }
        },
        "required": ["items"],
    }


class VLMScorer(Scorer):
    """Évalue une copie via un VLM (méthode C end-to-end)."""

    def __init__(
        self,
        model_config: ModelConfig,
        prompt_config: PromptConfig,
        base_url: str,
        api_key: str,
        grid_items: list[GridItem],
        scheme: str = "simplifiee",
    ) -> None:
        """Initialise le client et l'index des items de la grille.

        Args:
            model_config: Configuration du modèle (nom, température, retries, etc.).
            prompt_config: Configuration du prompt (chain-of-thought, etc.).
            base_url: URL de base de l'API compatible OpenAI.
            api_key: Clé d'API.
            grid_items: Items de la grille de codage.
            scheme: Grille de codage utilisée (par défaut « simplifiee »).
        """
        self.model_config = model_config
        self.prompt_config = prompt_config
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.grid_items = grid_items
        self.scheme = scheme
        self._items_by_id = {it.item_id: it for it in grid_items}

    def score_copy(self, copy: Copy, reference_text: str | None) -> CopyPrediction:
        """Évalue une copie complète en un appel au VLM (avec retries à température relevée).

        Args:
            copy: Copie à évaluer.
            reference_text: Texte de référence de la dictée, ou None.

        Returns:
            Prédiction de la copie ; `transcribed=False` si aucun essai n'a produit
            de transcription exploitable.
        """
        image = load_image(copy.image_path)
        items_a_coder = [self._items_by_id[i] for i in copy.item_ids if i in self._items_by_id]
        messages = cast(
            "list[ChatCompletionMessageParam]",
            attach_image(
                build_dictation_prompt(
                    reference_text=reference_text or "",
                    items=items_a_coder,
                    config=self.prompt_config,
                    scheme=self.scheme,
                ),
                _image_to_data_url(image),
            ),
        )
        # Lie la génération à la version du prompt dans les traces Langfuse (no-op si hors ligne).
        prompt_ref = fetch_prompt(PROMPT_DICTATION)
        trace_kwargs: dict[str, Any] = {"langfuse_prompt": prompt_ref} if prompt_ref else {}

        response_format_kwargs: dict[str, Any] = {}
        if self.model_config.structured_output:
            response_format_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "codage_dictee",
                    "schema": _items_json_schema(self.prompt_config.chain_of_thought),
                },
            }

        # Retry avec température légèrement relevée pour sortir d'une réponse vide déterministe.
        prediction: CopyPrediction | None = None
        for attempt in range(self.model_config.max_retries + 1):
            temp = self.model_config.temperature + (0.3 if attempt > 0 else 0.0)
            # disable_thinking : coupe le bloc <think>, néfaste au parsing JSON et à la latence.
            extra_body = (
                {"chat_template_kwargs": {"enable_thinking": False}}
                if self.model_config.disable_thinking
                else {}
            )
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                temperature=temp,
                max_tokens=self.model_config.max_tokens,
                messages=messages,
                extra_body=extra_body or None,
                **response_format_kwargs,
                **trace_kwargs,
            )
            content = response.choices[0].message.content or "{}"
            prediction = self._parse_response(copy, content)
            prediction.n_attempts = attempt + 1
            if prediction.transcribed:
                return prediction
        return prediction  # type: ignore[return-value]

    def _parse_response(self, copy: Copy, content: str) -> CopyPrediction:
        """Parse la réponse JSON en prédictions par item, avec ré-alignement si décalage détecté.

        Args:
            copy: Copie évaluée (fournit la liste d'items attendus).
            content: Contenu textuel brut renvoyé par le modèle (JSON éventuellement balisé).

        Returns:
            Prédiction de la copie ; `transcribed=False` et items codés « ? » si la
            réponse est vide ou inexploitable.
        """
        try:
            cleaned = content.strip().removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
            data = json.loads(cleaned)
            raw_items = data.get("items", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            raw_items = []

        # Séquences dans l'ordre renvoyé par le modèle (avant ré-alignement éventuel).
        codes_seq = [str(it.get("code", "?")).strip() for it in raw_items]
        trans_seq = [it.get("transcription") for it in raw_items]
        conf_seq = [it.get("confidence") for it in raw_items]

        # Aucune réponse exploitable : copie non transcrite (à réessayer puis exclure).
        n_trans_utiles = sum(1 for t in trans_seq if t and str(t).strip())
        if not raw_items or n_trans_utiles == 0:
            items_vides = [
                ItemPrediction(item_id=i, code="?", confidence=0.0) for i in copy.item_ids
            ]
            return CopyPrediction(copy_id=copy.copy_id, items=items_vides, transcribed=False)

        expected_words = [
            self._items_by_id[i].attendu for i in copy.item_ids if i in self._items_by_id
        ]

        # Filet de sécurité : ré-aligner si un décalage est détecté.
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
