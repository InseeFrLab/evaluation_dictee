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
from evaluation_dictee.models.base import CODE_NON_PARSE, CopyPrediction, ItemPrediction, Scorer
from evaluation_dictee.pipeline.alignment import best_realignment, needs_realignment
from evaluation_dictee.pipeline.prompts import (
    PROMPT_DICTATION,
    attach_image,
    build_dictation_prompt,
    fetch_prompt,
)
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


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


def thinking_kwargs(model_config: ModelConfig) -> dict[str, Any]:
    """`chat_template_kwargs` qui active ou coupe EXPLICITEMENT le raisonnement natif.

    La valeur est toujours transmise, dans les deux sens. Auparavant seule la coupure
    était envoyée : `disable_thinking: false` laissait le défaut du modèle décider, et
    ce défaut diffère d'un modèle à l'autre (thinking ON par défaut sur qwen3-6-35b-moe
    et qwen3-8-27b, OFF sur gemma4-26b-moe). Deux configs identiques ne décrivaient donc
    pas le même run selon le modèle choisi.

    Args:
        model_config: Configuration du modèle de l'étape concernée.

    Returns:
        Le dictionnaire à passer en `extra_body`.
    """
    return {"chat_template_kwargs": {"enable_thinking": not model_config.disable_thinking}}


def extract_reasoning(message: Any) -> str | None:
    """Récupère le raisonnement natif (`reasoning_content`) d'une réponse, s'il y en a.

    L'endpoint llm.lab isole le bloc de raisonnement dans un champ distinct de
    `content` : le JSON reste donc conforme au schéma même en mode thinking, et le
    raisonnement est récupérable au lieu d'être perdu.

    Args:
        message: Message de la réponse (`response.choices[0].message`).

    Returns:
        Le raisonnement, ou None si le modèle n'en a pas produit.
    """
    brut = getattr(message, "reasoning_content", None)
    if brut is None:
        extra = getattr(message, "model_extra", None) or {}
        brut = extra.get("reasoning_content")
    texte = str(brut).strip() if brut else ""
    return texte or None


def log_if_truncated(response: Any, copy_id: str, max_tokens: int) -> bool:
    """Journalise une génération coupée par `max_tokens` (JSON tronqué = copie perdue).

    Sans cette alerte, une troncature systématique est indiscernable d'un modèle
    incapable de lire la copie : le run de juillet 2026 a passé 43 h à coder 802 copies
    en « non transcrite » pour cette raison. En mode thinking, le raisonnement compte
    dans le même budget que le JSON, ce qui rend le plafond bien plus facile à atteindre.

    Args:
        response: Réponse complète de l'API.
        copy_id: Copie concernée, pour le message.
        max_tokens: Plafond configuré.

    Returns:
        True si la génération a été tronquée.
    """
    # getattr défensif : un endpoint qui n'expose pas `finish_reason` ne doit pas faire
    # tomber un run de 30 h sur une simple ligne de diagnostic.
    if getattr(response.choices[0], "finish_reason", None) != "length":
        return False
    usage = getattr(response, "usage", None)
    logger.warning(
        "Génération TRONQUÉE sur %s : plafond max_tokens=%d atteint (%s tokens générés). "
        "Le JSON est incomplet, la copie sera perdue. Augmenter `model.max_tokens` "
        "(en mode thinking, le raisonnement consomme le même budget que la réponse).",
        copy_id,
        max_tokens,
        getattr(usage, "completion_tokens", "?"),
    )
    return True


def _items_json_schema(chain_of_thought: bool, count_items: bool = False) -> dict[str, Any]:
    """Schéma JSON de la réponse attendue (méthode C) pour le décodage contraint vLLM.

    Contraint la structure de chaque item, pas leur nombre (variable selon la copie).

    En mode chain-of-thought, `comparaison` est déclarée AVANT `code` : c'est l'ordre
    qu'on demande au modèle de suivre, puisque verbaliser la différence après avoir
    choisi le code ne raisonne rien. Attention, le décodage contraint ne GARANTIT pas
    cet ordre (mesuré le 11/09/2026 : l'ordre effectif varie d'un modèle et d'une
    requête à l'autre, tout en restant stable au sein d'une réponse) — d'où le contrôle
    a posteriori dans `_parse_response`.

    Args:
        chain_of_thought: Si True, ajoute le champ obligatoire `comparaison` à chaque item.
        count_items: Si True, exige un champ `n_items_lus` en tête de réponse.

    Returns:
        Schéma JSON de la réponse attendue (objet avec une liste `items`).
    """
    properties: dict[str, Any] = {
        "item_id": {"type": "string"},
        "transcription": {"type": "string"},
    }
    if chain_of_thought:
        properties["comparaison"] = {"type": "string"}
    properties["code"] = {"type": "string"}
    properties["confidence"] = {"type": "number"}
    required = list(properties)
    liste_items = {
        "items": {
            "type": "array",
            "items": {"type": "object", "properties": properties, "required": required},
        }
    }
    if not count_items:
        return {"type": "object", "properties": liste_items, "required": ["items"]}
    # `n_items_lus` est déclaré AVANT `items` : le comptage doit précéder le codage
    # pour le contraindre. Même réserve que pour `comparaison` — l'ordre effectif
    # n'est pas garanti par le décodage contraint, il est donc vérifié a posteriori.
    return {
        "type": "object",
        "properties": {"n_items_lus": {"type": "integer"}, **liste_items},
        "required": ["n_items_lus", "items"],
    }


def comparaison_avant_code(entry: dict[str, Any]) -> bool | None:
    """Le modèle a-t-il écrit « comparaison » avant « code » dans CET item ?

    `json.loads` conserve l'ordre du document : on peut donc vérifier a posteriori si
    la verbalisation a précédé la décision. Le décodage contraint n'impose pas cet
    ordre, et une comparaison écrite après le code n'est plus un raisonnement mais une
    justification — sans ce contrôle, un run « chain-of-thought » peut être purement
    décoratif sans que rien ne le signale.

    Args:
        entry: item tel que renvoyé par le modèle, ordre des clés préservé.

    Returns:
        True/False, ou None si l'item ne porte pas les deux clés (mode CoT inactif).
    """
    cles = list(entry)
    if "comparaison" not in cles or "code" not in cles:
        return None
    return cles.index("comparaison") < cles.index("code")


def _ordre_dominant(ordres: list[bool | None]) -> bool | None:
    """Ordre majoritaire des clés sur une réponse (None si aucun item ne l'indique).

    Args:
        ordres: par item, True si « comparaison » précède « code ».

    Returns:
        True/False selon l'ordre dominant, None si l'information manque.
    """
    renseignes = [o for o in ordres if o is not None]
    if not renseignes:
        return None
    return sum(renseignes) * 2 >= len(renseignes)


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
                    "schema": _items_json_schema(
                        self.prompt_config.chain_of_thought,
                        self.prompt_config.count_items,
                    ),
                },
            }

        # Retry avec température légèrement relevée pour sortir d'une réponse vide déterministe.
        prediction: CopyPrediction | None = None
        for attempt in range(self.model_config.max_retries + 1):
            temp = self.model_config.temperature + (0.3 if attempt > 0 else 0.0)
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                temperature=temp,
                max_tokens=self.model_config.max_tokens,
                messages=messages,
                extra_body=thinking_kwargs(self.model_config),
                **response_format_kwargs,
                **trace_kwargs,
            )
            message = response.choices[0].message
            log_if_truncated(response, copy.copy_id, self.model_config.max_tokens)
            content = message.content or "{}"
            prediction = self._parse_response(copy, content)
            prediction.n_attempts = attempt + 1
            # Raisonnement de l'essai courant : c'est celui qui a produit les codes rendus.
            prediction.reasoning = extract_reasoning(message)
            if prediction.transcribed:
                return prediction
        return prediction  # type: ignore[return-value]

    def _alerter_si_cot_decorative(self, copy: Copy, ordres: list[bool | None]) -> None:
        """Signale une chain-of-thought produite APRÈS le code, donc sans effet.

        Args:
            copy: Copie évaluée.
            ordres: par item, True si « comparaison » précède « code ».
        """
        renseignes = [o for o in ordres if o is not None]
        if not renseignes or sum(renseignes) * 2 >= len(renseignes):
            return
        logger.warning(
            "Chain-of-thought DÉCORATIVE sur %s : le modèle a écrit « comparaison » "
            "APRÈS « code » sur %d item(s) sur %d. Il justifie sa décision au lieu de "
            "raisonner avant de la prendre : aucun gain n'est attendu de l'option. Le "
            "décodage contraint n'impose pas l'ordre des clés — envisager "
            "`structured_output: false` pour ce run.",
            copy.copy_id,
            len(renseignes) - sum(renseignes),
            len(renseignes),
        )

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
            n_lus = data.get("n_items_lus")
            n_items_lus = int(n_lus) if isinstance(n_lus, int | float) else None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            raw_items = []
            n_items_lus = None

        # Séquences dans l'ordre renvoyé par le modèle (avant ré-alignement éventuel).
        codes_seq = [str(it.get("code", CODE_NON_PARSE)).strip() for it in raw_items]
        trans_seq = [it.get("transcription") for it in raw_items]
        conf_seq = [it.get("confidence") for it in raw_items]
        comp_seq = [it.get("comparaison") for it in raw_items]
        ordres = [comparaison_avant_code(it) for it in raw_items]
        self._alerter_si_cot_decorative(copy, ordres)

        # Aucune réponse exploitable : copie non transcrite (à réessayer puis exclure).
        n_trans_utiles = sum(1 for t in trans_seq if t and str(t).strip())
        if not raw_items or n_trans_utiles == 0:
            items_vides = [
                ItemPrediction(item_id=i, code=CODE_NON_PARSE, confidence=0.0)
                for i in copy.item_ids
            ]
            return CopyPrediction(copy_id=copy.copy_id, items=items_vides, transcribed=False)

        expected_words = [
            self._items_by_id[i].attendu for i in copy.item_ids if i in self._items_by_id
        ]

        # Filet de sécurité : ré-aligner si un décalage est détecté.
        if codes_seq and needs_realignment(expected_words, trans_seq):
            aligned = best_realignment(
                expected_words, codes_seq, trans_seq, conf_seq, comparaisons=comp_seq
            )
            # L'ordre des clés est une propriété de la RÉPONSE, pas d'un item en
            # particulier : après ré-alignement il vaut donc pour toute la copie.
            ordre_copie = _ordre_dominant(ordres)
            items = [
                ItemPrediction(
                    item_id=item_id,
                    code=a.code,
                    confidence=a.confidence,
                    transcription=a.transcription,
                    comparaison=a.comparaison,
                    comparaison_avant_code=ordre_copie,
                )
                for item_id, a in zip(copy.item_ids, aligned, strict=False)
            ]
            return CopyPrediction(copy_id=copy.copy_id, items=items, n_items_lus=n_items_lus)

        by_id = {it.get("item_id"): it for it in raw_items}
        items = []
        for item_id in copy.item_ids:
            entry = by_id.get(item_id)
            if entry is None:
                items.append(ItemPrediction(item_id=item_id, code=CODE_NON_PARSE, confidence=0.0))
            else:
                items.append(
                    ItemPrediction(
                        item_id=item_id,
                        code=str(entry.get("code", CODE_NON_PARSE)).strip(),
                        confidence=entry.get("confidence"),
                        transcription=entry.get("transcription"),
                        comparaison=entry.get("comparaison"),
                        comparaison_avant_code=comparaison_avant_code(entry),
                    )
                )
        return CopyPrediction(copy_id=copy.copy_id, items=items, n_items_lus=n_items_lus)
