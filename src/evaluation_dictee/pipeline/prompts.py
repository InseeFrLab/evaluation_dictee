"""Construction des prompts d'évaluation (décisions méthodologiques CLAUDE.md §3-4).

Point critique : les élisions (n', S', l', d', qu') sont des items SÉPARÉS du mot suivant ;
sans consigne explicite le modèle les recolle et décale tous les items suivants.
Chaque prompt est un template chat versionné dans Langfuse, avec repli local si indisponible.
"""

from __future__ import annotations

from typing import cast

from langfuse import get_client
from langfuse.model import ChatMessageDict, ChatPromptClient

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.grid import GridItem

# Noms sous lesquels les prompts sont versionnés dans Langfuse.
PROMPT_DICTATION = "Dictation"
PROMPT_TRANSCRIPTION = "Transcription"
PROMPT_TEXT_CODING = "Text coding"

_GRILLE_SIMPLIFIEE = (
    "Pour chaque item, attribue un code :\n"
    '- "1" : le mot (ou la ponctuation) attendu est présent et correctement orthographié ;\n'
    '- "9" : le mot est présent mais comporte une erreur (orthographe, accord, '
    "conjugaison, accent, ponctuation erronée...) ;\n"
    "- \"0\" : le mot attendu est absent (l'élève ne l'a pas écrit)."
)

_GRILLE_COMPLETE = (
    "Pour chaque item, attribue un code.\n"
    "Pour un MOT :\n"
    '- "1" : correctement orthographié ;\n'
    '- "3" : erreur LEXICALE (n\'altère pas la prononciation : accent, lettre muette, '
    "mauvais graphème — ex. « soire » pour « soir ») ;\n"
    '- "4" : erreur GRAMMATICALE (accord, conjugaison, confusion de catégorie comme '
    "« on »/« ont », ou toute erreur qui change la prononciation) ;\n"
    '- "5" : erreur À LA FOIS lexicale ET grammaticale ;\n'
    "- \"0\" : le mot est absent (l'élève ne l'a pas écrit).\n"
    'Pour la PONCTUATION : "1" correcte, "9" erronée, "0" absente.'
)

_CONSIGNE_ALIGNEMENT = (
    "2 - RÈGLE D'ALIGNEMENT (la plus importante) : la liste de règles ci-dessous définit des items "
    "FIXES, un par ligne, dans l'ordre du texte. Tu dois rendre EXACTEMENT un code par "
    "item, dans le même ordre, sans en fusionner ni en omettre:\n"
    "a - Aligne-toi sur le MOT ATTENDU de chaque item, JAMAIS sur ta propre découpe de "
    "l'écriture de l'élève. L'élève peut écrire un mot avec un espace au milieu "
    "(« re trouver » au lieu de « retrouver ») ou coller deux mots (« nousles » au "
    "lieu de « nous les ») : ne te laisse pas décaler. Dans ces cas, rattache ce que "
    "tu lis au mot attendu correspondant, et continue d'aligner les items suivants "
    "sur leurs mots attendus respectifs.\n"
    "b - Les apostrophes d'élision sont des items SÉPARÉS du mot qui suit. Par exemple "
    "« n'étaient » se code en DEUX items distincts : « n' » puis « étaient ». "
    "De même « S'ils » = « S' » puis « ils » ; « l'olivier » = « l' » puis « olivier ».\n"
    "c - AVANT de coder, vérifie pour chaque item que la transcription que tu donnes "
    "correspond bien au mot attendu de CE numéro d'item ; si tu remarques un décalage, "
    "recale-toi immédiatement sur le mot attendu. Le nombre d'items de ta réponse doit "
    "être EXACTEMENT celui demandé.\n"
)

_CONSIGNE_FIDELITE = (
    "3 - Transcris EXACTEMENT ce que l'élève a écrit pour cet item, fautes comprises. "
    "Ne corrige jamais silencieusement l'orthographe : une faute non transcrite "
    "fausse l'évaluation. Si le mot écrit diffère du mot attendu, c'est une erreur (9).\n"
)

_CONSIGNE_COMPARAISON = (
    "4 - MÉTHODE DE CODAGE (à appliquer pour chaque item) : compare LETTRE À LETTRE ta "
    "transcription au mot attendu. Le code est 1 SEULEMENT si les deux sont rigoureusement "
    "identiques (mêmes lettres, mêmes accents, même terminaison). La MOINDRE différence, "
    "une lettre, un accent, une terminaison de conjugaison (ex. « mis » au lieu de "
    "« mit ») ou un singulier/pluriel, impose le code 9, même si le mot reste lisible et "
    "plausible. Ne te fie pas au sens : « mis » et « mit » se prononcent pareil mais "
    "l'un est faux. Code d'après la forme écrite exacte, pas d'après ce que l'élève "
    "voulait dire.\n"
)

_CONSIGNE_RATURES_HALLUCINATION = (
    "5 - Quand un passage est raturé/barré : ignore complètement le texte barré et lis "
    "uniquement ce que l'élève a retenu en version finale, DANS L'ORDRE OÙ C'EST ÉCRIT "
    "sur la copie. N'invente pas, ne réordonne pas les mots pour qu'ils collent au texte "
    "attendu : si l'élève a écrit les mots dans un certain ordre, transcris cet ordre réel.\n"
)

_CONSIGNE_RATURES = (
    "6 - Si l'élève a raturé puis réécrit un mot, lis uniquement la version FINALE "
    "(corrigée par l'élève), pas la version barrée.\n"
)

_CONSIGNE_CONFIANCE = (
    "7 - Pour chaque item, fournis un score de confiance entre 0 et 1 reflétant ta "
    "certitude (lisibilité, ambiguïté). Un score bas déclenchera une relecture humaine.\n"
)

_CONSIGNE_COT = (
    "8 - AVANT de choisir le code, écris un champ « comparaison » qui décrit "
    "explicitement en quoi la transcription diffère du mot attendu (ou "
    "précise « identique » si elles correspondent lettre à lettre). "
    "Exemple : attendu « inquiets » lu « inquiet » → « il manque le 's' final ».\n"
)

# Format de sortie JSON de la méthode C, avec ou sans champ « comparaison » (CoT).
_FORMAT_ITEMS_COT = (
    "Réponds UNIQUEMENT par un objet JSON, sans texte autour ni de notes, de la forme :\n"
    '{"items": [{"item_id": "...", "transcription": "ce que l\'élève a écrit", '
    '"comparaison": "identique" OU description brève de la différence, '
    '"code": "1", "confidence": 0.95, "reason": "les raisons du choix"}, ...]}.\n'
    "Tout ajout de texte hors de la structure du JSON sera pris comme une erreur par le pipeline."
)
_FORMAT_ITEMS_SIMPLE = (
    "Réponds UNIQUEMENT par un objet JSON, sans texte autour ni de notes, de la forme :\n"
    '{"items": [{"item_id": "...", "transcription": "ce que l\'élève a écrit", '
    '"code": "1", "confidence": 0.95, "reason": "les raisons du choix"}, ...]}.\n'
    "Tout ajout de texte hors de la structure du JSON sera pris comme une erreur par le pipeline."
)

# Consigne « ratures » propre à l'étape de transcription (formulation dédiée).
_CONSIGNE_RATURES_TRANSCRIPTION = (
    "Si l'élève a raturé puis réécrit, transcris uniquement la version "
    "FINALE (non barrée). Ignore complètement le texte barré.\n"
)

# Consigne « ratures » propre à l'étape d'évaluation directe
_CONSIGNE_RATURES_DICTATION = (
    "Si l'élève a raturé puis réécrit, ignore complètement le texte barré dans ton output."
    "Il s'agit d'une correction faite par l'élève, les mots raturés ne doivent pas être pris "
    "en compte dans la correction."
)


# ─────────────────────────────────────────────────────────────────────────────
# Templates chat (masques versionnés dans Langfuse, {{...}} remplis par les build_*).
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE_DICTATION: list[ChatMessageDict] = [
    {
        "role": "system",
        "content": (
            "Tu fais partie d'un groupe d'expert composé de professeurs et de formateurs pour une "
            "évaluation nationale de dictée.\n"
            "Ta tâche : On te montre l'image manuscrite de la dictée d'un élève de primaire. "
            "Tu dois noter chaque item de la dictée à l'aide des éléments définis "
            "dans la grille de notation.\n\n"
            "Règles à respecter impérativement :\n"
            "1 - Tu dois noter chaque item de la dictée avec la grille de notation disponible : "
            "{{grille}}\n"
            + _CONSIGNE_ALIGNEMENT
            + "{{consignes_optionnelles}}\n\n"
            + _CONSIGNE_CONFIANCE
            + "{{consigne_cot}}\n\n"
            + _CONSIGNE_RATURES_DICTATION
        ),
    },
    {
        "role": "user",
        "content": (
            "# Texte de référence (ce que l'élève devait écrire) :\n"
            "« {{reference_text}} »\n\n"
            "# Items à coder, dans l'ordre. Chaque ligne = un item fixe "
            "« identifiant → mot attendu » :\n"
            "{{items_list}}\n\n"
            "# Tu dois rendre EXACTEMENT {{n_items}} items, dans cet ordre.\n"
            "{{format_sortie}}"
        ),
    },
]

_TEMPLATE_TRANSCRIPTION: list[ChatMessageDict] = [
    {
        "role": "system",
        "content": (
            "Tu es un expert en lecture d'écriture manuscrite d'enfants.\n"
            "On te montre l'image manuscrite de la dictée d'un élève de primaire.\n\n"
            "Règles à respecter impérativement :\n"
            "1 - Transcris EXACTEMENT le texte écrit par l'élève, mot pour mot, "
            "FAUTES D'ORTHOGRAPHE COMPRISES.\n "
            "2 - Ne corrige rien, ne complète rien, "
            "ne réordonne rien.\n "
            " 3 - Reproduis fidèlement les erreurs, y compris les "
            "accents manquants, les mots mal orthographiés et la ponctuation.\n"
            "4 - Respecte l'ordre exact des items écrits sur la copie "
            "(mots, ponctuation, chiffres, ...).\n"
            "{{consigne_ratures}}"
        ),
    },
    {
        "role": "user",
        "content": (
            "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :\n"
            '{"transcription": "le texte exact écrit par l\'élève"}'
        ),
    },
]

_TEMPLATE_TEXT_CODING: list[ChatMessageDict] = [
    {
        "role": "system",
        "content": (
            "Tu fais partie d'un groupe d'expert, composé de professeur et de formateurs "
            "pour une évaluation nationale de dictée.\n"
            "Ta tâche : Tu ne vois PAS l'image, on te donne uniquement la transcription "
            "de ce que l'élève a écrit (produite par un système de lecture), et le "
            "texte de référence.\n"
            "Compare la transcription au mot attendu de chaque item. Si un mot attendu "
            "n'apparaît pas dans la transcription, code-le 0 (absent).\n\n"
            "Règles à respecter impérativement :\n"
            "1 - Tu dois noter chaque item de la dictée avec la grille de notation "
            "disponible ci-dessous : "
            "{{grille}}\n\n" + _CONSIGNE_ALIGNEMENT + "\n\n" + _CONSIGNE_COMPARAISON + "\n\n"
        ),
    },
    {
        "role": "user",
        "content": (
            "Texte de référence (ce que l'élève devait écrire) :\n"
            "« {{reference_text}} »\n\n"
            "Transcription de la copie de l'élève (fautes comprises) :\n"
            "« {{transcription}} »\n\n"
            "Items à coder, dans l'ordre. Chaque ligne = un item "
            "« identifiant → mot attendu » :\n"
            "{{items_list}}\n\n"
            "Tu dois rendre EXACTEMENT {{n_items}} items, dans cet ordre.\n"
            "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :\n"
            '{"items": [{"item_id": "...", "transcription": "mot lu pour cet item", '
            '"code": "1", "confidence": 0.95}, ...]}'
        ),
    },
]

# Registre exposé pour l'initialisation Langfuse (utils/add_langfuse_prompt.py).
PROMPT_TEMPLATES: dict[str, list[ChatMessageDict]] = {
    PROMPT_DICTATION: _TEMPLATE_DICTATION,
    PROMPT_TRANSCRIPTION: _TEMPLATE_TRANSCRIPTION,
    PROMPT_TEXT_CODING: _TEMPLATE_TEXT_CODING,
}


def _format_items(items: list[GridItem]) -> str:
    """Formate la liste des items « N. identifiant -> « mot » (nature) », un par ligne.

    Args:
        items: items de la grille à formater.

    Returns:
        Le texte des items numérotés, une ligne par item.
    """
    lignes = []
    for idx, it in enumerate(items, 1):
        nature = "ponctuation" if it.type == "ponctuation" else "mot"
        lignes.append(f"  {idx:>2}. {it.item_id} → « {it.attendu} » ({nature})")
    return "\n".join(lignes)


def _render_local(
    messages: list[ChatMessageDict], variables: dict[str, object]
) -> list[ChatMessageDict]:
    """Compile un template localement (repli si Langfuse est indisponible).

    Args:
        messages: messages du template, avec placeholders `{{clé}}`.
        variables: valeurs à substituer aux placeholders.

    Returns:
        Les messages avec les placeholders remplacés par leurs valeurs.
    """
    rendered: list[ChatMessageDict] = []
    for message in messages:
        content = message["content"]
        for key, value in variables.items():
            content = content.replace("{{" + key + "}}", str(value))
        rendered.append({"role": message["role"], "content": content})
    return rendered


def _compile_prompt(
    name: str,
    fallback: list[ChatMessageDict],
    variables: dict[str, object],
) -> list[ChatMessageDict]:
    """Récupère et compile un prompt chat Langfuse (repli sur `fallback` local si indisponible).

    Args:
        name: nom du prompt versionné dans Langfuse.
        fallback: template local utilisé si Langfuse est indisponible.
        variables: valeurs à injecter dans le template.

    Returns:
        Les messages compilés (role/content uniquement, placeholders vides écartés).
    """
    try:
        prompt = get_client().get_prompt(name, type="chat", fallback=fallback)
        messages = prompt.compile(**variables)
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        messages = _render_local(fallback, variables)
    # compile() peut renvoyer des placeholders sans "content" ; on ne garde que role/content.
    plain = cast("list[dict[str, object]]", messages)
    compiled: list[ChatMessageDict] = []
    for message in plain:
        if message.get("content"):
            compiled.append({"role": str(message["role"]), "content": str(message["content"])})
    return compiled


def attach_image(messages: list[ChatMessageDict], image_data_url: str) -> list[dict[str, object]]:
    """Attache une image au dernier message user, en contenu structuré (format multimodal vLLM).

    Args:
        messages: messages du prompt ; le dernier doit être le message user.
        image_data_url: image encodée en data URL à joindre.

    Returns:
        Les messages avec l'image ajoutée au contenu du dernier message user.
    """
    *head, user = messages
    out: list[dict[str, object]] = [dict(message) for message in head]
    out.append(
        {
            "role": user["role"],
            "content": [
                {"type": "text", "text": user["content"]},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    )
    return out


def fetch_prompt(name: str) -> ChatPromptClient | None:
    """Renvoie l'objet prompt Langfuse pour lier une trace à sa version (None si repli local).

    Args:
        name: nom du prompt versionné dans Langfuse.

    Returns:
        L'objet prompt Langfuse, ou None si Langfuse est indisponible ou renvoie le repli.
    """
    try:
        prompt = get_client().get_prompt(name, type="chat", fallback=PROMPT_TEMPLATES[name])
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        return None
    return None if prompt.is_fallback else prompt


def build_dictation_prompt(
    reference_text: str,
    items: list[GridItem],
    config: PromptConfig,
    scheme: str = "simplifiee",
) -> list[ChatMessageDict]:
    """Construit le prompt d'évaluation d'une dictée (méthode C) ; image jointe par l'appelant.

    Args:
        reference_text: texte de référence de la dictée.
        items: items de la grille à coder.
        config: options de prompt (fidélité, ratures, chain-of-thought).
        scheme: schéma de grille, "complete" ou "simplifiee" (défaut).

    Returns:
        Les messages du prompt compilés, prêts à recevoir l'image.
    """
    grille = _GRILLE_COMPLETE if scheme == "complete" else _GRILLE_SIMPLIFIEE

    blocs: list[str] = []
    if config.enforce_faithful:
        blocs += [_CONSIGNE_FIDELITE, _CONSIGNE_COMPARAISON]
    if config.read_final_state:
        blocs += [_CONSIGNE_RATURES, _CONSIGNE_RATURES_HALLUCINATION]
    consignes_optionnelles = ("\n\n" + "\n\n".join(blocs)) if blocs else ""

    if config.chain_of_thought:
        consigne_cot = "\n\n" + _CONSIGNE_COT
        format_sortie = _FORMAT_ITEMS_COT
    else:
        consigne_cot = ""
        format_sortie = _FORMAT_ITEMS_SIMPLE

    return _compile_prompt(
        PROMPT_DICTATION,
        fallback=_TEMPLATE_DICTATION,
        variables={
            "grille": grille,
            "consignes_optionnelles": consignes_optionnelles,
            "consigne_cot": consigne_cot,
            "reference_text": reference_text,
            "items_list": _format_items(items),
            "n_items": len(items),
            "format_sortie": format_sortie,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompts pour l'APPROCHE 1 (deux étapes) : transcription HTR puis codage textuel
# ─────────────────────────────────────────────────────────────────────────────


def build_transcription_prompt(read_final_state: bool = True) -> list[ChatMessageDict]:
    """Prompt ÉTAPE 1 (HTR) : transcrire l'image sans coder ni voir la référence (non biaisé).

    Args:
        read_final_state: si True, ajoute la consigne de ne lire que l'état final (ratures).

    Returns:
        Les messages du prompt de transcription compilés.
    """
    consigne_ratures = ("\n" + _CONSIGNE_RATURES_TRANSCRIPTION) if read_final_state else ""
    return _compile_prompt(
        PROMPT_TRANSCRIPTION,
        fallback=_TEMPLATE_TRANSCRIPTION,
        variables={"consigne_ratures": consigne_ratures},
    )


def build_text_coding_prompt(
    reference_text: str,
    transcription: str,
    items: list[GridItem],
    scheme: str = "simplifiee",
) -> list[ChatMessageDict]:
    """Prompt ÉTAPE 2 : coder à partir du texte transcrit (sans image, modèle texte seul).

    Args:
        reference_text: texte de référence de la dictée.
        transcription: transcription de la copie produite à l'étape 1.
        items: items de la grille à coder.
        scheme: schéma de grille, "complete" ou "simplifiee" (défaut).

    Returns:
        Les messages du prompt de codage textuel compilés.
    """
    grille = _GRILLE_COMPLETE if scheme == "complete" else _GRILLE_SIMPLIFIEE
    return _compile_prompt(
        PROMPT_TEXT_CODING,
        fallback=_TEMPLATE_TEXT_CODING,
        variables={
            "grille": grille,
            "reference_text": reference_text,
            "transcription": transcription,
            "items_list": _format_items(items),
            "n_items": len(items),
        },
    )
