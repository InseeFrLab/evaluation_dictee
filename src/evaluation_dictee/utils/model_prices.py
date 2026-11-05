"""Prix par token THÉORIQUES (GPU amorti) des modèles auto-hébergés sur llm.lab.

Source unique pour le suivi de coût : `add_langfuse_models` les enregistre côté Langfuse,
`tracking` les applique aux spans MLflow (qui n'a pas de table de prix personnalisée).
Toutes les constantes sont des ordres de grandeur à AJUSTER après mesure du débit réel.
Montants raisonnés en euros (les UI affichent « $ »).
"""

from __future__ import annotations

# ── Hypothèses de coût GPU amorti (À AJUSTER) ────────────────────────────────
COUT_GPU_EUR_H = 2.5  # coût horaire d'une H100 (référence cloud public)
RATIO_ENTREE_SORTIE = 0.25  # un token d'entrée coûte ~1/4 d'un token de sortie

# Débit de SORTIE agrégé sous charge (tokens/s), par modèle. À mesurer sur llm.lab.
DEBIT_SORTIE_TOK_S: dict[str, float] = {
    "gemma4-26b-moe": 2500,  # MoE ~26B total
    "qwen3-6-35b-moe": 4000,  # MoE 35B total / ~3B actifs → plus rapide
    # dense 27B (tous les paramètres actifs) → nettement plus lent ; ESTIMATION à mesurer
    "qwen3-8-27b": 1200,
}


def prix_par_token(debit_tok_s: float) -> tuple[float, float]:
    """Renvoie (prix_entrée, prix_sortie) en €/token pour un débit donné.

    Args:
        debit_tok_s: Débit de sortie agrégé du modèle, en tokens/seconde.

    Returns:
        Le couple (prix d'entrée, prix de sortie) en €/token, l'entrée valant
        `RATIO_ENTREE_SORTIE` fois la sortie.
    """
    # Entrée ≈ 1/4 de la sortie : le prefill est plus rapide que le decode.
    prix_sortie = COUT_GPU_EUR_H / (debit_tok_s * 3600)
    prix_entree = prix_sortie * RATIO_ENTREE_SORTIE
    return prix_entree, prix_sortie


def cout_appel(model: str, input_tokens: int, output_tokens: int) -> dict[str, float] | None:
    """Coût (€) d'un appel LLM, ou None si le modèle n'a pas de prix défini.

    Args:
        model: Nom du modèle, comparé sans tenir compte de la casse (comme le
            `matchPattern` ancré de Langfuse).
        input_tokens: Tokens d'entrée de l'appel.
        output_tokens: Tokens de sortie de l'appel.

    Returns:
        {input_cost, output_cost, total_cost}, ou None si le modèle est inconnu.
    """
    debit = {nom.lower(): d for nom, d in DEBIT_SORTIE_TOK_S.items()}.get(model.lower())
    if debit is None:
        return None
    prix_entree, prix_sortie = prix_par_token(debit)
    entree, sortie = input_tokens * prix_entree, output_tokens * prix_sortie
    return {"input_cost": entree, "output_cost": sortie, "total_cost": entree + sortie}
