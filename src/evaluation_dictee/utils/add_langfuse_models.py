"""Enregistre dans Langfuse un prix par token THÉORIQUE (GPU amorti) pour les modèles llm.lab.

Modèles auto-hébergés sans tarif commercial, afin que Langfuse valorise chaque trace.
Toutes les constantes de coût ci-dessous sont des ordres de grandeur à AJUSTER
après mesure du débit réel. Montants raisonnés en euros (l'UI Langfuse affiche « $ »).

Usage :
    uv run --env-file .env add-langfuse-models
"""

from __future__ import annotations

import os
import re

import httpx

from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

# ── Hypothèses de coût GPU amorti (À AJUSTER) ────────────────────────────────
COUT_GPU_EUR_H = 2.5  # coût horaire d'une H100 (référence cloud public)
RATIO_ENTREE_SORTIE = 0.25  # un token d'entrée coûte ~1/4 d'un token de sortie

# Débit de SORTIE agrégé sous charge (tokens/s), par modèle. À mesurer sur llm.lab.
DEBIT_SORTIE_TOK_S: dict[str, float] = {
    "gemma4-26b-moe": 2500,  # MoE ~26B total
    "qwen3.6-35b-moe": 4000,  # MoE 35B total / ~3B actifs → plus rapide
}


def _prix_par_token(debit_tok_s: float) -> tuple[float, float]:
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


def _auth_and_base() -> tuple[tuple[str, str], str]:
    """Récupère (auth basic, base_url) depuis l'environnement.

    Returns:
        Le couple ((clé publique, clé secrète), base_url sans slash final), lu
        depuis `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` et `LANGFUSE_BASE_URL`.

    Raises:
        RuntimeError: Si l'une des trois variables d'environnement est absente.
    """
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    base = os.environ.get("LANGFUSE_BASE_URL")
    if not (public and secret and base):
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY et LANGFUSE_BASE_URL doivent "
            "être définis. Lance : uv run --env-file .env add-langfuse-models"
        )
    return (public, secret), base.rstrip("/")


def push_models() -> None:
    """Enregistre une définition de prix Langfuse pour chaque modèle.

    Relancer le script crée une nouvelle définition (la plus récente s'applique) :
    ajuster les constantes puis relancer suffit à mettre à jour les prix.
    """
    auth, base = _auth_and_base()
    url = f"{base}/api/public/models"

    for model_name, debit in DEBIT_SORTIE_TOK_S.items():
        prix_entree, prix_sortie = _prix_par_token(debit)
        payload = {
            "modelName": model_name,
            # Ancrage strict sur le nom exact (échappé), insensible à la casse.
            "matchPattern": f"(?i)^{re.escape(model_name)}$",
            "unit": "TOKENS",
            "inputPrice": prix_entree,
            "outputPrice": prix_sortie,
        }
        resp = httpx.post(url, auth=auth, json=payload, timeout=30)
        resp.raise_for_status()
        logger.info(
            "Modèle « %s » enregistré : entrée %.3g €/1M, sortie %.3g €/1M",
            model_name,
            prix_entree * 1e6,
            prix_sortie * 1e6,
        )


if __name__ == "__main__":
    push_models()
