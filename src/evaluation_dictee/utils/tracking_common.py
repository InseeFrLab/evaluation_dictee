"""Éléments communs aux backends de suivi (Langfuse, MLflow) : metadata, tags, session, user.

Ces fonctions décrivent un run de la même façon quel que soit l'outil, pour que les deux
reçoivent exactement les mêmes informations.
"""

from __future__ import annotations

import os
import time

from evaluation_dictee.config import ExperimentConfig


def _run_metadata(config: ExperimentConfig) -> dict[str, str]:
    """Paramètres de l'expérience en chaînes (MLflow : tags et params sont des chaînes).

    Args:
        config: Configuration de l'expérience à décrire.

    Returns:
        Dictionnaire {modèle, modèle étape 2, approche, méthode, schéma, n_few_shot,
        corpus, raisonnement, sortie contrainte} en chaînes. `model_stage2` vaut "-"
        en end_to_end.
    """
    return {
        "model": config.model.name,
        # Sans ce champ, deux runs two_stage ne différant que par le modèle de codage
        # sont indiscernables dans MLflow.
        "model_stage2": config.model_stage2.name if config.model_stage2 is not None else "-",
        "approach": config.approach,
        "method": config.prompt.method,
        "scheme": config.grid.scheme,
        "n_few_shot": str(config.prompt.n_few_shot),
        "corpus": config.data.corpus,
        # Sans ces champs, deux runs qui ne diffèrent que par la façon de faire
        # raisonner le modèle sont indiscernables dans MLflow — or c'est exactement
        # la comparaison que l'on cherche à faire. Chaque option de prompt y figure,
        # pour qu'un run reste interprétable sans son YAML sous les yeux.
        "thinking": "off" if config.model.disable_thinking else "on",
        "chain_of_thought": str(config.prompt.chain_of_thought).lower(),
        "count_items": str(config.prompt.count_items).lower(),
        "enforce_count": str(config.prompt.enforce_count).lower(),
        "check_neighbours": str(config.prompt.check_neighbours).lower(),
        "show_error_examples": str(config.prompt.show_error_examples).lower(),
        "structured_output": str(config.model.structured_output).lower(),
        "max_tokens": str(config.model.max_tokens),
    }


def _run_tags(config: ExperimentConfig) -> list[str]:
    """Tags de filtrage/regroupement (jointure en tag MLflow `tags`, cf. `_trace_tags`).

    Args:
        config: Configuration de l'expérience.

    Returns:
        Liste de tags (méthode, corpus, `scheme:…`, `model:…`, `thinking:…`,
        éventuellement `cot`, `model_stage2:…`, `run:<nom>`).
    """
    tags = [
        config.prompt.method,
        config.data.corpus,
        f"scheme:{config.grid.scheme}",
        f"model:{config.model.name}",
        f"thinking:{'off' if config.model.disable_thinking else 'on'}",
    ]
    # Un tag par option active : c'est ce qui permet de retrouver « tous les runs
    # avec exemples de fautes » d'un clic dans l'UI, sans lire les métadonnées.
    for actif, tag in [
        (config.prompt.chain_of_thought, "cot"),
        (config.prompt.count_items, "comptage"),
        (config.prompt.enforce_count, "coherence-comptage"),
        (config.prompt.check_neighbours, "voisinage"),
        (config.prompt.show_error_examples, "exemples-fautes"),
    ]:
        if actif:
            tags.append(tag)
    # Tag distinct seulement si l'étape 2 utilise un autre modèle : évite un tag
    # redondant sur les runs two_stage mono-modèle, tout en rendant filtrables les
    # runs qui croisent deux modèles.
    if config.model_stage2 is not None and config.model_stage2.name != config.model.name:
        tags.append(f"model_stage2:{config.model_stage2.name}")
    # session_id unique par lancement : ce tag regroupe les lancements d'une config
    tags.append(f"run:{config.name}")
    return tags


def _trace_tags(config: ExperimentConfig) -> dict[str, str]:
    """Tags MLflow d'une trace : metadata du run + liste de tags jointe (pas de tag-liste).

    Args:
        config: Configuration de l'expérience.

    Returns:
        Dictionnaire de chaînes ; MLflow filtre par clé/valeur (`tags.method = 'C'`),
        pas par appartenance à une liste comme les tags Langfuse.
    """
    return {**_run_metadata(config), "run": config.name, "tags": ",".join(_run_tags(config))}


def _user_id() -> str | None:
    """Identifiant utilisateur = namespace Kubernetes (KUBERNETES_NAMESPACE), ou None.

    Returns:
        La valeur de `KUBERNETES_NAMESPACE`, ou None si absente ou vide.
    """
    return os.environ.get("KUBERNETES_NAMESPACE") or None


def _launch_session_id(config: ExperimentConfig) -> str:
    """Session UNIQUE PAR LANCEMENT : « <nom>-AAAAMMJJ-HHMMSS » (triable dans le temps).

    Args:
        config: Configuration de l'expérience (fournit `config.name`).

    Returns:
        L'identifiant de session, nom du run suffixé de l'horodatage courant.
    """
    return f"{config.name}-{time.strftime('%Y%m%d-%H%M%S')}"
