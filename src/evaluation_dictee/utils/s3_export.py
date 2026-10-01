"""Export des prédictions vers S3 (répertoire `predictions/` du bucket projet).

Le pipeline écrit les prédictions en local — `data/processed/<name>_predictions.jsonl` —
en mode append + fsync par copie, pour la reprise sur crash. Ce mode est propre au
disque local : S3 ne supporte ni l'append ni le fsync. On sépare donc l'écriture
(locale, incrémentale) de l'export (S3, une fois le run terminé).

But : pouvoir relancer les notebooks et le site Quarto sans réexécuter le pipeline.
Rien n'est jamais commité dans Git — les transcriptions sont des données d'élèves
mineurs, hébergées sur le SSP Cloud.

L'accès S3 suit le même pattern que `data/loaders.py` : `fsspec` lit les identifiants
et l'endpoint MinIO depuis l'environnement (variables injectées par Onyxia).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import fsspec
import yaml

from evaluation_dictee.config import load_config, run_output_name
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

# Suffixes des fichiers produits par les deux pipelines. Le scoring (end_to_end ET
# two_stage) partage un seul format : c'est le `name` du run qui distingue les runs.
SCORING_SUFFIX = "_predictions.jsonl"
HTR_SUFFIX = "_htr_predictions.jsonl"
# Liste des copies écartées des métriques (vierges/illisibles, décision D8), à
# vérifier à l'œil. Propre au scoring : le pipeline HTR ne produit pas ce fichier.
COPIES_ECARTEES_SUFFIX = "_copies_ecartees.csv"

# Sous-dossier des runs D'EXPÉRIMENTATION (bras testés sur un échantillon partiel :
# chain-of-thought, comptage, exemples de fautes...). Le site (`website/_analyse.py`)
# liste `predictions/` sans y descendre : un fichier posé ici n'apparaît donc JAMAIS
# comme un modèle, par construction, sans aucune liste de noms à maintenir. Un run
# n'alimente le site que lorsqu'il est mené sur le corpus complet (`data.limit: null`)
# et exporté dans `predictions/` — c'est déjà la règle qu'applique
# `launchers/launch_eval.sh`, qui désactive son export si `--limit` est utilisé.
EXPERIMENTATIONS_SUBDIR = "experimentations"


def resolve_run_name(config_path: str | Path, htr: bool = False) -> str:
    """Résout le préfixe des fichiers de sortie d'un run à partir de son YAML.

    Pour le scoring, ce préfixe inclut le(s) nom(s) de modèle (`run_output_name`) :
    se contenter du champ `name` désignerait un fichier inexistant, puisque le
    benchmark suffixe ses sorties par le modèle. Le pipeline HTR, lui, nomme
    encore ses sorties d'après le seul champ `name`.

    Args:
        config_path: Chemin du YAML du run.
        htr: True pour un run HTR (nommage par `name` seul).

    Returns:
        Le préfixe des fichiers de sortie du run.
    """
    if htr:
        with open(config_path, encoding="utf-8") as f:
            return str(yaml.safe_load(f)["name"])
    return run_output_name(load_config(config_path))


def _join_s3(prefix: str, name: str) -> str:
    """Concatène un préfixe (S3 ou local) et un nom de fichier (un seul slash)."""
    return prefix.rstrip("/") + "/" + name


def resolve_dest_prefix(base_prefix: str | Path, experimentation: bool) -> str:
    """Choisit `predictions/` ou `predictions/experimentations/` selon la nature du run.

    Args:
        base_prefix: préfixe S3 des runs de référence (ex. `S3_PREDICTIONS_PREFIX`).
        experimentation: True pour un bras testé sur un échantillon partiel.

    Returns:
        Le préfixe de destination effectif.
    """
    base = str(base_prefix).rstrip("/")
    return f"{base}/{EXPERIMENTATIONS_SUBDIR}" if experimentation else base


def upload_predictions(local_path: str | Path, dest_prefix: str | Path) -> str:
    """Copie un fichier de prédictions local vers un préfixe S3.

    Args:
        local_path: Chemin local du fichier JSONL de prédictions.
        dest_prefix: Préfixe de destination (ex. s3://bucket/predictions).

    Returns:
        L'URI de destination du fichier écrit.

    Raises:
        FileNotFoundError: Si le fichier local est absent (run non lancé).
    """
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(
            f"Fichier de prédictions introuvable : {local_path}. "
            "Lancer d'abord le benchmark (scripts/run_benchmark.py) pour le produire."
        )

    dest = _join_s3(str(dest_prefix), local_path.name)
    # Copie binaire par flux (pas de réencodage, robuste aux gros fichiers).
    with open(local_path, "rb") as src, fsspec.open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)

    logger.info("Prédictions exportées : %s → %s", local_path, dest)
    return dest


def export_run(
    run_name: str,
    dest_prefix: str | Path,
    source_dir: str | Path = "data/processed",
    htr: bool = False,
) -> str:
    """Exporte vers S3 le fichier de prédictions d'un run donné, et sa liste d'écartées.

    La liste des copies écartées (vierges/illisibles, décision D8) est exportée à côté
    du fichier de prédictions quand elle existe — c'est-à-dire quand le run en a
    effectivement écarté au moins une. Elle doit être vérifiée à l'œil (une copie peut
    être déclarée vierge à tort) ; ne pas l'exporter la laisserait bloquée sur la
    machine qui a produit le run, invérifiable par quiconque d'autre.

    Args:
        run_name: Nom du run (préfixe des fichiers de sortie).
        dest_prefix: Préfixe S3 de destination.
        source_dir: Dossier local des prédictions. [défaut : data/processed]
        htr: Si True, exporte `<name>_htr_predictions.jsonl` (transcription seule)
            au lieu du fichier de scoring ; la liste des écartées n'est alors pas
            concernée, le pipeline HTR ne la produit pas.

    Returns:
        L'URI S3 du fichier de prédictions écrit.
    """
    suffix = HTR_SUFFIX if htr else SCORING_SUFFIX
    local_path = Path(source_dir) / f"{run_name}{suffix}"
    dest = upload_predictions(local_path, dest_prefix)

    if not htr:
        ecartees = Path(source_dir) / f"{run_name}{COPIES_ECARTEES_SUFFIX}"
        if ecartees.is_file():
            dest_ecartees = _join_s3(str(dest_prefix), ecartees.name)
            with open(ecartees, "rb") as src, fsspec.open(dest_ecartees, "wb") as dst:
                shutil.copyfileobj(src, dst)
            logger.info("Copies écartées exportées : %s → %s", ecartees, dest_ecartees)
        else:
            logger.info(
                "Aucune copie écartée pour %s (fichier %s absent) : rien à exporter "
                "en plus des prédictions.",
                run_name,
                ecartees.name,
            )

    return dest
