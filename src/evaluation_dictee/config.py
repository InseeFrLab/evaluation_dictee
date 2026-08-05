"""Chargement et validation des configurations d'expérience (YAML validé par Pydantic)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Secrets lus depuis les variables d'environnement (ou le fichier .env)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_endpoint: str = "minio.lab.sspcloud.fr"
    s3_bucket: str = ""
    # Répertoire S3 où sont déposées les prédictions finies (JAMAIS dans Git : les
    # transcriptions sont des données d'élèves mineurs). Sert à relancer notebooks
    # et site Quarto sans réexécuter le pipeline. Surchargé par S3_PREDICTIONS_PREFIX.
    s3_predictions_prefix: str = "s3://projet-production-ecrits-depp/predictions"

    llm_base_url: str = "https://llm.lab.sspcloud.fr/api/v1"
    llm_api_key: str = "dummy"


class ModelConfig(BaseModel):
    """Paramètres du modèle à interroger."""

    name: str  # nom du modèle servi par vLLM (ex: "Qwen/Qwen2.5-VL-7B-Instruct")
    # "vlm" = multimodal image+texte ; "llm" = texte seul ; "htr" = HTR classique
    kind: Literal["vlm", "llm", "htr"] = "vlm"
    temperature: float = 0.0
    max_tokens: int = 2048
    request_logprobs: bool = True  # log-probs pour estimer la confiance par item
    # 0 = pas de retry. Copie sans transcription après tous les essais = exclue des métriques.
    max_retries: int = 2
    # Désactive le bloc <think> des modèles thinking (Qwen3, DeepSeek-R1, QwQ) qui casse
    # le parsing JSON et multiplie la latence. Sans effet sur gemma4, Qwen2.5-VL...
    disable_thinking: bool = True
    # Force une sortie JSON conforme au schéma (décodage contraint vLLM), supprimant les
    # copies non parsables. Mettre à False si l'endpoint ne supporte pas json_schema.
    structured_output: bool = True


class DataConfig(BaseModel):
    """Localisation et périmètre des données."""

    corpus: Literal["dictee", "production_ecrite"] = "dictee"
    images_path: str  # chemin local OU préfixe S3 (s3://.../dictee_2015/)
    # CSV des codes experts, faisant office de gold standard (annotateur unique). Local ou S3.
    labels_path: str
    # Grille JSON (versionnée dans configs/) : mot attendu + fautes connues par item.
    grid_path: str = "configs/grille_dictee_2015.json"
    limit: int | None = None  # limiter le nombre de copies (tests rapides)
    # Seuil de densité d'encre en dessous duquel une copie est jugée VIERGE (l'élève
    # n'a rien écrit : seul le pré-imprimé marque ~2%). Ces copies sont codées "0"
    # (absent) sur tous les items, sans appel modèle, identiquement pour toutes les
    # méthodes — évite que l'end-to-end hallucine la référence. 0 pour désactiver.
    blank_ink_threshold: float = 0.025


class GridConfig(BaseModel):
    """Schéma de codage cible."""

    # "simplifiee" = 1/erreur/0 (cible principale) ; "complete" = 1/3/4/5/9/0
    scheme: Literal["simplifiee", "complete"] = "simplifiee"


class PromptConfig(BaseModel):
    """Stratégie de prompting."""

    method: Literal["A", "B", "C", "D"] = "C"
    n_few_shot: int = 0  # nombre d'exemples annotés dans le prompt
    enforce_faithful: bool = True  # consigne anti-sur-correction
    read_final_state: bool = True  # règle des ratures : lire l'état final
    # Force un champ "comparaison" avant le code (verbalise la différence lue-attendue).
    chain_of_thought: bool = False


class ExperimentConfig(BaseModel):
    """Configuration complète d'une expérience (un fichier YAML = un run)."""

    name: str = Field(..., description="Nom unique du run, utilisé comme nom de trace Langfuse")
    seed: int = 42
    # Nombre de copies évaluées EN PARALLÈLE (le endpoint vLLM batche les requêtes
    # concurrentes). 1 = séquentiel (ancien comportement). Monter tant que le serveur
    # suit (débit borné par le GPU) ; redescendre en cas d'erreurs/timeout.
    concurrency: int = Field(default=8, ge=1)
    # "end_to_end" : un VLM lit l'image ET code en une passe.
    # "two_stage" : étape 1 HTR (transcription) puis étape 2 codage (isole lecture/jugement).
    approach: Literal["end_to_end", "two_stage"] = "end_to_end"
    model: ModelConfig
    # Modèle de l'étape 2 (codage textuel) en two_stage ; si absent, on réutilise `model`.
    model_stage2: ModelConfig | None = None
    data: DataConfig
    grid: GridConfig = GridConfig()
    prompt: PromptConfig = PromptConfig()


def load_config(path: str | Path) -> ExperimentConfig:
    """Charge et valide une configuration d'expérience depuis un fichier YAML.

    Args:
        path: Chemin du fichier YAML de configuration.

    Returns:
        La configuration d'expérience validée.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)


def _slugify_model_name(name: str) -> str:
    """Convertit un nom de modèle en fragment de nom de fichier sûr, sans perte.

    Ne remplace QUE les caractères réellement invalides dans un nom de fichier
    (`/`, espaces, `:`…) : `.` et `-` sont conservés tels quels et NE SONT PAS
    interchangés, pour que deux noms de modèle différents (ex. un nom fautif
    avec un point vs. le nom réel avec un tiret) ne produisent jamais le même
    suffixe — et donc n'écrasent jamais le même fichier de sortie.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-.")


def run_output_name(config: ExperimentConfig) -> str:
    """Nom de base des fichiers de sortie d'un run : `name` + modèle(s), sans doublon.

    Source unique de vérité pour nommer `<...>_predictions.jsonl` et
    `<...>_failed_copies.txt`. Le(s) modèle(s) figurent TOUJOURS dans le nom, que
    le run soit lancé via le YAML seul ou via les surcharges `--model-name` /
    `--model-stage2-name` : deux modèles n'écrasent donc jamais le même fichier.

    Idempotent : si `config.name` porte déjà le suffixe de modèle (cas d'un
    lancement CLI, où `override_model_names` a déjà renommé le run), il n'est pas
    ajouté une seconde fois. C'est ce qui garantit qu'un même run écrit dans le
    même fichier — et retrouve donc son checkpoint de reprise — quel que soit son
    mode de lancement.

    Args:
        config: Configuration de l'expérience.

    Returns:
        Le préfixe des fichiers de sortie du run.
    """
    parts = [_slugify_model_name(config.model.name)]
    if config.model_stage2 is not None:
        slug_stage2 = _slugify_model_name(config.model_stage2.name)
        if slug_stage2 not in parts:
            parts.append(slug_stage2)
    suffix = "_".join(parts)

    if config.name == suffix or config.name.endswith(f"_{suffix}"):
        return config.name
    return f"{config.name}_{suffix}"


def override_model_names(
    config: ExperimentConfig,
    model_name: str | None = None,
    model_stage2_name: str | None = None,
) -> ExperimentConfig:
    """Surcharge le(s) nom(s) de modèle d'une config, sans dupliquer le YAML.

    Les noms doivent correspondre EXACTEMENT à ceux servis sur llm.lab. Dès qu'un
    nom est surchargé, le `name` du run (donc le fichier de sortie
    `data/processed/<name>_predictions.jsonl`) est suffixé par le(s) modèle(s)
    utilisé(s), pour ne jamais écraser le checkpoint d'un autre modèle.

    Args:
        config: Configuration chargée depuis le YAML.
        model_name: Nom de modèle pour l'étape 1 (unique étape en end_to_end,
            transcription en two_stage). None = ne pas surcharger.
        model_stage2_name: Nom de modèle pour l'étape 2 (codage textuel,
            two_stage uniquement). None = ne pas surcharger.

    Returns:
        Une nouvelle config avec les noms de modèle et le `name` mis à jour.

    Raises:
        ValueError: Si `model_stage2_name` est fourni alors que la config n'a
            pas de bloc `model_stage2` (approche `end_to_end`).
    """
    if model_name is None and model_stage2_name is None:
        return config

    current_stage2 = config.model_stage2
    if model_stage2_name is not None and current_stage2 is None:
        raise ValueError(
            "--model-stage2-name n'a de sens qu'en approche two_stage "
            "(la config chargée n'a pas de bloc `model_stage2`)."
        )

    updates: dict[str, object] = {}
    suffix_parts = []

    if model_name is not None:
        updates["model"] = config.model.model_copy(update={"name": model_name})
        suffix_parts.append(_slugify_model_name(model_name))

    if model_stage2_name is not None and current_stage2 is not None:
        updates["model_stage2"] = current_stage2.model_copy(update={"name": model_stage2_name})
        stage2_slug = _slugify_model_name(model_stage2_name)
        if stage2_slug not in suffix_parts:
            suffix_parts.append(stage2_slug)

    updates["name"] = f"{config.name}_{'_'.join(suffix_parts)}"
    return config.model_copy(update=updates)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration pour le pipeline HTR (transcription seule, corpus Scoledit)
# ─────────────────────────────────────────────────────────────────────────────


class HTRDataConfig(BaseModel):
    """Localisation des données Scoledit pour l'évaluation HTR."""

    scans_path: str  # local ou s3://.../scans/CE1/
    annotations_path: str  # transcriptions JSON de référence, local ou s3://.../annotation/CE1/
    limit: int | None = None  # limiter le nombre d'échantillons (tests rapides)


class HTRExperimentConfig(BaseModel):
    """Configuration d'une expérience HTR : données et métriques (CER/WER) spécifiques."""

    name: str = Field(..., description="Nom unique du run.")
    seed: int = 42
    # Nombre d'échantillons transcrits EN PARALLÈLE (cf. `concurrency` du scoring).
    concurrency: int = Field(default=8, ge=1)
    model: ModelConfig
    data: HTRDataConfig
    read_final_state: bool = True  # en cas de rature, lire l'état final


def load_htr_config(path: str | Path) -> HTRExperimentConfig:
    """Charge et valide une configuration HTR depuis un fichier YAML.

    Args:
        path: Chemin du fichier YAML de configuration HTR.

    Returns:
        La configuration d'expérience HTR validée.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return HTRExperimentConfig.model_validate(raw)
