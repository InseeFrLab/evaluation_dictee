"""Test du chargement de configuration."""

from pathlib import Path

import pytest

from evaluation_dictee.config import load_config, override_model_names, run_output_name


def test_charge_config_exemple() -> None:
    # On part de la racine du dépôt (pytest est lancé depuis là)
    config = load_config(Path("configs/scoring/dictee_REFERENCE.yaml"))
    assert config.name == "dictee_REFERENCE"
    assert config.model.kind == "vlm"
    assert config.grid.scheme == "simplifiee"
    assert config.prompt.method == "C"
    assert config.prompt.read_final_state is True


def test_override_model_names_sans_argument_ne_change_rien() -> None:
    config = load_config(Path("configs/scoring/dictee_end2end.yaml"))
    assert override_model_names(config) is config


def test_override_model_names_end_to_end_suffixe_le_nom() -> None:
    config = load_config(Path("configs/scoring/dictee_end2end.yaml"))
    modele_yaml = config.model.name
    updated = override_model_names(config, model_name="qwen3.6-35b-moe")

    assert updated.model.name == "qwen3.6-35b-moe"
    # Le point est CONSERVÉ par `_slugify_model_name` : un nom fautif avec un point
    # et le nom réel avec un tiret doivent produire deux suffixes distincts, donc
    # deux fichiers de sortie distincts. Ne pas « corriger » en tiret.
    assert updated.name == "dictee_end2end_qwen3.6-35b-moe"
    # La config d'origine n'est pas mutée (on ne fige pas le modèle du YAML ici,
    # il change au fil des expériences).
    assert config.model.name == modele_yaml
    assert config.name == "dictee_end2end"


def test_override_model_names_two_stage_les_deux_etapes() -> None:
    config = load_config(Path("configs/scoring/dictee_two_stage.yaml"))
    updated = override_model_names(
        config, model_name="qwen3.6-35b-moe", model_stage2_name="qwen3.6-35b-moe"
    )

    assert updated.model.name == "qwen3.6-35b-moe"
    assert updated.model_stage2.name == "qwen3.6-35b-moe"
    # Même modèle aux deux étapes ⇒ un seul suffixe (pas de doublon).
    assert updated.name == "dictee_two_stage_qwen3.6-35b-moe"


def test_override_model_names_deux_modeles_differents() -> None:
    config = load_config(Path("configs/scoring/dictee_two_stage.yaml"))
    updated = override_model_names(
        config, model_name="qwen3.6-35b-moe", model_stage2_name="gemma4-26b-moe"
    )

    assert updated.name == "dictee_two_stage_qwen3.6-35b-moe_gemma4-26b-moe"


def test_run_output_name_ajoute_le_modele_depuis_le_yaml() -> None:
    """Sans surcharge CLI, le modèle du YAML figure quand même dans le nom de sortie."""
    config = load_config(Path("configs/scoring/dictee_end2end.yaml"))
    assert run_output_name(config) == f"dictee_end2end_{config.model.name}"


def test_run_output_name_est_idempotent() -> None:
    """Un `name` déjà suffixé (lancement CLI) ne reçoit pas un second suffixe.

    C'est la garantie qu'un run lancé via le YAML et le même run lancé via
    --model-name écrivent dans le MÊME fichier, donc partagent leur checkpoint.
    """
    config = load_config(Path("configs/scoring/dictee_end2end.yaml"))
    via_yaml = run_output_name(config)
    via_cli = run_output_name(override_model_names(config, model_name=config.model.name))
    assert via_yaml == via_cli


def test_run_output_name_two_stage_deux_modeles_differents() -> None:
    """Deux modèles différents ⇒ deux fichiers distincts (pas de checkpoint mélangé)."""
    config = load_config(Path("configs/scoring/dictee_two_stage.yaml"))
    croise = override_model_names(
        config, model_name="qwen3.6-35b-moe", model_stage2_name="gemma4-26b-moe"
    )
    mono = override_model_names(config, model_name="qwen3.6-35b-moe")

    assert run_output_name(croise) == "dictee_two_stage_qwen3.6-35b-moe_gemma4-26b-moe"
    assert run_output_name(mono) != run_output_name(croise)


def test_run_output_name_slugifie_les_caracteres_interdits() -> None:
    """Un nom de modèle avec « / » ne doit pas produire un chemin à sous-dossier."""
    config = load_config(Path("configs/scoring/dictee_end2end.yaml"))
    updated = config.model_copy(
        update={"model": config.model.model_copy(update={"name": "Qwen/Qwen2.5-VL-7B-Instruct"})}
    )
    assert "/" not in run_output_name(updated)
    assert run_output_name(updated) == "dictee_end2end_Qwen-Qwen2.5-VL-7B-Instruct"


def test_override_model_stage2_sans_bloc_leve_une_erreur() -> None:
    config = load_config(Path("configs/scoring/dictee_end2end.yaml"))
    with pytest.raises(ValueError, match="model_stage2"):
        override_model_names(config, model_stage2_name="qwen3.6-35b-moe")
