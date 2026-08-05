"""Test du chargement de configuration."""

from pathlib import Path

import pytest

from evaluation_dictee.config import load_config, override_model_names


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
    updated = override_model_names(config, model_name="qwen3.6-35b-moe")

    assert updated.model.name == "qwen3.6-35b-moe"
    assert updated.name == "dictee_end2end_qwen3-6-35b-moe"
    # La config d'origine n'est pas mutée.
    assert config.model.name == "gemma4-26b-moe"
    assert config.name == "dictee_end2end"


def test_override_model_names_two_stage_les_deux_etapes() -> None:
    config = load_config(Path("configs/scoring/dictee_two_stage.yaml"))
    updated = override_model_names(
        config, model_name="qwen3.6-35b-moe", model_stage2_name="qwen3.6-35b-moe"
    )

    assert updated.model.name == "qwen3.6-35b-moe"
    assert updated.model_stage2.name == "qwen3.6-35b-moe"
    # Même modèle aux deux étapes ⇒ un seul suffixe (pas de doublon).
    assert updated.name == "dictee_two_stage_qwen3-6-35b-moe"


def test_override_model_names_deux_modeles_differents() -> None:
    config = load_config(Path("configs/scoring/dictee_two_stage.yaml"))
    updated = override_model_names(
        config, model_name="qwen3.6-35b-moe", model_stage2_name="gemma4-26b-moe"
    )

    assert updated.name == "dictee_two_stage_qwen3-6-35b-moe_gemma4-26b-moe"


def test_override_model_stage2_sans_bloc_leve_une_erreur() -> None:
    config = load_config(Path("configs/scoring/dictee_end2end.yaml"))
    with pytest.raises(ValueError, match="model_stage2"):
        override_model_names(config, model_stage2_name="qwen3.6-35b-moe")
