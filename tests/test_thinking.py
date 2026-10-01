"""Tests du mode raisonnement natif (thinking) : activation, capture, troncature.

À ne pas confondre avec `test_chain_of_thought.py` : la chain-of-thought du dépôt ajoute
un champ « comparaison » DANS la sortie structurée, alors que le raisonnement natif est
un mode du modèle dont la sortie JSON reste inchangée (le raisonnement arrive dans un
champ `reasoning_content` séparé).
"""

from types import SimpleNamespace

from evaluation_dictee.config import ExperimentConfig, ModelConfig
from evaluation_dictee.models.vlm import extract_reasoning, log_if_truncated, thinking_kwargs
from evaluation_dictee.pipeline.benchmark import resume_durees
from evaluation_dictee.utils.tracking import _run_metadata, _run_tags


def _config(**model_kwargs: object) -> ModelConfig:
    """ModelConfig minimale, surchargeable champ par champ."""
    base: dict[str, object] = {"name": "qwen3-6-35b-moe", "kind": "vlm"}
    return ModelConfig(**{**base, **model_kwargs})  # type: ignore[arg-type]


def _experiment(**model_kwargs: object) -> ExperimentConfig:
    """ExperimentConfig minimale pour les tests de traçabilité."""
    return ExperimentConfig(
        name="run_test",
        model=_config(**model_kwargs),
        data={"images_path": "img/", "labels_path": "labels.csv"},  # type: ignore[arg-type]
    )


# ── Activation explicite du mode ─────────────────────────────────────────────


def test_thinking_actif_envoie_enable_thinking_true() -> None:
    """`disable_thinking: false` demande EXPLICITEMENT le raisonnement."""
    assert thinking_kwargs(_config(disable_thinking=False)) == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_thinking_coupe_envoie_enable_thinking_false() -> None:
    """`disable_thinking: true` coupe le raisonnement, comme avant."""
    assert thinking_kwargs(_config(disable_thinking=True)) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_le_flag_est_toujours_transmis() -> None:
    """La valeur est envoyée dans les DEUX sens, jamais laissée au défaut du modèle.

    Le défaut diffère selon le modèle (thinking ON par défaut sur qwen3-6-35b-moe, OFF
    sur gemma4-26b-moe) : ne rien envoyer rendrait une même config non reproductible
    d'un modèle à l'autre.
    """
    for disable in (True, False):
        kwargs = thinking_kwargs(_config(disable_thinking=disable))
        assert "enable_thinking" in kwargs["chat_template_kwargs"]


# ── Capture du raisonnement ──────────────────────────────────────────────────


def test_extrait_le_reasoning_content() -> None:
    message = SimpleNamespace(content='{"items": []}', reasoning_content="Item 1 : identique.")
    assert extract_reasoning(message) == "Item 1 : identique."


def test_extrait_le_reasoning_content_depuis_model_extra() -> None:
    """Le champ n'est pas typé par le SDK : il peut n'exister que dans `model_extra`."""
    message = SimpleNamespace(model_extra={"reasoning_content": "Item 1 : il manque le 's'."})
    assert extract_reasoning(message) == "Item 1 : il manque le 's'."


def test_pas_de_raisonnement_renvoie_none() -> None:
    """Un modèle sans mode thinking (qwen3-vl) ne doit pas produire de chaîne vide."""
    assert extract_reasoning(SimpleNamespace(content="{}", model_extra={})) is None
    assert extract_reasoning(SimpleNamespace(content="{}", reasoning_content="   ")) is None


# ── Détection de troncature ──────────────────────────────────────────────────


def _reponse(finish_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason)],
        usage=SimpleNamespace(completion_tokens=16384),
    )


def test_troncature_detectee() -> None:
    """`finish_reason == "length"` = JSON incomplet : la copie sera perdue."""
    assert log_if_truncated(_reponse("length"), "c1.png", 16384) is True


def test_reponse_complete_non_signalee() -> None:
    assert log_if_truncated(_reponse("stop"), "c1.png", 16384) is False


# ── Traçabilité Langfuse ─────────────────────────────────────────────────────


def test_metadata_distingue_les_deux_bras() -> None:
    """Deux runs ne différant que par le raisonnement doivent être discernables."""
    avec = _run_metadata(_experiment(disable_thinking=False))
    sans = _run_metadata(_experiment(disable_thinking=True))
    assert avec["thinking"] == "on"
    assert sans["thinking"] == "off"


def test_tag_thinking_present() -> None:
    assert "thinking:on" in _run_tags(_experiment(disable_thinking=False))
    assert "thinking:off" in _run_tags(_experiment(disable_thinking=True))


# ── Coût en temps ────────────────────────────────────────────────────────────


def test_resume_durees_projette_le_run_complet() -> None:
    """La projection tient compte du parallélisme, sinon elle surestime d'un facteur N."""
    # 12 copies de 60 s, 12 en parallèle → 3469 copies ≈ 3469 x 60 / 12 / 3600 ≈ 4,8 h.
    ligne = resume_durees([60.0] * 12, workers=12, n_copies_total=3469)
    assert "médiane 60.0 s" in ligne
    assert "4.8 h" in ligne


def test_resume_durees_sans_mesure() -> None:
    """Aucune copie traitée (run entièrement repris du checkpoint) : pas de ligne."""
    assert resume_durees([], workers=12, n_copies_total=3469) == ""
