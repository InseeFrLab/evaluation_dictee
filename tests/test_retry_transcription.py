"""Tests de la détection des copies non transcrites."""

from evaluation_dictee.config import ModelConfig, PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.data.loaders import Copy
from evaluation_dictee.models.vlm import VLMScorer


def _scorer() -> VLMScorer:
    items = [GridItem("i1", "Le", "mot", [], [], []), GridItem("i2", "soir", "mot", [], [], [])]
    return VLMScorer(
        ModelConfig(name="x"), PromptConfig(), "http://x", "k", items, scheme="simplifiee"
    )


def test_reponse_vide_marque_non_transcrite() -> None:
    copy = Copy("c.png", "c.png", ["1", "1"], ["i1", "i2"])
    pred = _scorer()._parse_response(copy, "{}")
    assert pred.transcribed is False


def test_json_invalide_marque_non_transcrite() -> None:
    copy = Copy("c.png", "c.png", ["1", "1"], ["i1", "i2"])
    pred = _scorer()._parse_response(copy, "désolé je ne peux pas lire cette image")
    assert pred.transcribed is False


def test_reponse_valide_est_transcrite() -> None:
    copy = Copy("c.png", "c.png", ["1", "1"], ["i1", "i2"])
    content = (
        '{"items": [{"item_id":"i1","transcription":"Le","code":"1","confidence":0.9},'
        '{"item_id":"i2","transcription":"soir","code":"1","confidence":0.9}]}'
    )
    pred = _scorer()._parse_response(copy, content)
    assert pred.transcribed is True
    assert len(pred.items) == 2
