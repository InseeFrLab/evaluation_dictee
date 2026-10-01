"""Tests du garde-fou en temps réel sur la cohérence du comptage (expérimentation 2).

Pipeline pur : aucune option de prompt dédiée. L'alerte est active dès que
`count_items` est vrai, quel que soit le bras — E1, E3 et E4 en héritent
automatiquement puisqu'ils gardent les innovations de comptage+.
"""

import logging

from evaluation_dictee.config import ModelConfig, PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.data.loaders import Copy
from evaluation_dictee.models.base import ItemPrediction
from evaluation_dictee.models.vlm import VLMScorer

GRID = [GridItem(f"i{n}", f"mot{n}", "mot", [], [], []) for n in range(5)]


def _scorer(count_items: bool) -> VLMScorer:
    return VLMScorer(
        model_config=ModelConfig(name="m"),
        prompt_config=PromptConfig(count_items=count_items, enforce_count=count_items),
        base_url="http://x",
        api_key="k",
        grid_items=GRID,
    )


def _copy() -> Copy:
    return Copy(copy_id="c1.png", image_path="c1.png", item_ids=[g.item_id for g in GRID])


def _items(codes: list[str]) -> list[ItemPrediction]:
    return [ItemPrediction(item_id=f"i{n}", code=c) for n, c in enumerate(codes)]


def test_alerte_declenchee_si_comptage_contredit(caplog) -> None:
    """5 items, n_items_lus=3 -> 2 absents attendus, mais 0 codé « 0 »."""
    scorer = _scorer(count_items=True)
    with caplog.at_level(logging.WARNING):
        scorer._alerter_si_comptage_incoherent(
            _copy(), _items(["1", "1", "9", "1", "9"]), n_items_lus=3
        )
    assert any("INCOHÉRENT" in m for m in caplog.messages)


def test_pas_d_alerte_si_comptage_coherent(caplog) -> None:
    scorer = _scorer(count_items=True)
    with caplog.at_level(logging.WARNING):
        scorer._alerter_si_comptage_incoherent(
            _copy(), _items(["1", "1", "0", "1", "0"]), n_items_lus=3
        )
    assert not any("INCOHÉRENT" in m for m in caplog.messages)


def test_pas_d_alerte_si_count_items_inactif(caplog) -> None:
    """Le garde-fou n'a pas de sens hors comptage : il ne doit jamais se déclencher."""
    scorer = _scorer(count_items=False)
    with caplog.at_level(logging.WARNING):
        scorer._alerter_si_comptage_incoherent(
            _copy(), _items(["1", "1", "9", "1", "9"]), n_items_lus=3
        )
    assert not any("INCOHÉRENT" in m for m in caplog.messages)


def test_pas_d_alerte_si_n_items_lus_absent(caplog) -> None:
    """Le modèle n'a pas fourni de comptage exploitable : rien à comparer."""
    scorer = _scorer(count_items=True)
    with caplog.at_level(logging.WARNING):
        scorer._alerter_si_comptage_incoherent(
            _copy(), _items(["1", "1", "9", "1", "9"]), n_items_lus=None
        )
    assert not any("INCOHÉRENT" in m for m in caplog.messages)
