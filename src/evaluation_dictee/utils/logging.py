"""Configuration du logging avec rich pour une sortie lisible."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Renvoie un logger configuré avec un affichage enrichi (rich).

    Le handler rich (niveau INFO, tracebacks enrichis) n'est ajouté qu'une seule
    fois : un logger déjà pourvu de handlers est renvoyé tel quel.

    Args:
        name: Nom du logger (typiquement `__name__` du module appelant).

    Returns:
        Le logger correspondant, muni d'un `RichHandler` au premier appel.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
