"""Point d'entrée pour lancer un benchmark de scoring.

Usage : uv run scripts/run_benchmark.py --config configs/scoring/dictee_REFERENCE.yaml
"""

from __future__ import annotations

import argparse

from langfuse import get_client

from evaluation_dictee.config import Secrets, load_config, override_model_names
from evaluation_dictee.data.grid import load_grid
from evaluation_dictee.evaluation.calibration import referral_curve
from evaluation_dictee.models.factory import build_scorer
from evaluation_dictee.pipeline.benchmark import run_benchmark
from evaluation_dictee.utils.logging import get_logger
from evaluation_dictee.utils.tracking import experiment_run, log_metrics

logger = get_logger(__name__)


def main() -> None:
    """Lance le benchmark et affiche métriques et calibration."""
    parser = argparse.ArgumentParser(description="Lance un benchmark d'évaluation.")
    parser.add_argument("--config", required=True, help="Chemin du fichier YAML.")
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Surcharge model.name (étape 1 / unique étape en end_to_end). "
            "Doit correspondre exactement au nom servi sur llm.lab."
        ),
    )
    parser.add_argument(
        "--model-stage2-name",
        default=None,
        help=(
            "Surcharge model_stage2.name (étape 2, codage textuel). "
            "Uniquement valide en approche two_stage."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = override_model_names(config, args.model_name, args.model_stage2_name)
    secrets = Secrets()

    if config.model_stage2 is not None:
        logger.info(
            "Run : %s | modèles : %s (étape 1), %s (étape 2)",
            config.name,
            config.model.name,
            config.model_stage2.name,
        )
    else:
        logger.info("Run : %s | modèle : %s", config.name, config.model.name)

    grid = load_grid(config.data.grid_path)
    scorer = build_scorer(
        config=config,
        grid_items=grid.items,
        base_url=secrets.llm_base_url,
        api_key=secrets.llm_api_key,
    )

    try:
        with experiment_run(config):
            result = run_benchmark(config, scorer)
            log_metrics(
                config,
                {
                    "raw_agreement": result.metrics.raw_agreement,
                    "cohen_kappa": result.metrics.cohen_kappa,
                    "n_items": result.metrics.n_items,
                    "n_blank": len(result.blank_copies),
                    "n_non_transcribed": len(result.non_transcribed),
                },
            )
    finally:
        # Langfuse envoie les traces en asynchrone : flush obligatoire sinon les
        # dernières traces sont perdues si le script se termine avant l'envoi.
        get_client().flush()

    logger.info("Accord brut : %.1f%%", result.metrics.raw_agreement * 100)
    logger.info("Kappa de Cohen : %.3f", result.metrics.cohen_kappa)
    logger.info(
        "Copies vierges auto-codées « 0 » : %d | copies non transcrites (exclues) : %d",
        len(result.blank_copies),
        len(result.non_transcribed),
    )

    logger.info("Courbe de renvoi humain :")
    for point in referral_curve(result.y_true, result.y_pred, result.confidences):
        logger.info(
            "  seuil %.1f → renvoi humain %.0f%% | erreur résiduelle %.1f%%",
            point.threshold,
            point.human_referral_rate * 100,
            point.residual_error_rate * 100,
        )


if __name__ == "__main__":
    main()
