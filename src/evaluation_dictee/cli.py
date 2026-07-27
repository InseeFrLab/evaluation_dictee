"""Interface en ligne de commande, exposée via `eval-ecrit` (voir pyproject.toml).

Exemple : `eval-ecrit benchmark configs/scoring/dictee_REFERENCE.yaml`
"""

from __future__ import annotations

import typer
import yaml
from langfuse import get_client
from rich.console import Console
from rich.table import Table

from evaluation_dictee.config import Secrets, load_config
from evaluation_dictee.data.grid import load_grid
from evaluation_dictee.evaluation.metrics import ScoringMetrics
from evaluation_dictee.models.factory import build_scorer
from evaluation_dictee.pipeline.benchmark import run_benchmark
from evaluation_dictee.utils.s3_export import export_run
from evaluation_dictee.utils.tracking import experiment_run, log_metrics

app = typer.Typer(help="Évaluation automatique de la production d'écrit (DEPP × SSP Lab).")
console = Console()


@app.command()
def benchmark(config_path: str) -> None:
    """Lance un benchmark à partir d'un fichier de configuration YAML.

    Charge la config et les secrets, construit le scorer, exécute le run (tracé
    dans Langfuse), journalise les métriques puis les affiche.

    Args:
        config_path: Chemin du fichier YAML de configuration du run.
    """
    config = load_config(config_path)
    secrets = Secrets()

    console.print(f"[bold]Run :[/bold] {config.name}")
    console.print(f"Modèle : {config.model.name} | Méthode : {config.prompt.method}")

    scorer = build_scorer(
        config=config,
        grid_items=load_grid(config.data.grid_path).items,
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
                },
            )
    finally:
        # Flush explicite : Langfuse envoie en asynchrone, sinon les dernières traces sont perdues.
        get_client().flush()

    _print_metrics(result.metrics)


@app.command()
def export(
    config_path: str | None = typer.Argument(
        None, help="YAML du run (le nom est lu dans le champ `name`)."
    ),
    run_name: str | None = typer.Option(
        None, "--run-name", help="Nom du run (alternative à config_path)."
    ),
    htr: bool = typer.Option(
        False, "--htr", help="Exporte le fichier HTR au lieu du fichier de scoring."
    ),
    source_dir: str = typer.Option("data/processed", help="Dossier local des prédictions."),
    dest_prefix: str | None = typer.Option(
        None, help="Préfixe S3 de destination. [défaut : S3_PREDICTIONS_PREFIX]"
    ),
) -> None:
    """Exporte les prédictions d'un run vers S3 (répertoire predictions/).

    Le pipeline écrit les prédictions en local ; cette commande pousse le JSONL
    fini vers S3 pour relancer notebooks et site Quarto sans réexécuter le run.
    Fournir SOIT un YAML de run, SOIT `--run-name`.

    Args:
        config_path: Chemin du YAML du run (le nom est lu dans `name`).
        run_name: Nom du run, alternative au YAML.
        htr: Exporte `<name>_htr_predictions.jsonl` (transcription seule).
        source_dir: Dossier local des prédictions.
        dest_prefix: Préfixe S3 de destination (sinon celui des secrets/env).
    """
    if not (config_path or run_name):
        raise typer.BadParameter("Fournir un YAML de run ou --run-name.")

    if run_name is None:
        with open(config_path, encoding="utf-8") as f:
            run_name = str(yaml.safe_load(f)["name"])

    prefix = dest_prefix or Secrets().s3_predictions_prefix
    dest = export_run(run_name, prefix, source_dir=source_dir, htr=htr)
    console.print(f"[green]Exporté sur S3 :[/green] {dest}")


def _print_metrics(metrics: ScoringMetrics) -> None:
    """Affiche les métriques dans un tableau lisible.

    Args:
        metrics: Métriques de scoring à présenter.
    """
    table = Table(title="Résultats du benchmark")
    table.add_column("Métrique")
    table.add_column("Valeur", justify="right")
    table.add_row("Items comparés", str(metrics.n_items))
    table.add_row("Accord brut", f"{metrics.raw_agreement:.1%}")
    table.add_row("Kappa de Cohen", f"{metrics.cohen_kappa:.3f}")
    console.print(table)


if __name__ == "__main__":
    app()
