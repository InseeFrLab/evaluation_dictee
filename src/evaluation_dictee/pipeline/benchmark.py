"""Orchestration d'un run de benchmark : données -> modèle -> métriques, via `ExperimentConfig`.

Prédictions écrites dans data/processed/<run_name>_predictions.jsonl (une ligne par item x copie).
"""

from __future__ import annotations

import contextvars
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.data import reference
from evaluation_dictee.data.grid import load_grid
from evaluation_dictee.data.loaders import Copy, ink_ratio, load_dataset, load_image
from evaluation_dictee.evaluation.metrics import ScoringMetrics, compute_scoring_metrics
from evaluation_dictee.models.base import CopyPrediction, ItemPrediction, Scorer
from evaluation_dictee.utils.logging import get_logger
from evaluation_dictee.utils.tracking import copy_trace

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Résultat d'un run : métriques + prédictions/labels alignés (y_true/y_pred aplatis)."""

    metrics: ScoringMetrics
    y_true: list[str]
    y_pred: list[str]
    confidences: list[float | None]
    item_ids: list[str] = field(default_factory=list)
    copy_ids: list[str] = field(default_factory=list)
    predictions_path: Path | None = None
    non_transcribed: list[str] = field(default_factory=list)
    blank_copies: list[str] = field(default_factory=list)


@dataclass
class _CopyOutcome:
    """Résultat du scoring d'une copie, produit par un worker et consommé au principal."""

    copy_id: str
    status: str  # "ok" | "failed" | "non_transcribed"
    lines: str = ""  # lignes JSONL prêtes à écrire (status "ok")
    error: str = ""  # message d'erreur (status "failed")
    n_attempts: int = 1  # nombre d'essais (status "non_transcribed")
    blank: bool = False  # copie vierge auto-codée "0" (status "ok")


def _load_processed_copy_ids(predictions_path: Path) -> set[str]:
    """Renvoie les copy_id déjà présents dans un fichier de prédictions (pour reprendre un run).

    Args:
        predictions_path: chemin du fichier JSONL de prédictions.

    Returns:
        L'ensemble des copy_id déjà traités (vide si le fichier n'existe pas).
    """
    if not predictions_path.exists():
        return set()
    processed: set[str] = set()
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                processed.add(rec["copy_id"])
            except (json.JSONDecodeError, KeyError):
                # Ligne tronquée par un crash : ignorée.
                continue
    return processed


def run_benchmark(
    config: ExperimentConfig,
    scorer: Scorer,
    output_dir: str | Path = "data/processed",
    concurrency: int | None = None,
) -> BenchmarkResult:
    """Exécute un benchmark complet pour une config et un modèle donnés.

    Écriture incrémentale (fsync par copie) + reprise auto depuis le JSONL existant ;
    une erreur API sur une copie est loggée dans failed_copies.txt et le run continue.
    Pour repartir de zéro : supprimer le JSONL ou changer `config.name`.

    Les copies sont évaluées en parallèle (`concurrency` requêtes vLLM concurrentes) :
    seul le scoring est parallélisé, l'écriture du JSONL et l'agrégation restent dans
    le thread principal (aucun verrou, crash-safety et reprise inchangées).

    Args:
        config: configuration de l'expérience (données, grille, nom du run).
        scorer: modèle chargé de coder chaque copie.
        output_dir: dossier où écrire les prédictions et les fichiers d'échec.
        concurrency: nombre de copies évaluées en parallèle ; None = valeur de la
            config (`config.concurrency`).

    Returns:
        Le résultat du run : métriques agrégées, labels/prédictions alignés et
        chemin du fichier de prédictions.

    Raises:
        RuntimeError: si aucune copie n'a pu être chargée.
    """
    logger.info("Labels : %s", config.data.labels_path)
    logger.info("Images : %s", config.data.images_path)

    copies: list[Copy] = load_dataset(
        images_dir=config.data.images_path,
        labels_csv=config.data.labels_path,
        limit=config.data.limit,
    )

    # Échec explicite si aucune copie — évite la boucle silencieuse
    if not copies:
        raise RuntimeError(
            "Aucune copie chargée. Causes possibles :\n"
            f"  1. Le CSV est inaccessible : {config.data.labels_path}\n"
            f"  2. Les images sont introuvables : {config.data.images_path}\n"
            "     (les noms dans le CSV doivent correspondre aux fichiers du dossier)\n"
            "  3. Les identifiants S3 ne sont pas configurés.\n"
            'Diagnostic : uv run python -c "'
            "from evaluation_dictee.data.loaders import load_labels; "
            f"labels=load_labels('{config.data.labels_path}'); "
            "print(len(labels), 'copies dans le CSV')\""
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{config.name}_predictions.jsonl"
    failed_path = output_dir / f"{config.name}_failed_copies.txt"

    processed = _load_processed_copy_ids(out_path)
    if processed:
        logger.info(
            "Reprise détectée : %d copies déjà traitées dans %s. On saute ces copies.",
            len(processed),
            out_path,
        )
    copies_a_traiter = [c for c in copies if c.copy_id not in processed]
    logger.info(
        "%d copies au total, %d à traiter (%d déjà faites).",
        len(copies),
        len(copies_a_traiter),
        len(processed),
    )

    reference_text = load_grid(config.data.grid_path).reference_text
    scheme = config.grid.scheme
    workers = concurrency if concurrency is not None else config.concurrency
    blank_threshold = config.data.blank_ink_threshold

    non_transcrites: list[str] = []
    failed_copies: list[tuple[str, str]] = []  # (copy_id, message d'erreur)
    blank_copies: list[str] = []  # copies vierges auto-codées "0"

    def _score_one(copy: Copy) -> _CopyOutcome:
        """Score une copie et prépare ses lignes JSONL (exécuté dans un thread worker).

        Aucune écriture fichier ni mutation d'état partagé ici : uniquement l'appel
        modèle (thread-safe) et la trace Langfuse de la copie. Le résultat est
        consommé dans le thread principal.
        """
        # Une trace Langfuse par copie (no-op si Langfuse indisponible, trace vaut None).
        with copy_trace(copy) as trace:
            # Détection copie vierge AVANT tout appel modèle : formulaire non rempli
            # (seul le pré-imprimé marque quelques % d'encre). Traitée identiquement
            # pour toutes les méthodes → tous les items codés "0" (absent), sans appel
            # modèle. Empêche l'end-to-end d'halluciner la référence sur une page vide.
            # La densité mesurée est conservée telle quelle dans le JSONL, et pas
            # seulement le verdict : c'est ce qui permet de tracer la distribution
            # et de rejuger le seuil sans relancer un benchmark de 30 h.
            try:
                densite_encre = ink_ratio(load_image(copy.image_path))
                is_blank = blank_threshold > 0 and densite_encre < blank_threshold
            except Exception as exc:  # noqa: BLE001 — image illisible : on remonte un échec
                if trace is not None:
                    trace.update(level="ERROR", status_message=str(exc))
                return _CopyOutcome(copy.copy_id, "failed", error=str(exc))

            if is_blank:
                prediction: CopyPrediction = CopyPrediction(
                    copy_id=copy.copy_id,
                    items=[
                        ItemPrediction(item_id=i, code="0", confidence=1.0, transcription="")
                        for i in copy.item_ids
                    ],
                )
            else:
                try:
                    prediction = scorer.score_copy(copy, reference_text)
                except Exception as exc:  # noqa: BLE001 — on veut TOUT rattraper ici
                    if trace is not None:
                        trace.update(level="ERROR", status_message=str(exc))
                    return _CopyOutcome(copy.copy_id, "failed", error=str(exc))

                if not prediction.transcribed:
                    if trace is not None:
                        trace.update(
                            level="WARNING",
                            status_message="copie non transcrite",
                            output={"transcribed": False, "n_attempts": prediction.n_attempts},
                        )
                    return _CopyOutcome(
                        copy.copy_id, "non_transcribed", n_attempts=prediction.n_attempts
                    )

            pred_by_id = {it.item_id: it for it in prediction.items}
            records = []
            n_accord = 0
            for item_id, expert_code in zip(copy.item_ids, copy.expert_codes, strict=True):
                pred = pred_by_id.get(item_id)
                true_code = reference.normalize(expert_code, scheme)
                pred_code = reference.normalize(pred.code, scheme) if pred else "?"
                records.append(
                    {
                        "copy_id": copy.copy_id,
                        "item_id": item_id,
                        "y_true": true_code,
                        "y_pred": pred_code,
                        "confidence": pred.confidence if pred else 0.0,
                        "transcription": pred.transcription if pred else None,
                        "comparaison": pred.comparaison if pred else None,
                        "raw_transcription": prediction.raw_transcription,
                        "approach": config.approach,
                        "blank": is_blank,
                        "ink_ratio": densite_encre,
                    }
                )
                if pred_code == true_code:
                    n_accord += 1

            if trace is not None and records:
                accord_copie = n_accord / len(records)
                trace.update(
                    output={
                        "n_items": len(records),
                        "raw_agreement": accord_copie,
                        "blank": is_blank,
                    }
                )
                trace.score_trace(name="raw_agreement", value=accord_copie)

            lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
            return _CopyOutcome(copy.copy_id, "ok", lines=lines, blank=is_blank)

    def _consume(outcome: _CopyOutcome, f_pred: object) -> None:
        """Écrit/comptabilise le résultat d'une copie (thread principal uniquement)."""
        if outcome.status == "failed":
            logger.error(
                "Échec sur la copie %s : %s. On passe à la suivante.",
                outcome.copy_id,
                outcome.error,
            )
            failed_copies.append((outcome.copy_id, outcome.error))
        elif outcome.status == "non_transcribed":
            non_transcrites.append(outcome.copy_id)
            logger.warning(
                "Copie non transcrite après %d tentative(s) : %s (exclue des métriques).",
                outcome.n_attempts,
                outcome.copy_id,
            )
        else:  # "ok"
            if outcome.blank:
                blank_copies.append(outcome.copy_id)
            f_pred.write(outcome.lines)  # type: ignore[attr-defined]
            # Flush + fsync : garantit que la copie écrite survit à un crash ultérieur.
            f_pred.flush()  # type: ignore[attr-defined]
            os.fsync(f_pred.fileno())  # type: ignore[attr-defined]

    # Mode APPEND : conserve les copies déjà traitées lors d'une reprise.
    with open(out_path, "a", encoding="utf-8") as f_pred:
        desc = f"Évaluation ({config.name}, {workers} en parallèle)"
        if workers <= 1:
            # Chemin séquentiel (comportement historique), sans thread ni contexte copié.
            for copy in tqdm(copies_a_traiter, desc=desc):
                _consume(_score_one(copy), f_pred)
        else:
            # Chaque copie est soumise avec une COPIE du contexte courant, pour que la
            # trace Langfuse hérite des attributs de session (propagate_attributes est
            # actif dans le thread principal, pas dans les workers).
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(contextvars.copy_context().run, _score_one, copy)
                    for copy in copies_a_traiter
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    _consume(future.result(), f_pred)

    if failed_copies:
        with open(failed_path, "w", encoding="utf-8") as f:
            for cid, msg in failed_copies:
                f.write(f"{cid}\t{msg}\n")
        logger.warning(
            "%d copie(s) en échec, listées dans %s (le run continuera à les "
            "reprendre au prochain lancement).",
            len(failed_copies),
            failed_path,
        )

    logger.info("Prédictions sauvegardées : %s", out_path)

    # Recharger tout le JSONL (copies de ce run + reprises) pour construire les métriques.
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float | None] = []
    item_ids: list[str] = []
    copy_ids: list[str] = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            y_true.append(rec["y_true"])
            y_pred.append(rec["y_pred"])
            confidences.append(rec.get("confidence"))
            item_ids.append(rec["item_id"])
            copy_ids.append(rec["copy_id"])
    # Garde-fou : après normalisation, un code hors de l'alphabet attendu signale une
    # incohérence (prétraitement expert oublié, prompt non aligné sur le schéma).
    attendus = reference.allowed_codes(scheme)
    codes_vus = set(y_true) | set(y_pred)
    intrus = codes_vus - attendus - {"?"}  # "?" = réponse modèle non parsée, traité à part
    if intrus:
        logger.warning(
            "Codes hors du schéma '%s' (attendu %s) détectés : %s. "
            "Vérifier le prétraitement des codes experts et le prompt du modèle.",
            scheme,
            sorted(attendus),
            sorted(intrus),
        )

    if non_transcrites:
        logger.warning(
            "%d copie(s) non transcrite(s) exclue(s) des métriques : %s",
            len(non_transcrites),
            non_transcrites,
        )

    if blank_copies:
        logger.info(
            "%d copie(s) vierge(s) détectée(s) (encre < %.1f%%) et auto-codée(s) "
            '"0" (absent) sans appel modèle : %s',
            len(blank_copies),
            blank_threshold * 100,
            blank_copies,
        )

    metrics = compute_scoring_metrics(y_true, y_pred)
    return BenchmarkResult(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        item_ids=item_ids,
        copy_ids=copy_ids,
        predictions_path=out_path,
        non_transcribed=non_transcrites,
        blank_copies=blank_copies,
    )
