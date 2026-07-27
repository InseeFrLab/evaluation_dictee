"""Pipeline d'évaluation de la TRANSCRIPTION (HTR) sur le corpus Scoledit (CER/WER).

Le transcripteur encapsule l'appel VLM et n'expose que `transcribe(image_path) -> str`,
pour changer de modèle facilement.
"""

from __future__ import annotations

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image
from tqdm import tqdm

from evaluation_dictee.config import ModelConfig
from evaluation_dictee.data.loaders import load_image
from evaluation_dictee.pipeline.prompts import attach_image, build_transcription_prompt
from evaluation_dictee.transcription.htr_metrics import (
    TranscriptionMetrics,
    compute_transcription_metrics,
)
from evaluation_dictee.transcription.scoledit import ScoledtSample


def _image_to_data_url(image: Image.Image) -> str:
    """Encode une image PIL en data URL base64.

    Args:
        image: Image PIL à encoder.

    Returns:
        Data URL PNG encodée en base64.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


class VLMTranscriber:
    """Transcripteur HTR basé sur un VLM (API compatible OpenAI)."""

    def __init__(
        self,
        model_config: ModelConfig,
        base_url: str,
        api_key: str,
        read_final_state: bool = True,
    ) -> None:
        """Initialise le client de transcription (read_final_state : état final si rature).

        Args:
            model_config: Configuration du modèle VLM (nom, température, retries...).
            base_url: URL de base de l'API compatible OpenAI.
            api_key: Clé d'API.
            read_final_state: Si True, lire l'état final (version corrigée) en cas de rature.
        """
        self.model_config = model_config
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.prompt_messages = build_transcription_prompt(read_final_state=read_final_state)

    def transcribe(self, image_path: str) -> str:
        """Transcrit une image manuscrite en texte brut, fautes préservées (vide si échec).

        Args:
            image_path: Chemin de l'image à transcrire.

        Returns:
            Transcription du texte, ou chaîne vide si tous les essais échouent.
        """
        image = load_image(image_path)
        messages = cast(
            "list[ChatCompletionMessageParam]",
            attach_image(self.prompt_messages, _image_to_data_url(image)),
        )
        for attempt in range(self.model_config.max_retries + 1):
            temp = self.model_config.temperature + (0.3 if attempt > 0 else 0.0)
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                temperature=temp,
                max_tokens=self.model_config.max_tokens,
                messages=messages,
            )
            content = response.choices[0].message.content or "{}"
            transcription = self._parse(content)
            if transcription.strip():
                return transcription
        return ""

    @staticmethod
    def _parse(content: str) -> str:
        """Extrait le champ 'transcription' du JSON, avec repli sur texte brut.

        Args:
            content: Réponse brute du modèle (JSON éventuellement en bloc markdown).

        Returns:
            Transcription extraite, ou le contenu brut si le JSON est invalide.
        """
        try:
            cleaned = content.strip().removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
            return str(json.loads(cleaned).get("transcription", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            return content.strip()


@dataclass
class HTRBenchmarkResult:
    """Résultat d'un benchmark de transcription."""

    per_sample: pd.DataFrame
    mean_cer: float
    mean_wer: float
    n_echecs: int
    predictions_path: Path | None = None
    failed_scans: list[str] = field(default_factory=list)


def run_htr_benchmark(
    samples: list[ScoledtSample],
    transcriber: VLMTranscriber,
    run_name: str,
    output_dir: str | Path = "data/processed",
    concurrency: int = 8,
) -> HTRBenchmarkResult:
    """Transcrit tous les échantillons et calcule les métriques HTR.

    Les échantillons sont transcrits en parallèle (`concurrency` requêtes vLLM
    concurrentes), mais les enregistrements sont ré-ordonnés dans l'ordre de
    `samples` : la sortie est identique au mode séquentiel (transcription
    déterministe à température 0). `concurrency=1` = ancien comportement.

    Args:
        samples: Échantillons Scoledit à transcrire.
        transcriber: Transcripteur VLM à utiliser (thread-safe).
        run_name: Nom du run (préfixe du fichier de prédictions JSONL).
        output_dir: Dossier de sortie des prédictions.
        concurrency: nombre de transcriptions menées en parallèle.

    Returns:
        Résultat du benchmark : prédictions par échantillon, CER/WER moyens
        (micro-pondérés par la longueur de la référence) et scans en échec.
    """

    def _transcribe_one(sample: ScoledtSample) -> dict:
        """Transcrit un échantillon et construit son enregistrement (thread worker)."""
        hypothese = transcriber.transcribe(sample.image_path)
        m: TranscriptionMetrics = compute_transcription_metrics(sample.reference, hypothese)
        return {
            "scan": sample.scan,
            "level": sample.level,
            "reference": sample.reference,
            "hypothese": hypothese,
            "cer": m.cer,
            "wer": m.wer,
            "cer_normalise": m.cer_normalise,
            "wer_normalise": m.wer_normalise,
            "n_char_ref": m.n_char_ref,
            "n_mots_ref": m.n_mots_ref,
            "transcrit": bool(hypothese.strip()),
        }

    desc = f"Transcription ({run_name}, {concurrency} en parallèle)"
    if concurrency <= 1:
        records = [_transcribe_one(s) for s in tqdm(samples, desc=desc)]
    else:
        # Résultats stockés par indice d'origine → ordre de `samples` préservé,
        # quel que soit l'ordre d'achèvement des workers.
        records = [None] * len(samples)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            index_of = {executor.submit(_transcribe_one, s): i for i, s in enumerate(samples)}
            for future in tqdm(as_completed(index_of), total=len(samples), desc=desc):
                records[index_of[future]] = future.result()

    # Échecs (transcription vide), dans l'ordre des échantillons.
    failed = [r["scan"] for r in records if not r["transcrit"]]

    df = pd.DataFrame(records)
    # CER/WER micro-pondérés par la longueur de la référence
    ok = df[df["transcrit"]]
    mean_cer = (
        float((ok["cer"] * ok["n_char_ref"]).sum() / ok["n_char_ref"].sum())
        if len(ok) and ok["n_char_ref"].sum() > 0
        else float("nan")
    )
    mean_wer = (
        float((ok["wer"] * ok["n_mots_ref"]).sum() / ok["n_mots_ref"].sum())
        if len(ok) and ok["n_mots_ref"].sum() > 0
        else float("nan")
    )

    out_path = Path(output_dir) / f"{run_name}_htr_predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return HTRBenchmarkResult(
        per_sample=df,
        mean_cer=mean_cer,
        mean_wer=mean_wer,
        n_echecs=len(failed),
        predictions_path=out_path,
        failed_scans=failed,
    )
