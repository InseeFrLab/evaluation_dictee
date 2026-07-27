"""HTML autonome (image base64) comparant, par copie, codage expert et codage modèle.

DONNÉES SENSIBLES : le HTML contient l'image d'une copie d'élève. À générer
uniquement sur le SSP Cloud, à ne jamais committer ni sortir de l'environnement.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import pandas as pd
from PIL import Image

from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.data.loaders import load_image

_COULEUR_CODE = {
    "1": "#d7f0d7",  # correct → vert clair
    "9": "#f7d4d4",  # erreur → rouge clair
    "0": "#e0e0e0",  # absent → gris
    "?": "#fff3cd",  # non lu → jaune
}


def _img_base64(path: str, max_width: int = 1100) -> str:
    """Charge l'image, la redimensionne si besoin, renvoie une data URL PNG.

    Args:
        path: chemin de l'image de la copie.
        max_width: largeur maximale ; l'image est réduite au-delà (ratio conservé).

    Returns:
        Une data URL `data:image/png;base64,...` prête à inliner dans le HTML.
    """
    img: Image.Image = load_image(path)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _chip(
    mot: str,
    code: str,
    transcription: str | None = None,
    comparaison: str | None = None,
    divergent: bool = False,
) -> str:
    """Rend une « puce » HTML pour un item : mot attendu + code, colorée.

    Args:
        mot: mot attendu (texte de référence de l'item).
        code: code attribué, qui détermine la couleur de fond.
        transcription: lecture du modèle, affichée si elle diffère du mot attendu.
        comparaison: commentaire du modèle, affiché s'il n'est pas « identique ».
        divergent: si True, bordure rouge signalant un désaccord expert/modèle.

    Returns:
        Le fragment HTML de la puce.
    """
    bg = _COULEUR_CODE.get(code, "#ffffff")
    bord = "2px solid #b00020" if divergent else "1px solid #ccc"
    trans = ""
    if transcription is not None and transcription.strip() not in ("", mot):
        trans = f"<div class='trans'>lu : « {html.escape(transcription)} »</div>"
    comp = ""
    if (
        comparaison is not None
        and comparaison.strip()
        and comparaison.strip().lower() != "identique"
    ):
        comp = f"<div class='comp'>→ {html.escape(comparaison)}</div>"
    return (
        f"<span class='chip' style='background:{bg};border:{bord}'>"
        f"<span class='mot'>{html.escape(mot)}</span>"
        f"<span class='code'>{html.escape(code)}</span>"
        f"{trans}{comp}</span>"
    )


def build_copy_comparison(
    copy_id: str,
    image_path: str,
    grid_items: list[GridItem],
    expert_codes: dict[str, str],
    model_preds: dict[str, dict],
    raw_transcription: str | None = None,
) -> str:
    """Construit le HTML de comparaison pour une copie.

    Args:
        copy_id: identifiant de la copie.
        image_path: chemin de l'image numérisée de la copie.
        grid_items: items de la grille (mot attendu et item_id), dans l'ordre.
        expert_codes: code expert par item_id.
        model_preds: par item_id, dict {code, transcription, comparaison, ...}.
        raw_transcription: transcription brute de l'étape 1 (HTR), prioritaire
            si fournie sur la reconstitution par item.

    Returns:
        Le fragment HTML `<section>` de la copie (image, transcription, codages
        expert et modèle, badge d'accord).
    """
    img_data = _img_base64(image_path)

    n = len(grid_items)
    n_div = sum(
        1
        for it in grid_items
        if expert_codes.get(it.item_id) != model_preds.get(it.item_id, {}).get("code")
    )
    accord = (n - n_div) / n if n else 0.0

    chips_expert, chips_modele = [], []
    transcription_libre = []
    for it in grid_items:
        e = expert_codes.get(it.item_id, "?")
        m = model_preds.get(it.item_id, {})
        mcode = m.get("code", "?")
        mtrans = m.get("transcription")
        mcomp = m.get("comparaison")
        divergent = e != mcode
        chips_expert.append(_chip(it.attendu, e, divergent=divergent))
        chips_modele.append(
            _chip(
                it.attendu,
                mcode,
                transcription=mtrans,
                comparaison=mcomp,
                divergent=divergent,
            )
        )
        if mtrans:
            transcription_libre.append(mtrans)

    # Priorité à la transcription brute de l'étape 1 (VRAIE lecture du modèle) ;
    # à défaut on recolle les transcriptions par item.
    if raw_transcription and raw_transcription.strip():
        transcription_txt = html.escape(raw_transcription.strip())
        source_trans = "étape 1 (HTR)"
    else:
        transcription_txt = html.escape(" ".join(transcription_libre)) or "<em>(non fournie)</em>"
        source_trans = "reconstituée par item"

    return f"""
    <section class="copie">
      <h2>{html.escape(copy_id)}
        <span class="badge">accord {accord:.0%} · {n_div} désaccords / {n}</span>
      </h2>
      <div class="bloc">
        <h3>1 · Copie numérisée de l'élève</h3>
        <img src="{img_data}" alt="copie {html.escape(copy_id)}"/>
      </div>
      <div class="bloc">
        <h3>3 · Transcription par le modèle <small>({source_trans})</small></h3>
        <p class="transcription">{transcription_txt}</p>
      </div>
      <div class="bloc">
        <h3>2 · Codage de l'annotateur expert</h3>
        <div class="chips">{"".join(chips_expert)}</div>
      </div>
      <div class="bloc">
        <h3>4 · Codage par le modèle <small>(bordure rouge = désaccord avec l'expert)</small></h3>
        <div class="chips">{"".join(chips_modele)}</div>
      </div>
    </section>
    """


_PAGE_CSS = """
  body { font-family: system-ui, sans-serif; margin: 24px; color: #222; background:#fafafa; }
  h1 { color:#1f3864; }
  section.copie { background:#fff; border:1px solid #ddd; border-radius:8px;
                  padding:18px; margin-bottom:32px; }
  h2 { color:#2f5496; border-bottom:2px solid #eee; padding-bottom:6px; }
  .badge { font-size:0.6em; background:#eef; color:#2f5496; padding:3px 8px;
           border-radius:10px; margin-left:10px; font-weight:normal; }
  .bloc { margin:14px 0; }
  h3 { font-size:1em; color:#444; margin-bottom:6px; }
  h3 small { font-weight:normal; color:#888; }
  img { max-width:100%; border:1px solid #ccc; border-radius:4px; }
  .transcription { background:#f5f5f5; padding:10px; border-radius:4px;
                   font-family:Georgia,serif; line-height:1.5; }
  .chips { display:flex; flex-wrap:wrap; gap:4px; }
  .chip { display:inline-flex; flex-direction:column; align-items:center;
          padding:3px 6px; border-radius:5px; min-width:26px; }
  .chip .mot { font-size:0.9em; }
  .chip .code { font-size:0.7em; color:#555; font-weight:bold; }
  .chip .trans { font-size:0.65em; color:#b00020; margin-top:2px; }
  .chip .comp { font-size:0.65em; color:#1565c0; margin-top:1px; font-style:italic; }
  .legende span { padding:3px 8px; border-radius:4px; margin-right:8px; }
"""


def build_html_report(
    copies_html: list[str],
    title: str = "Diagnostic visuel — comparaison modèle / expert",
) -> str:
    """Assemble la page HTML complète à partir de fragments de copies.

    Args:
        copies_html: fragments HTML de copies (issus de `build_copy_comparison`).
        title: titre de la page (aussi utilisé dans l'en-tête).

    Returns:
        Le document HTML complet et autonome (CSS et légende inclus).
    """
    legende = (
        "<p class='legende'>Légende des codes : "
        "<span style='background:#d7f0d7'>1 correct</span>"
        "<span style='background:#f7d4d4'>9 erreur</span>"
        "<span style='background:#e0e0e0'>0 absent</span>"
        "<span style='background:#fff3cd'>? non lu</span></p>"
    )
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"/>
<title>{html.escape(title)}</title><style>{_PAGE_CSS}</style></head>
<body>
  <h1>{html.escape(title)}</h1>
  {legende}
  <p><strong>Données sensibles</strong> : ce fichier contient des copies d'élèves.
     Ne pas le sortir de l'environnement sécurisé.</p>
  {"".join(copies_html)}
</body></html>"""


def generate_comparison_report(
    copy_ids: list[str],
    predictions_df: pd.DataFrame,
    grid_items: list[GridItem],
    images_dir: str,
    expert_labels: dict[str, dict[str, str]],
    output_path: str | Path,
    scheme: str = "simplifiee",
) -> Path:
    """Génère le rapport HTML de comparaison pour une liste de copies.

    Args:
        copy_ids: copies à inclure dans le rapport.
        predictions_df: prédictions à l'item (codes, transcription, comparaison…).
        grid_items: items de la grille de la dictée, dans l'ordre.
        images_dir: dossier des images ; le chemin est `images_dir/<copy_id>`.
        expert_labels: codes experts bruts par copie, puis par item_id.
        output_path: chemin du fichier HTML à écrire.
        scheme: schéma de codage pour normaliser les codes experts (ex. « simplifiee »).

    Returns:
        Le chemin du fichier HTML écrit.
    """
    from evaluation_dictee.data.reference import normalize as _normalize

    has_trans = "transcription" in predictions_df.columns
    has_comp = "comparaison" in predictions_df.columns
    has_raw = "raw_transcription" in predictions_df.columns
    fragments = []
    for copy_id in copy_ids:
        sub = predictions_df[predictions_df["copy_id"] == copy_id]
        model_preds = {
            r["item_id"]: {
                "code": r["y_pred"],
                "transcription": r["transcription"] if has_trans else None,
                "comparaison": r["comparaison"] if has_comp else None,
                "confidence": r.get("confidence"),
            }
            for _, r in sub.iterrows()
        }
        # Transcription brute étape 1 : identique sur toutes les lignes de la copie
        raw_trans = None
        if has_raw and len(sub):
            vals = sub["raw_transcription"].dropna()
            raw_trans = vals.iloc[0] if len(vals) else None
        # Normaliser les codes experts bruts (ex. 3/4/5 → 9 en mode simplifié)
        raw_expert = expert_labels.get(copy_id, {})
        norm_expert = {iid: _normalize(code, scheme) for iid, code in raw_expert.items()}
        fragments.append(
            build_copy_comparison(
                copy_id=copy_id,
                image_path=images_dir.rstrip("/") + "/" + copy_id,
                grid_items=grid_items,
                expert_codes=norm_expert,
                model_preds=model_preds,
                raw_transcription=raw_trans,
            )
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html_report(fragments), encoding="utf-8")
    return out


def report_worst_copies(
    predictions_df: pd.DataFrame,
    grid_items: list[GridItem],
    images_dir: str,
    expert_labels: dict[str, dict[str, str]],
    output_path: str | Path,
    n: int = 10,
    scheme: str = "simplifiee",
) -> Path:
    """Génère le HTML diagnostic des N copies au plus fort taux de désaccord.

    Args:
        predictions_df: prédictions à l'item.
        grid_items: items de la grille de la dictée, dans l'ordre.
        images_dir: dossier des images.
        expert_labels: codes experts bruts par copie, puis par item_id.
        output_path: chemin du fichier HTML à écrire.
        n: nombre de copies (les pires) à inclure.
        scheme: schéma de codage pour normaliser les codes experts.

    Returns:
        Le chemin du fichier HTML écrit.
    """
    from evaluation_dictee.evaluation.report import copies_by_disagreement

    classement = copies_by_disagreement(predictions_df)
    pires = classement.head(n).index.tolist()
    return generate_comparison_report(
        copy_ids=pires,
        predictions_df=predictions_df,
        grid_items=grid_items,
        images_dir=images_dir,
        expert_labels=expert_labels,
        output_path=output_path,
        scheme=scheme,
    )


def report_single_copy(
    copy_id: str,
    predictions_df: pd.DataFrame,
    grid_items: list[GridItem],
    images_dir: str,
    expert_labels: dict[str, dict[str, str]],
    output_path: str | Path,
    scheme: str = "simplifiee",
) -> Path:
    """Génère le HTML diagnostic d'UNE copie identifiée par son copy_id.

    Args:
        copy_id: identifiant de la copie à diagnostiquer.
        predictions_df: prédictions à l'item.
        grid_items: items de la grille de la dictée, dans l'ordre.
        images_dir: dossier des images.
        expert_labels: codes experts bruts par copie, puis par item_id.
        output_path: chemin du fichier HTML à écrire.
        scheme: schéma de codage pour normaliser les codes experts.

    Returns:
        Le chemin du fichier HTML écrit.

    Raises:
        ValueError: si le copy_id est absent des prédictions.
    """
    if copy_id not in set(predictions_df["copy_id"]):
        dispo = sorted(set(predictions_df["copy_id"]))[:5]
        raise ValueError(
            f"Copie {copy_id!r} absente des prédictions. Exemples disponibles : {dispo}..."
        )
    return generate_comparison_report(
        copy_ids=[copy_id],
        predictions_df=predictions_df,
        grid_items=grid_items,
        images_dir=images_dir,
        expert_labels=expert_labels,
        output_path=output_path,
        scheme=scheme,
    )
