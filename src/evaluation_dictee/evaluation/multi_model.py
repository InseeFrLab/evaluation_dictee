"""Comparaison multi-modèles pour construire un score de confiance par copie.

Le désaccord inter-modèles prédit bien mieux l'incertitude que la confiance d'un
modèle unique (quasi-constante à 1.0 dans nos benchmarks, inexploitable). Marche
de N=1 à N quelconque.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from evaluation_dictee.evaluation.report import load_predictions


def load_multi_runs(
    run_names: list[str],
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Charge N runs indépendants et les joint sur (copy_id, item_id).

    Le premier run de la liste fournit les colonnes communes (y_true, etc.).

    Args:
        run_names: noms des runs à charger (fichiers `<run>_predictions.jsonl`).
        output_dir: dossier contenant les fichiers de prédictions.

    Returns:
        Un DataFrame joint (inner) sur (copy_id, item_id), avec y_true et une
        paire de colonnes `y_pred__<run>` / `conf__<run>` par run.

    Raises:
        ValueError: si run_names est vide ou si aucun item n'est commun aux runs.
        FileNotFoundError: si un fichier de prédictions est absent.
    """
    output_dir = Path(output_dir)
    if not run_names:
        raise ValueError("Au moins un run est requis.")

    dfs = {}
    for run in run_names:
        path = output_dir / f"{run}_predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Prédictions manquantes : {path}")
        df_run = load_predictions(path)
        renamed = df_run.rename(columns={"y_pred": f"y_pred__{run}", "confidence": f"conf__{run}"})
        dfs[run] = renamed

    ref = dfs[run_names[0]][["copy_id", "item_id", "y_true"]].copy()

    merged = ref
    for run in run_names:
        cols = ["copy_id", "item_id", f"y_pred__{run}", f"conf__{run}"]
        merged = merged.merge(
            dfs[run][cols],
            on=["copy_id", "item_id"],
            how="inner",  # ne garde que les (copy_id, item_id) présents PARTOUT
        )

    if merged.empty:
        raise ValueError(
            "Aucun (copy_id, item_id) commun aux runs. Vérifier qu'ils portent "
            "sur les mêmes données."
        )
    return merged


def agreement_per_item(df_multi: pd.DataFrame) -> pd.DataFrame:
    """Ajoute au DataFrame multi-runs les colonnes de désaccord par item.

    Args:
        df_multi: DataFrame issu de `load_multi_runs` (colonnes `y_pred__<run>`).

    Returns:
        Une copie du DataFrame enrichie de : `modal_pred` (code majoritaire),
        `n_accord_modeles`, `n_modeles`, `unanimite`, et un booléen
        `modele_vs_expert__<run>` par run (prédiction == expert).
    """
    pred_cols = [c for c in df_multi.columns if c.startswith("y_pred__")]
    n_modeles = len(pred_cols)

    out = df_multi.copy()

    def _modal_and_count(row):
        counts = Counter(row[c] for c in pred_cols)
        modal, n_acc = counts.most_common(1)[0]
        return pd.Series({"modal_pred": modal, "n_accord_modeles": n_acc})

    modal_df = out[pred_cols].apply(_modal_and_count, axis=1)
    out["modal_pred"] = modal_df["modal_pred"]
    out["n_accord_modeles"] = modal_df["n_accord_modeles"].astype(int)
    out["n_modeles"] = n_modeles
    out["unanimite"] = out["n_accord_modeles"] == n_modeles

    for c in pred_cols:
        run = c.removeprefix("y_pred__")
        out[f"modele_vs_expert__{run}"] = out[c] == out["y_true"]

    return out


def confidence_score(df_agree: pd.DataFrame) -> pd.DataFrame:
    """Agrège au niveau copie un score de confiance basé sur le désaccord.

    Args:
        df_agree: DataFrame issu de `agreement_per_item`.

    Returns:
        Un DataFrame indexé par copy_id : effectif, part d'items unanimes,
        accord inter-modèles moyen, nombre d'items sous la majorité et
        `score_confiance` (part d'items unanimes), trié par score décroissant.
    """
    n_modeles = int(df_agree["n_modeles"].iloc[0])
    rows = []
    for copy_id, grp in df_agree.groupby("copy_id"):
        n = len(grp)
        n_unan = int(grp["unanimite"].sum())
        accord_moy = float(grp["n_accord_modeles"].mean())
        # Items où l'accord inter-modèles est en dessous de la majorité (dispersion max)
        seuil_min = (n_modeles // 2) + 1
        n_dis_max = int((grp["n_accord_modeles"] < seuil_min).sum()) if n_modeles > 1 else 0
        rows.append(
            {
                "copy_id": copy_id,
                "n_items": n,
                "pct_unanime": n_unan / n * 100 if n else 0.0,
                "accord_moyen_mod": accord_moy,
                "n_desaccord_max": n_dis_max,
                "score_confiance": n_unan / n * 100 if n else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("copy_id").sort_values("score_confiance", ascending=False)


def referral_curve_multi(
    df_agree: pd.DataFrame,
    conf: pd.DataFrame,
    reference_run: str,
) -> pd.DataFrame:
    """Courbe de renvoi humain : renvoie les copies dont le score < τ, mesure l'accord retenu.

    Args:
        df_agree: DataFrame issu de `agreement_per_item` (contient le run de référence).
        conf: scores de confiance par copie issus de `confidence_score`.
        reference_run: run dont l'accord avec l'expert sert de référence.

    Returns:
        Un DataFrame, une ligne par seuil τ (0 à 100 par pas de 5) : part de
        copies renvoyées, accord des copies retenues, nombre de copies retenues.

    Raises:
        ValueError: si le run de référence est absent de df_agree.
    """
    pred_col = f"y_pred__{reference_run}"
    if pred_col not in df_agree.columns:
        raise ValueError(
            f"Run de référence {reference_run!r} absent de df_agree. "
            f"Colonnes disponibles : {[c for c in df_agree.columns if c.startswith('y_pred__')]}"
        )

    copies = conf.copy()
    accord_par_copie = df_agree.groupby("copy_id").apply(
        lambda g: (g[pred_col] == g["y_true"]).mean() * 100
    )
    copies["accord_modele_expert"] = accord_par_copie

    n_total = len(copies)
    rows = []
    for tau in range(0, 101, 5):
        retenues = copies[copies["score_confiance"] >= tau]
        renvoyees = copies[copies["score_confiance"] < tau]
        n_ret = len(retenues)
        pct_renv = len(renvoyees) / n_total * 100 if n_total else 0.0
        acc_ret = float(retenues["accord_modele_expert"].mean()) if n_ret else float("nan")
        rows.append(
            {
                "seuil_confiance": tau,
                "pct_copies_renvoyees": pct_renv,
                "pct_accord_retenues": acc_ret,
                "n_copies_retenues": n_ret,
            }
        )
    return pd.DataFrame(rows)


def disagreement_type_summary(df_agree: pd.DataFrame) -> pd.DataFrame:
    """Répartition des items selon le niveau d'accord inter-modèles.

    Args:
        df_agree: DataFrame issu de `agreement_per_item`.

    Returns:
        Un DataFrame indexé par `n_accord_modeles` : effectif, part des items et
        accord du code modal avec l'expert, pour chaque niveau d'accord observé.
    """
    n_modeles = int(df_agree["n_modeles"].iloc[0])
    n_total = len(df_agree)
    rows = []
    for n_acc in range(1, n_modeles + 1):
        sub = df_agree[df_agree["n_accord_modeles"] == n_acc]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "n_accord_modeles": n_acc,
                "n_items": len(sub),
                "pct_items": len(sub) / n_total * 100,
                "accord_avec_expert": float((sub["modal_pred"] == sub["y_true"]).mean() * 100),
            }
        )
    return pd.DataFrame(rows).set_index("n_accord_modeles")


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de Wilson (pourcentages). Utilisé pour l'accord retenu.

    Args:
        k: nombre de succès.
        n: nombre total d'observations.
        z: quantile normal (1.96 pour un niveau de 95 %).

    Returns:
        Les bornes basse et haute de l'intervalle, en pourcentages, ou (NaN, NaN)
        si n vaut 0.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    return (max(0.0, centre - margin) * 100, min(1.0, centre + margin) * 100)


def referral_curve_with_ci(
    df_agree: pd.DataFrame,
    conf: pd.DataFrame,
    reference_run: str,
) -> pd.DataFrame:
    """Comme `referral_curve_multi`, avec IC Wilson sur l'accord retenu.

    Args:
        df_agree: DataFrame issu de `agreement_per_item` (contient le run de référence).
        conf: scores de confiance par copie issus de `confidence_score`.
        reference_run: run dont l'accord avec l'expert sert de référence.

    Returns:
        Un DataFrame, une ligne par seuil τ (0 à 100 par pas de 5) : part de
        copies renvoyées, accord des items retenus avec son IC Wilson, nombre de
        copies et d'items retenus.

    Raises:
        ValueError: si le run de référence est absent de df_agree.
    """
    pred_col = f"y_pred__{reference_run}"
    if pred_col not in df_agree.columns:
        raise ValueError(f"Run {reference_run!r} absent.")

    copies_conf = conf["score_confiance"].to_dict()
    df = df_agree.copy()
    df["_correct"] = (df[pred_col] == df["y_true"]).astype(int)
    df["_score"] = df["copy_id"].map(copies_conf)

    n_copies_total = df["copy_id"].nunique()
    rows = []
    for tau in range(0, 101, 5):
        retenues_items = df[df["_score"] >= tau]
        n_items_ret = len(retenues_items)
        n_copies_ret = retenues_items["copy_id"].nunique()
        k_ok = int(retenues_items["_correct"].sum())
        lo, hi = _wilson_ci(k_ok, n_items_ret)
        rows.append(
            {
                "seuil_confiance": tau,
                "pct_copies_renvoyees": (1 - n_copies_ret / n_copies_total) * 100
                if n_copies_total
                else 0.0,
                "pct_accord_retenues": (k_ok / n_items_ret * 100) if n_items_ret else float("nan"),
                "accord_lo": lo,
                "accord_hi": hi,
                "n_copies_retenues": n_copies_ret,
                "n_items_retenus": n_items_ret,
            }
        )
    return pd.DataFrame(rows)
