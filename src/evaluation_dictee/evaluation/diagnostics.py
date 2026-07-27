"""Diagnostic des écarts modèle/annotateur : d'où viennent les désaccords.

Pointe les copies à inspecter (effet de position, sur-confiance, décalages).
Complète report.py, qui agrège ; ici on cherche les causes.
"""

from __future__ import annotations

import pandas as pd


def add_position(df: pd.DataFrame, ordered_item_ids: list[str]) -> pd.DataFrame:
    """Ajoute les colonnes `position` (rang dans la dictée, 1..N) et `correct`.

    Args:
        df: prédictions à l'item (colonnes item_id, y_true, y_pred).
        ordered_item_ids: item_ids dans l'ordre de la dictée (définit le rang).

    Returns:
        Une copie du DataFrame avec `position` (rang de l'item) et `correct`
        (y_true == y_pred).
    """
    rang = {iid: i + 1 for i, iid in enumerate(ordered_item_ids)}
    out = df.copy()
    out["position"] = out["item_id"].map(rang)
    out["correct"] = out["y_true"] == out["y_pred"]
    return out


def accuracy_by_position(
    df: pd.DataFrame, ordered_item_ids: list[str], n_bins: int = 10
) -> pd.DataFrame:
    """Accord moyen par tranche de position dans la dictée.

    Une chute en fin de dictée signe un décalage propagé (oubli de mot, ligne sautée).

    Args:
        df: prédictions à l'item.
        ordered_item_ids: item_ids dans l'ordre de la dictée.
        n_bins: nombre de tranches de position.

    Returns:
        Un DataFrame, une ligne par tranche : position moyenne, accord et effectif.
    """
    d = add_position(df, ordered_item_ids)
    d["tranche"] = pd.cut(d["position"], bins=n_bins)
    g = d.groupby("tranche", observed=True).agg(
        position_moy=("position", "mean"),
        accord=("correct", "mean"),
        n=("correct", "size"),
    )
    return g.reset_index(drop=True)


def position_accuracy_correlation(df: pd.DataFrame, ordered_item_ids: list[str]) -> float:
    """Corrélation de Pearson position/justesse (négative = dégradation en fin).

    Args:
        df: prédictions à l'item.
        ordered_item_ids: item_ids dans l'ordre de la dictée.

    Returns:
        Le coefficient de corrélation de Pearson entre position et justesse.
    """
    d = add_position(df, ordered_item_ids)
    return float(d["position"].corr(d["correct"].astype(float)))


def per_copy_position_effect(df: pd.DataFrame, ordered_item_ids: list[str]) -> pd.DataFrame:
    """Compare, par copie, l'accord 1re vs 2de moitié (une chute signe un décalage en cours).

    Args:
        df: prédictions à l'item.
        ordered_item_ids: item_ids dans l'ordre de la dictée.

    Returns:
        Un DataFrame indexé par copy_id : accord de chaque moitié et `chute`
        (moitié 1 − moitié 2), trié par chute décroissante.
    """
    d = add_position(df, ordered_item_ids)
    milieu = len(ordered_item_ids) / 2
    rows = []
    for copy_id, grp in d.groupby("copy_id"):
        m1 = grp[grp["position"] <= milieu]["correct"].mean()
        m2 = grp[grp["position"] > milieu]["correct"].mean()
        rows.append(
            {
                "copy_id": copy_id,
                "accord_moitie1": m1,
                "accord_moitie2": m2,
                "chute": m1 - m2,
            }
        )
    return pd.DataFrame(rows).set_index("copy_id").sort_values("chute", ascending=False)


def longest_disagreement_run(df: pd.DataFrame, ordered_item_ids: list[str]) -> pd.DataFrame:
    """Plus longue séquence de désaccords CONSÉCUTIFS par copie.

    Erreurs isolées = vraies erreurs de codage ; longue séquence = signature d'un décalage.

    Args:
        df: prédictions à l'item.
        ordered_item_ids: item_ids dans l'ordre de la dictée.

    Returns:
        Un DataFrame indexé par copy_id : nombre de désaccords, longueur de la
        plus longue séquence consécutive et sa position de début, trié par
        longueur décroissante.
    """
    d = add_position(df, ordered_item_ids).sort_values(["copy_id", "position"])
    rows = []
    for copy_id, grp in d.groupby("copy_id"):
        faux = (~grp["correct"]).tolist()
        positions = grp["position"].tolist()
        best_len, best_start = 0, None
        cur_len, cur_start = 0, None
        for f, pos in zip(faux, positions, strict=True):
            if f:
                if cur_len == 0:
                    cur_start = pos
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        rows.append(
            {
                "copy_id": copy_id,
                "n_desaccords": int(sum(faux)),
                "plus_longue_sequence": best_len,
                "position_debut_sequence": best_start,
            }
        )
    return (
        pd.DataFrame(rows).set_index("copy_id").sort_values("plus_longue_sequence", ascending=False)
    )


def overconfident_errors(df: pd.DataFrame, seuil_confiance: float = 0.9) -> pd.DataFrame:
    """Prédictions FAUSSES annoncées avec une confiance élevée.

    Cas les plus dangereux : le renvoi humain ne les rattrape pas. En nombre =
    confiance inexploitable (ECE élevé).

    Args:
        df: prédictions à l'item (colonnes y_true, y_pred, confidence).
        seuil_confiance: seuil au-delà duquel une erreur est jugée sur-confiante.

    Returns:
        Les lignes fausses de confiance >= seuil, triées par confiance décroissante.
    """
    faux = df[df["y_true"] != df["y_pred"]]
    surconf = faux[faux["confidence"].fillna(0) >= seuil_confiance]
    return surconf.sort_values("confidence", ascending=False).reset_index(drop=True)


def disagreement_hotspots(
    df: pd.DataFrame, ordered_item_ids: list[str], top: int = 5
) -> dict[str, pd.DataFrame]:
    """Sélectionne des copies-types à inspecter (un palmarès par mécanisme d'erreur).

    Args:
        df: prédictions à l'item.
        ordered_item_ids: item_ids dans l'ordre de la dictée.
        top: nombre de copies retenues par palmarès.

    Returns:
        Un dict de DataFrames, un par mécanisme : `pire_accord`,
        `plus_longue_sequence`, `plus_forte_chute` et `sur_confiance`.
    """
    d = add_position(df, ordered_item_ids)

    accord_copie = (
        d.groupby("copy_id")["correct"].mean().sort_values().head(top).rename("accord").to_frame()
    )

    runs = longest_disagreement_run(df, ordered_item_ids).head(top)

    chute = per_copy_position_effect(df, ordered_item_ids).head(top)

    surconf = overconfident_errors(df)
    surconf_copie = (
        surconf.groupby("copy_id")
        .size()
        .sort_values(ascending=False)
        .head(top)
        .rename("n_erreurs_surconfiantes")
        .to_frame()
    )

    return {
        "pire_accord": accord_copie,
        "plus_longue_sequence": runs,
        "plus_forte_chute": chute,
        "sur_confiance": surconf_copie,
    }


def simulate_word_omission_shift(
    expert_codes: list[str],
    omission_position: int,
) -> dict[str, float]:
    """Teste l'effet d'un décalage dû à un mot oublié, sur des codes synthétiques.

    Outil pédagogique : démontre qu'un oubli non géré peut détruire l'évaluation
    de toute la fin de copie.

    Args:
        expert_codes: codes experts de référence.
        omission_position: index du mot supposé oublié (déclenche le décalage).

    Returns:
        Un dict : accord sans décalage, accord avec décalage, position de
        l'omission et nombre d'items situés après l'omission.
    """
    n = len(expert_codes)
    parfait = list(expert_codes)
    accord_parfait = sum(a == b for a, b in zip(expert_codes, parfait, strict=True)) / n

    # Modèle décalé : à partir de l'omission, tout glisse d'un cran
    decale = list(expert_codes[:omission_position])
    decale += list(expert_codes[omission_position + 1 :])
    decale += ["1"]
    decale = decale[:n]
    accord_decale = sum(a == b for a, b in zip(expert_codes, decale, strict=True)) / n

    return {
        "accord_sans_decalage": accord_parfait,
        "accord_avec_decalage": accord_decale,
        "position_omission": omission_position,
        "items_apres_omission": n - omission_position,
    }
