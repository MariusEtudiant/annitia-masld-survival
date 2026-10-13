"""
ANNITIA — Préparation des cibles de survie (censure droite).
MIT License.

Conventions du dataset
----------------------
- `Age_v1` : âge du patient à la première visite (baseline)
- `Age_vk` : âge à la visite k (NaN si visite absente)
- `evenements_hepatiques_majeurs` ∈ {0, 1}
- `evenements_hepatiques_age_occur` : âge à l'événement (NaN sinon)
- `death` ∈ {0, 1, NaN}   (NaN = statut inconnu -> à exclure)
- `death_age_occur`       : âge au décès (NaN sinon)

Temps de survie (en années d'âge)
---------------------------------
    t = age_at_event - Age_v1        si événement observé
    t = last_observed_age - Age_v1   sinon (censure droite)

`last_observed_age = max(Age_v1..Age_vK)` n'est utilisé QUE pour construire
le temps, jamais comme feature (proxy de la durée de suivi -> fuite).
"""
import numpy as np
import pandas as pd
from sksurv.util import Surv

from .config import TARGET_COLS


def compute_last_observed_age(df: pd.DataFrame) -> pd.Series:
    age_cols = [c for c in df.columns if c.startswith("Age_v")]
    assert age_cols, "Aucune colonne Age_v* trouvée."
    return df[age_cols].max(axis=1)


def prepare_survival_targets(df: pd.DataFrame, outcome: str = "hepatic"):
    """
    Paramètres
    ----------
    df      : DataFrame contenant Age_v1, Age_v*, et les 4 colonnes cibles.
    outcome : "hepatic" | "death"

    Retours
    -------
    df_valid : DataFrame filtré (patients utilisables).
    y        : structured array sksurv (event: bool, time: float).
    mask     : np.ndarray booléen de la même longueur que df original
               (True = patient conservé) — utile pour filtrer X en parallèle.
    """
    df = df.copy()
    df["_last_obs_age"] = compute_last_observed_age(df)

    if outcome == "hepatic":
        event_col, age_occur_col, name = (
            "evenements_hepatiques_majeurs",
            "evenements_hepatiques_age_occur",
            "Hepatic_event",
        )
        is_event = df[event_col] == 1
        # On garde les événement=0 ; pour event=1 on exige un âge d'occurrence.
        invalid = is_event & df[age_occur_col].isna()
        mask = ~invalid

    elif outcome == "death":
        event_col, age_occur_col, name = (
            "death", "death_age_occur", "Death",
        )
        is_event = df[event_col] == 1
        unknown = df[event_col].isna()
        invalid = is_event & df[age_occur_col].isna()
        mask = ~unknown & ~invalid

    else:
        raise ValueError(f"outcome must be 'hepatic' or 'death', got {outcome!r}")

    df_valid = df[mask].copy().reset_index(drop=True)
    is_event_v = (df_valid[event_col] == 1)

    t = np.where(
        is_event_v,
        df_valid[age_occur_col] - df_valid["Age_v1"],
        df_valid["_last_obs_age"] - df_valid["Age_v1"],
    ).astype(float)

    # sksurv exige t > 0 strictement.
    t = np.maximum(t, 0.001)

    # Sanity checks — anti-fuite
    assert np.all(np.isfinite(t)), "Temps non finis détectés."
    assert np.all(t >= 0), "Temps négatifs détectés."

    y = Surv.from_arrays(
        event=is_event_v.astype(bool).values,
        time=t,
        name_event=name,
        name_time="Time_years",
    )
    return df_valid, y, mask.values


def get_event_age(df: pd.DataFrame, outcome: str) -> pd.Series:
    """
    Retourne pour chaque patient l'âge de "fin d'observation utile" :
    - âge de l'événement si survenu
    - sinon last_observed_age
    Sert à masquer les visites postérieures (anti-fuite).
    """
    last_age = compute_last_observed_age(df)
    if outcome == "hepatic":
        age_event = df.get("evenements_hepatiques_age_occur")
        is_event = df.get("evenements_hepatiques_majeurs", pd.Series(0, index=df.index)) == 1
    elif outcome == "death":
        age_event = df.get("death_age_occur")
        is_event = df.get("death", pd.Series(0, index=df.index)) == 1
    else:
        return last_age
    # âge-limite = âge événement si event=1 et disponible, sinon last_age
    if age_event is None:
        return last_age
    cap = np.where(is_event & age_event.notna(), age_event, last_age)
    return pd.Series(cap, index=df.index)
