"""
ANNITIA — Feature engineering longitudinal (v3 — anti-overfitting).
MIT License.

Changements v3 vs v2 :
- SUPPRIMÉ : n_obs, n_visits_total, follow_up_duration, age_last_visit,
  *_coverage → fuites temporelles (corr=0.90 avec le temps de survie)
- AJOUTÉ : sélection automatique par univariate C-index
- AJOUTÉ : imputation cohérente (retourne les médianes du train)
- Log-transform APRÈS les scores dérivés (inchangé)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BIOMARKER_PREFIXES, CLINICAL_THRESHOLDS, STATIC_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _visit_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix + "_v")]
    cols.sort(key=lambda c: int(c.rsplit("_v", 1)[-1]))
    return cols


def _safe_slope(ages: np.ndarray, vals: np.ndarray) -> float:
    m = np.isfinite(ages) & np.isfinite(vals)
    if m.sum() < 2:
        return 0.0
    a, v = ages[m], vals[m]
    if np.ptp(a) < 1e-6:
        return 0.0
    return float(np.polyfit(a, v, 1)[0])


# ---------------------------------------------------------------------------
# Masquage post-événement
# ---------------------------------------------------------------------------
def mask_post_event_visits(df: pd.DataFrame, outcome: str,
                           buffer_days: float = 30.0) -> pd.DataFrame:
    """
    Masque uniquement les visites STRICTEMENT après l'événement
    pour les patients qui ont eu un événement.
    Pour les censurés : aucun masquage.
    Pour les événementiels : on autorise une marge de 30 jours
    avant l'événement pour exclure les mesures réactives.
    """
    from .targets import get_event_age, compute_last_observed_age

    if outcome not in {"hepatic", "death"}:
        return df
    if outcome == "hepatic":
        event_col = "evenements_hepatiques_majeurs"
        age_event_col = "evenements_hepatiques_age_occur"
    else:
        event_col = "death"
        age_event_col = "death_age_occur"

    if event_col not in df.columns:
        return df

    df = df.copy()
    is_event = (df[event_col] == 1) & df[age_event_col].notna()
    if not is_event.any():
        return df

    age_event = df[age_event_col]
    age_cols = [c for c in df.columns if c.startswith("Age_v")]
    if not age_cols:
        return df

    # marge de sécurité = 30 jours avant l'événement (en années)
    buffer_years = buffer_days / 365.25
    safe_cap = age_event - buffer_years

    for age_col in age_cols:
        suffix = age_col.replace("Age", "")
        ages = df[age_col]
        # On masque UNIQUEMENT pour les patients événementiels,
        # et UNIQUEMENT si la visite est >= cap (avec buffer).
        post_event = is_event & (ages >= safe_cap)
        post_event = post_event.fillna(False)
        if not post_event.any():
            continue
        for pref in BIOMARKER_PREFIXES:
            col = pref + suffix
            if col in df.columns:
                df.loc[post_event, col] = np.nan
        df.loc[post_event, age_col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Scores cliniques dérivés par visite (sur valeurs BRUTES)
# ---------------------------------------------------------------------------
def add_derived_visit_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    age_cols = _visit_cols(df, "Age")

    for age_col in age_cols:
        suf = age_col.replace("Age", "")
        alt = df.get(f"alt{suf}")
        ast = df.get(f"ast{suf}")
        plt_ = df.get(f"plt{suf}")
        age = df.get(age_col)
        bmi = df.get(f"BMI{suf}")
        t2dm = df.get("T2DM", 0)

        if alt is None or ast is None or plt_ is None or age is None:
            continue

        safe_alt = alt.clip(lower=1.0)
        safe_plt = plt_.clip(lower=1.0)

        df[f"fib4{suf}"] = (age * ast) / (safe_plt * np.sqrt(safe_alt))
        df[f"apri{suf}"] = (ast / 40.0) * 100.0 / safe_plt
        df[f"ast_alt{suf}"] = ast / safe_alt

        if bmi is not None:
            df[f"nfs{suf}"] = (-1.675 + (0.037 * age) + (0.094 * bmi) +
                               (1.13 * t2dm) + (0.99 * (ast / safe_alt)) -
                               (0.013 * plt_))

        ggt = df.get(f"ggt{suf}")
        if ggt is not None:
            df[f"ggt_plt{suf}"] = ggt / safe_plt

    return df


# ---------------------------------------------------------------------------
# Agrégation longitudinale SANS n_obs (anti-fuite)
# ---------------------------------------------------------------------------

def _aggregate_landmark_window(
    df: pd.DataFrame,
    prefix: str,
    age_values: np.ndarray,
    window_start: float,
    window_end: float,
    age_baseline: np.ndarray,
) -> pd.DataFrame:
    """
    Agrège un biomarqueur sur une fenêtre temporelle FIXE depuis Age_v1.
    Pour patient i, on garde les visites avec
        Age_v1[i] + window_start <= Age_vk[i] < Age_v1[i] + window_end
    Cela neutralise la confusion durée-de-suivi vs censoring.

    Retourne {prefix}_lm{ws}_{we}_{stat} pour stat in [last, mean, slope].
    """
    cols = _visit_cols(df, prefix)
    if not cols:
        return pd.DataFrame(index=df.index)
    V = df[cols].to_numpy(dtype=float)
    n, K_bio = V.shape
    K_age = age_values.shape[1]
    K = min(K_bio, K_age)

    # Pour chaque visite k, vérifier si elle tombe dans la fenêtre
    rel_ages = age_values[:, :K] - age_baseline[:, None]    # (n, K)
    in_window = (rel_ages >= window_start) & (rel_ages < window_end)

    # Masquer V hors fenêtre
    V_win = np.where(in_window, V[:, :K], np.nan)

    out = {}
    label = f"lm{int(window_start)}_{int(window_end)}"
    last = np.full(n, np.nan)
    mean = np.full(n, np.nan)
    slope = np.zeros(n)

    for i in range(n):
        valid = np.isfinite(V_win[i])
        if not valid.any():
            continue
        idx = np.where(valid)[0]
        last[i] = V_win[i, idx[-1]]
        mean[i] = np.nanmean(V_win[i])
        if valid.sum() >= 2:
            slope[i] = _safe_slope(rel_ages[i], V_win[i])

    out[f"{prefix}_{label}_last"]  = last
    out[f"{prefix}_{label}_mean"]  = mean
    out[f"{prefix}_{label}_slope"] = slope
    return pd.DataFrame(out, index=df.index)


# Dans build_features, après les agrégations classiques, ajouter :

def _aggregate_recent(
    df: pd.DataFrame,
    prefix: str,
    age_values: np.ndarray,
    n_recent: int = 3,        # 3 dernières visites valides
) -> pd.DataFrame:
    cols = _visit_cols(df, prefix)
    if not cols:
        return pd.DataFrame(index=df.index)
    V = df[cols].to_numpy(dtype=float)
    n, K_bio = V.shape
    K_age = age_values.shape[1]
    K = min(K_bio, K_age)
    A = age_values[:, :K]
    V_al = V[:, :K]

    recent_last  = np.full(n, np.nan)
    recent_mean  = np.full(n, np.nan)
    recent_slope = np.zeros(n)
    recent_max   = np.full(n, np.nan)

    for i in range(n):
        valid = np.isfinite(V_al[i])
        if not valid.any():
            continue
        idx = np.where(valid)[0]
        idx_recent = idx[-n_recent:]
        if len(idx_recent) == 0:
            continue
        recent_last[i] = V_al[i, idx_recent[-1]]
        recent_mean[i] = np.nanmean(V_al[i, idx_recent])
        recent_max[i]  = np.nanmax(V_al[i, idx_recent])
        if len(idx_recent) >= 2:
            recent_slope[i] = _safe_slope(A[i, idx_recent], V_al[i, idx_recent])

    return pd.DataFrame({
        f"{prefix}_recent3_last":  recent_last,
        f"{prefix}_recent3_mean":  recent_mean,
        f"{prefix}_recent3_slope": recent_slope,
        f"{prefix}_recent3_max":   recent_max,
    }, index=df.index)

def add_landmark_features(df, age_mat, age_baseline_arr):
    landmark_prefixes = [
        "fibs_stiffness_med_BM_1", "fibrotest_BM_2",
        "aixp_aix_result_BM_3", "fib4", "nfs",
        "alt", "ast", "ggt", "plt", "BMI", "gluc_fast",
    ]
    windows = [(0, 3), (0, 5), (3, 6), (5, 10)]   # années depuis baseline
    parts = []
    for pref in landmark_prefixes:
        for ws, we in windows:
            parts.append(_aggregate_landmark_window(
                df, pref, age_mat, ws, we, age_baseline_arr))
    return pd.concat(parts, axis=1)
def _aggregate_biomarker(
    df: pd.DataFrame,
    prefix: str,
    age_values: np.ndarray,
    threshold: float | None = None,
    full_stats: bool = False,
) -> pd.DataFrame:
    cols = _visit_cols(df, prefix)
    if not cols:
        return pd.DataFrame(index=df.index)
    V = df[cols].to_numpy(dtype=float)
    n, K_bio = V.shape
    K_age = age_values.shape[1]
    K = min(K_bio, K_age)
    V_al = V[:, :K]
    A_al = age_values[:, :K]

    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean   = np.nanmean(V, axis=1)
            std    = np.nanstd(V,  axis=1)
            mn     = np.nanmin(V,  axis=1)
            mx     = np.nanmax(V,  axis=1)
            median = np.nanmedian(V, axis=1)
            q25    = np.nanpercentile(V, 25, axis=1)
            q75    = np.nanpercentile(V, 75, axis=1)

    first = np.full(n, np.nan)
    last  = np.full(n, np.nan)
    slope = np.zeros(n)
    delta = np.full(n, np.nan)
    last_diff = np.zeros(n)
    rate_of_change = np.full(n, np.nan)
    ratio_last_mean = np.full(n, np.nan)

    for i in range(n):
        valid = np.isfinite(V[i])
        if not valid.any():
            continue
        idx_valid = np.where(valid)[0]
        first[i] = V[i, idx_valid[0]]
        last[i]  = V[i, idx_valid[-1]]
        delta[i] = last[i] - first[i]

        if abs(first[i]) > 1e-6:
            rate_of_change[i] = (last[i] - first[i]) / (abs(first[i]) + 1e-9)
        if abs(mean[i]) > 1e-6:
            ratio_last_mean[i] = last[i] / (mean[i] + 1e-9)
        if len(idx_valid) >= 2:
            last_diff[i] = V[i, idx_valid[-1]] - V[i, idx_valid[-2]]

        slope[i] = _safe_slope(A_al[i], V_al[i])

    # Features de base — SANS n_obs (supprimé : fuite temporelle)
    out = pd.DataFrame({
        f"{prefix}_last":      last,
        f"{prefix}_first":     first,
        f"{prefix}_mean":      mean,
        f"{prefix}_std":       std,
        f"{prefix}_slope":     slope,
        f"{prefix}_delta":     delta,
        f"{prefix}_last_diff": last_diff,
    }, index=df.index)

    if full_stats:
        out[f"{prefix}_min"]             = mn
        out[f"{prefix}_max"]             = mx
        out[f"{prefix}_median"]          = median
        out[f"{prefix}_q25"]             = q25
        out[f"{prefix}_q75"]             = q75
        out[f"{prefix}_iqr"]             = q75 - q25
        out[f"{prefix}_cv"]              = np.where(np.abs(mean) > 1e-6, std / (np.abs(mean) + 1e-9), 0)
        out[f"{prefix}_rate_of_change"]  = rate_of_change
        out[f"{prefix}_ratio_last_mean"] = ratio_last_mean

    if threshold is not None:
        above = (V > threshold).astype(float)
        above[~np.isfinite(V)] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out[f"{prefix}_frac_above_{threshold}"] = np.nanmean(above, axis=1)

    return out


# Biomarqueurs avec stats enrichies
FULL_STATS_PREFIXES = {
    "fibs_stiffness_med_BM_1", "fibrotest_BM_2", "fib4", "nfs",
    "BMI", "plt", "ggt", "bilirubin", "ast_alt",
}


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------
def _compute_interaction_features(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    age_bl = df["Age_v1"].values if "Age_v1" in df.columns else np.zeros(len(X))

    if "fibs_stiffness_med_BM_1_last" in X.columns:
        out["age_x_stiffness"] = age_bl * X["fibs_stiffness_med_BM_1_last"].values
    if "fibrotest_BM_2_last" in X.columns:
        out["age_x_fibrotest"] = age_bl * X["fibrotest_BM_2_last"].values
    if "fib4_last" in X.columns:
        out["age_x_fib4"] = age_bl * X["fib4_last"].values
    if "BMI_last" in X.columns and "fib4_last" in X.columns:
        out["bmi_x_fib4"] = X["BMI_last"].values * X["fib4_last"].values
    if "T2DM" in X.columns and "nfs_last" in X.columns:
        out["t2dm_x_nfs"] = X["T2DM"].values * X["nfs_last"].values

    # Concordance NITs
    if "fibs_stiffness_med_BM_1_last" in X.columns and "fibrotest_BM_2_last" in X.columns:
        out["nit_concordance_last"] = X["fibs_stiffness_med_BM_1_last"].values * X["fibrotest_BM_2_last"].values
    if "fibs_stiffness_med_BM_1_slope" in X.columns and "fibrotest_BM_2_slope" in X.columns:
        out["nit_slope_concordance"] = X["fibs_stiffness_med_BM_1_slope"].values * X["fibrotest_BM_2_slope"].values

    # FIB-4 acceleration
    if "fib4_slope" in X.columns and "fib4_last" in X.columns:
        out["fib4_acceleration"] = X["fib4_slope"].values * X["fib4_last"].values

    # Seuils binaires cliniques
    if "fibs_stiffness_med_BM_1_last" in X.columns:
        out["stiffness_last_above_12"] = (X["fibs_stiffness_med_BM_1_last"].values > 12).astype(float)
        out["stiffness_last_above_8"] = (X["fibs_stiffness_med_BM_1_last"].values > 8).astype(float)
    if "fibrotest_BM_2_last" in X.columns:
        out["fibrotest_last_above_058"] = (X["fibrotest_BM_2_last"].values > 0.58).astype(float)
    if "fib4_last" in X.columns:
        out["fib4_last_above_267"] = (X["fib4_last"].values > 2.67).astype(float)
        out["fib4_last_above_132"] = (X["fib4_last"].values > 1.32).astype(float)

    return pd.DataFrame(out, index=X.index)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    outcome: str | None = None,
    mask_post_event: bool = True,
) -> pd.DataFrame:
    df = df.copy()

    # 1) Masquage anti-fuite
    if mask_post_event and outcome is not None and \
       "evenements_hepatiques_majeurs" in df.columns:
        df = mask_post_event_visits(df, outcome)

    # 2) Scores cliniques (AVANT log-transform)
    df = add_derived_visit_scores(df)

    # 3) Log-transform APRÈS les scores dérivés
    for p in ["alt", "ast", "ggt", "bilirubin", "triglyc"]:
        cols = [c for c in df.columns if c.startswith(p + "_v")]
        if cols:
            df[cols] = np.log1p(df[cols].clip(lower=0))

    # 4) Matrice des âges
    age_cols = _visit_cols(df, "Age")
    age_mat  = df[age_cols].to_numpy(dtype=float)

    # 5) Agrégations
    all_prefixes = list(BIOMARKER_PREFIXES) + ["fib4", "apri", "ast_alt", "nfs", "ggt_plt"]
    parts = []
    for pref in all_prefixes:
        thr = CLINICAL_THRESHOLDS.get(pref)
        full = pref in FULL_STATS_PREFIXES
        parts.append(_aggregate_biomarker(df, pref, age_mat, threshold=thr, full_stats=full))
    X_long = pd.concat(parts, axis=1)
    # 5b) features landmark à temps fixe (neutralisent la durée de suivi)
    age_baseline_arr = df["Age_v1"].to_numpy()
    X_landmark = add_landmark_features(df, age_mat, age_baseline_arr)
 

    # 6) Features statiques
    statics = [c for c in STATIC_COLS if c in df.columns]
    X_stat  = df[statics].copy()
    X_stat["age_baseline"] = df.get("Age_v1", pd.Series(np.nan, index=df.index))
    if "BMI_v1" in df.columns:
        X_stat["bmi_baseline"] = df["BMI_v1"]

    # 7) Concaténation (PAS de n_visits, follow_up, age_last, n_obs, coverage)
    X = pd.concat([
        X_stat.reset_index(drop=True),
        X_long.reset_index(drop=True),
        X_landmark.reset_index(drop=True),
    ], axis=1)

    # 8) Interactions
    X_interact = _compute_interaction_features(X, df)
    X = pd.concat([X, X_interact.reset_index(drop=True)], axis=1)

    # 9) Nettoyage
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


# ---------------------------------------------------------------------------
# Imputation cohérente train/test
# ---------------------------------------------------------------------------
def impute_consistent(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Calcule les médianes sur train et les applique aux deux."""
    medians = X_train.median()
    X_train = X_train.fillna(medians).fillna(0)
    X_test  = X_test.fillna(medians).fillna(0)
    return X_train, X_test, medians


# ---------------------------------------------------------------------------
# Sélection de features par univariate C-index
# ---------------------------------------------------------------------------
def select_features_by_cindx(
    X: pd.DataFrame,
    t: np.ndarray,
    e: np.ndarray,
    min_abs_delta: float = 0.04,
) -> list[str]:
    """
    Sélectionne les features avec un C-index univarié significativement ≠ 0.5.
    Exclut les features qui sont du bruit pur.
    """
    from sksurv.metrics import concordance_index_censored

    selected = []
    for col in X.columns:
        try:
            vals = X[col].values
            if np.std(vals) < 1e-8:  # constante
                continue
            c, *_ = concordance_index_censored(e.astype(bool), t, vals)
            if abs(c - 0.5) >= min_abs_delta:
                selected.append(col)
        except Exception:
            pass

    return selected


# ---------------------------------------------------------------------------
# Alignement train / test
# ---------------------------------------------------------------------------
def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    all_cols = sorted(set(X_train.columns) | set(X_test.columns))
    for c in all_cols:
        if c not in X_train.columns:
            X_train[c] = np.nan
        if c not in X_test.columns:
            X_test[c] = np.nan
    return X_train[all_cols].copy(), X_test[all_cols].copy()
