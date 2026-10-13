"""
ANNITIA — Checks anti-fuite à lancer avant l'entraînement.
MIT License.
"""
import numpy as np
import pandas as pd

from .config import TARGET_COLS, ID_COLS


def check_no_target_in_features(X: pd.DataFrame, name: str = "X"):
    leaked = [c for c in X.columns if c in TARGET_COLS + ID_COLS]
    assert not leaked, f"[LEAK] {name} contient des colonnes cibles/id : {leaked}"


def check_no_last_observed_age(X: pd.DataFrame, name: str = "X"):
    bad = [c for c in X.columns if c.lower() in {"last_observed_age", "_last_obs_age"}]
    assert not bad, f"[LEAK] {name} contient last_observed_age : {bad}"


def check_no_age_vk_columns(X: pd.DataFrame, name: str = "X"):
    """Age_v1..Age_v22 encodent le calendrier des visites et sont utilisés
    pour le calcul des slopes UNIQUEMENT ; ils ne doivent pas rester dans X."""
    bad = [c for c in X.columns if c.startswith("Age_v")]
    assert not bad, f"[LEAK] {name} contient des Age_v* : {bad[:5]}..."


def check_positive_times(t: np.ndarray):
    assert np.all(t > 0), "[LEAK] temps de survie <= 0 détectés."
    assert np.all(np.isfinite(t)), "[LEAK] temps non finis détectés."


def check_column_alignment(X_train: pd.DataFrame, X_test: pd.DataFrame):
    a, b = list(X_train.columns), list(X_test.columns)
    assert a == b, (
        f"[MISMATCH] colonnes train/test différentes.\n"
        f"  train only: {set(a) - set(b)}\n  test only: {set(b) - set(a)}"
    )


def run_all_checks(X_train, X_test, t_train):
    check_no_target_in_features(X_train, "X_train")
    check_no_target_in_features(X_test,  "X_test")
    check_no_last_observed_age(X_train,  "X_train")
    check_no_last_observed_age(X_test,   "X_test")
    check_no_age_vk_columns(X_train,     "X_train")
    check_no_age_vk_columns(X_test,      "X_test")
    check_positive_times(t_train)
    check_column_alignment(X_train, X_test)
    print("[OK] Tous les checks anti-fuite sont passés.")
