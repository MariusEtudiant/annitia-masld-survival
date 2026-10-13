"""
ANNITIA — Modèles de survie (v3 — anti-overfitting).
MIT License.

Changements v3 :
- Poids d'ensemble optimisés GLOBALEMENT sur tous les OOF (pas par fold)
- Modèles plus simples (max_depth réduit, plus de régularisation)
- Multi-seed avec 5 seeds
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from .config import SEED, N_SPLITS


# ---------------------------------------------------------------------------
# GBSA — plus conservateur pour généraliser
# ---------------------------------------------------------------------------
def fit_gbsa(X_tr, t_tr, e_tr, params=None, seed=SEED):
    default = dict(
        loss="coxph",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=25,
        min_samples_split=40,
        subsample=0.7,
        max_features="sqrt",
        random_state=seed,
    )
    if params:
        default.update(params)
    y = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("gb",  GradientBoostingSurvivalAnalysis(**default)),
    ])
    pipe.fit(X_tr, y)
    return pipe


def gbsa_predict(pipe, X):
    return pipe.predict(X)


# ---------------------------------------------------------------------------
# XGBoost Cox — plus régularisé
# ---------------------------------------------------------------------------
def fit_xgb_cox(X_tr, t_tr, e_tr, X_val=None, t_val=None, e_val=None, params=None, seed=42):
    t_safe = np.maximum(t_tr, 1e-3)
    y_cox = np.where(e_tr == 1, t_safe, -t_safe)
    dtrain = xgb.DMatrix(X_tr, label=y_cox)

    evals = [(dtrain, "train")]
    if X_val is not None:
        t_val_safe = np.maximum(t_val, 1e-3)
        y_val_cox = np.where(e_val == 1, t_val_safe, -t_val_safe)
        dval = xgb.DMatrix(X_val, label=y_val_cox)
        evals.append((dval, "val"))

    default_params = {
        "objective": "survival:cox",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 2,
        "lambda": 8,
        "alpha": 3,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 8,
        "verbosity": 0,
        "seed": seed,
    }
    if params:
        default_params.update(params)

    bst = xgb.train(
        default_params, dtrain,
        num_boost_round=400,
        evals=evals,
        early_stopping_rounds=50 if X_val is not None else None,
        verbose_eval=False,
    )
    return bst


def xgb_predict_risk(model, X):
    return model.predict(xgb.DMatrix(X))


# ---------------------------------------------------------------------------
# Coxnet
# ---------------------------------------------------------------------------
def fit_coxnet(X_tr, t_tr, e_tr, l1_ratio: float = 0.5, seed: int = SEED):
    """Coxnet avec sélection d'alpha par CV interne 3-fold."""
    from sklearn.model_selection import StratifiedKFold
    y = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr)

    # 1. Path d'alphas
    pipe_path = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("cox", CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio, alpha_min_ratio=0.01,
            max_iter=10000, fit_baseline_model=False,
        )),
    ])
    try:
        pipe_path.fit(X_tr, y)
    except Exception as exc:
        return None
    alphas = pipe_path.named_steps["cox"].alphas_

    # 2. CV interne pour choisir alpha
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cv_scores = np.zeros(len(alphas))
    for tr, va in inner_cv.split(X_tr, e_tr):
        try:
            sub = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
                ("cox", CoxnetSurvivalAnalysis(
                    l1_ratio=l1_ratio, alphas=alphas,
                    max_iter=10000, fit_baseline_model=False)),
            ])
            sub.fit(X_tr.iloc[tr],
                    Surv.from_arrays(event=e_tr[tr].astype(bool), time=t_tr[tr]))
            for i, a in enumerate(alphas):
                pred = sub.predict(X_tr.iloc[va], alpha=a)
                c, *_ = concordance_index_censored(
                    e_tr[va].astype(bool), t_tr[va], pred)
                cv_scores[i] += c / 3
        except Exception:
            pass
    best_alpha_idx = int(np.argmax(cv_scores))
    pipe_path.best_alpha_idx_ = best_alpha_idx
    return pipe_path


def coxnet_predict(pipe, X, alpha_idx=None):
    if pipe is None:
        return None
    cox = pipe.named_steps["cox"]
    alphas = cox.alphas_
    idx = alpha_idx if alpha_idx is not None else getattr(pipe, "best_alpha_idx_", len(alphas)//3)
    return pipe.predict(X, alpha=alphas[idx])


def coxnet_predict(pipe, X, alpha_idx: int | None = None) -> np.ndarray:
    if pipe is None:
        return None
    cox = pipe.named_steps["cox"]
    alphas = cox.alphas_
    if alpha_idx is None:
        alpha_idx = max(0, int(len(alphas) * 0.3))
    return pipe.predict(X, alpha=alphas[alpha_idx])


# ---------------------------------------------------------------------------
# RSF — plus conservateur
# ---------------------------------------------------------------------------
def fit_rsf(X_tr, t_tr, e_tr, params=None, seed=SEED):
    default = dict(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=20,
        min_samples_split=30,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    if params:
        default.update(params)
    y = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rsf", RandomSurvivalForest(**default)),
    ])
    pipe.fit(X_tr, y)
    return pipe


def rsf_predict(pipe, X):
    return pipe.predict(X)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def _rank_normalize(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order / max(1, len(x) - 1)


def _optimize_weights_global(
    oof_preds: dict[str, np.ndarray],
    e: np.ndarray,
    t: np.ndarray,
) -> dict[str, float]:
    """
    Optimise les poids UNE SEULE FOIS sur TOUS les OOF.
    Beaucoup plus stable que l'optimisation par fold.
    """
    model_names = list(oof_preds.keys())
    n = len(model_names)
    ranked = {k: _rank_normalize(v) for k, v in oof_preds.items()}

    best_weights = np.ones(n) / n
    best_c = 0.0
    steps = np.arange(0, 1.05, 0.05)

    # Coordinate descent - 5 passes pour convergence
    weights = np.ones(n) / n
    for _pass in range(5):
        for i in range(n):
            best_w_i = weights[i]
            for w in steps:
                test_w = weights.copy()
                test_w[i] = w
                s = test_w.sum()
                if s < 1e-6:
                    continue
                test_w /= s
                r = sum(test_w[j] * ranked[model_names[j]] for j in range(n))
                c, *_ = concordance_index_censored(e.astype(bool), t, r)
                if c > best_c:
                    best_c = c
                    best_w_i = w
                    best_weights = test_w.copy()
            weights[i] = best_w_i
        s = weights.sum()
        if s > 0:
            weights /= s

    return {model_names[i]: float(best_weights[i]) for i in range(n)}, best_c


# ---------------------------------------------------------------------------
# CV stratifié + ensemble avec poids GLOBAUX
# ---------------------------------------------------------------------------
def stratified_cv_ensemble(
    X: pd.DataFrame,
    t: np.ndarray,
    e: np.ndarray,
    X_test: pd.DataFrame,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    verbose: bool = True,
):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    n_train = len(X)
    n_test = len(X_test)

    # Collecter TOUTES les prédictions OOF par modèle
    oof_gbsa = np.zeros(n_train)
    oof_xgb  = np.zeros(n_train)
    oof_cox  = np.zeros(n_train)
    oof_rsf  = np.zeros(n_train)
    cox_valid = True

    # Prédictions test accumulées par fold
    test_gbsa = np.zeros(n_test)
    test_xgb  = np.zeros(n_test)
    test_cox  = np.zeros(n_test)
    test_rsf  = np.zeros(n_test)

    scores_per_model = {"gbsa": [], "xgb": [], "cox": [], "rsf": [], "ensemble": []}

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, e)):
        rng = np.random.default_rng(seed + fold)
        tr_events = e[tr_idx]
        idx_pos = tr_idx[tr_events == 1]
        idx_neg = tr_idx[tr_events == 0]
        rng.shuffle(idx_pos); rng.shuffle(idx_neg)
        n_val_pos = max(2, int(len(idx_pos) * 0.15))
        n_val_neg = max(5, int(len(idx_neg) * 0.15))
        val_idx = np.concatenate([idx_pos[:n_val_pos], idx_neg[:n_val_neg]])
        tr_only = np.setdiff1d(tr_idx, val_idx)

        X_tr, X_val, X_te = X.iloc[tr_only], X.iloc[val_idx], X.iloc[te_idx]
        t_tr, t_val, t_te = t[tr_only], t[val_idx], t[te_idx]
        e_tr, e_val, e_te = e[tr_only], e[val_idx], e[te_idx]

        X_tr_full = pd.concat([X_tr, X_val])
        t_tr_full = np.concatenate([t_tr, t_val])
        e_tr_full = np.concatenate([e_tr, e_val])

        # Entraîner les modèles
        gbsa_m = fit_gbsa(X_tr_full, t_tr_full, e_tr_full, seed=seed + fold)
        xgb_m  = fit_xgb_cox(X_tr, t_tr, e_tr, X_val, t_val, e_val, seed=seed + fold)
        cox_m  = fit_coxnet(X_tr_full, t_tr_full, e_tr_full, seed=seed + fold)
        rsf_m  = fit_rsf(X_tr_full, t_tr_full, e_tr_full, seed=seed + fold)

        # Prédictions OOF
        oof_gbsa[te_idx] = gbsa_predict(gbsa_m, X_te)
        oof_xgb[te_idx]  = xgb_predict_risk(xgb_m, X_te)
        r_cox = coxnet_predict(cox_m, X_te)
        if r_cox is not None:
            oof_cox[te_idx] = r_cox
        else:
            cox_valid = False
        oof_rsf[te_idx]  = rsf_predict(rsf_m, X_te)

        # Prédictions test (moyennées sur folds)
        test_gbsa += gbsa_predict(gbsa_m, X_test) / n_splits
        test_xgb  += xgb_predict_risk(xgb_m, X_test) / n_splits
        rt_cox = coxnet_predict(cox_m, X_test)
        if rt_cox is not None:
            test_cox += rt_cox / n_splits
        test_rsf  += rsf_predict(rsf_m, X_test) / n_splits

        # Scores par fold
        c_gbsa, *_ = concordance_index_censored(e_te.astype(bool), t_te, oof_gbsa[te_idx])
        c_xgb,  *_ = concordance_index_censored(e_te.astype(bool), t_te, oof_xgb[te_idx])
        c_rsf,  *_ = concordance_index_censored(e_te.astype(bool), t_te, oof_rsf[te_idx])
        c_cox = np.nan
        if r_cox is not None:
            c_cox, *_ = concordance_index_censored(e_te.astype(bool), t_te, r_cox)
        scores_per_model["gbsa"].append(c_gbsa)
        scores_per_model["xgb"].append(c_xgb)
        scores_per_model["cox"].append(c_cox)
        scores_per_model["rsf"].append(c_rsf)

        if verbose:
            print(f"  Fold {fold+1}/{n_splits} | "
                  f"gbsa={c_gbsa:.4f}  xgb={c_xgb:.4f}  "
                  f"cox={c_cox:.4f}  rsf={c_rsf:.4f}")

    # --- Stacking meta-learner (remplace la pondération linéaire) ---
    oof_preds = {"gbsa": oof_gbsa, "xgb": oof_xgb, "rsf": oof_rsf}
    test_preds = {"gbsa": test_gbsa, "xgb": test_xgb, "rsf": test_rsf}
    if cox_valid:
        oof_preds["cox"] = oof_cox
        test_preds["cox"] = test_cox

    from .stacking import stack_predictions
    oof_ens, test_ens, c_oof, meta = stack_predictions(
        oof_preds, test_preds, e, t, ranked=True,
    )
    if verbose:
        print(f"  Stacking OOF C-index = {c_oof:.4f}")

    # Vérifier les scores par fold pour l'ensemble
    for fold, (_, te_idx) in enumerate(cv.split(X, e)):
        c_ens, *_ = concordance_index_censored(
            e[te_idx].astype(bool), t[te_idx], oof_ens[te_idx]
        )
        scores_per_model["ensemble"].append(c_ens)

    if verbose:
        print(f"\n  Stacking meta-learner appliqué.")
        print(f"  OOF global C-index = {c_oof:.4f}")
        for k, v in scores_per_model.items():
            print(f"  {k:>10s} mean = {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")

    return oof_ens, test_ens, c_oof, scores_per_model


# ---------------------------------------------------------------------------
# Multi-seed averaging
# ---------------------------------------------------------------------------
def multi_seed_ensemble(
    X: pd.DataFrame,
    t: np.ndarray,
    e: np.ndarray,
    X_test: pd.DataFrame,
    seeds: list[int] = None,
    n_splits: int = N_SPLITS,
    verbose: bool = True,
):
    if seeds is None:
        seeds = [42, 123, 456, 789, 2026]

    all_test_preds = []
    all_oof_scores = []

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'='*60}")
            print(f" Seed {seed} ({i+1}/{len(seeds)})")
            print(f"{'='*60}")

        _, test_pred, c_oof, _ = stratified_cv_ensemble(
            X, t, e, X_test, n_splits=n_splits, seed=seed, verbose=verbose,
        )
        all_test_preds.append(test_pred)
        all_oof_scores.append(c_oof)

    final_test = np.mean([_rank_normalize(p) for p in all_test_preds], axis=0)

    if verbose:
        print(f"\n  Multi-seed: mean OOF = {np.mean(all_oof_scores):.4f} "
              f"± {np.std(all_oof_scores):.4f}")

    return final_test, np.mean(all_oof_scores), all_oof_scores
