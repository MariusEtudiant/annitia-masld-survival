"""
ANNITIA — Explicabilité (SHAP + permutation) pour le scoring qualitatif.
MIT License.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

from .config import SEED, OUT


def fit_final_model(X, t, e, seed=SEED):
    """Ré-entraîne un GBSA sur tout le train pour l'analyse d'importance."""
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y = Surv.from_arrays(event=e.astype(bool), time=t)
    model = GradientBoostingSurvivalAnalysis(
        loss="coxph", n_estimators=250, learning_rate=0.05,
        max_depth=2, min_samples_leaf=25, min_samples_split=40,
        subsample=0.7, max_features="sqrt", random_state=seed,
    )
    model.fit(X_imp, y)
    return model, X_imp, y


def permutation_importance_survival(model, X_imp, t, e, n_repeats=3, seed=SEED,
                                    top_k: int = 60):
    """
    Permutation importance sur le C-index. Agnostique au modèle et fiable.
    Calculée seulement sur les `top_k` features les plus variables pour la vitesse.
    """
    rng = np.random.default_rng(seed)
    # focus sur les features non-constantes les plus variables
    variances = X_imp.var().sort_values(ascending=False)
    cols = variances.head(top_k).index.tolist()

    base_pred = model.predict(X_imp)
    base_c, *_ = concordance_index_censored(e.astype(bool), t, base_pred)

    results = []
    for col in cols:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_imp.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            pred = model.predict(X_perm)
            c, *_ = concordance_index_censored(e.astype(bool), t, pred)
            drops.append(base_c - c)
        results.append({"feature": col,
                        "mean_drop": float(np.mean(drops)),
                        "std_drop":  float(np.std(drops))})
    return pd.DataFrame(results).sort_values("mean_drop", ascending=False)


def shap_report(model, X_imp: pd.DataFrame, t, e, outcome: str,
                max_display: int = 20, sample: int = 300):
    """
    Produit :
      - permutation_importance_<outcome>.csv
      - permutation_importance_<outcome>.png (top 20)
      - shap_summary_<outcome>.png + .csv (si SHAP dispo)
    """
    print(f"[importance] permutation importance for {outcome}...")
    imp = permutation_importance_survival(model, X_imp, t, e)

    path_csv = OUT / f"permutation_importance_{outcome}.csv"
    imp.to_csv(path_csv, index=False)

    top = imp.head(20)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top["feature"][::-1], top["mean_drop"][::-1])
    ax.set_xlabel("Chute du C-index (moyenne sur 3 permutations)")
    ax.set_title(f"Permutation importance — {outcome}")
    plt.tight_layout()
    path_png = OUT / f"permutation_importance_{outcome}.png"
    plt.savefig(path_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[importance] {path_png.name} + {path_csv.name}")

    # --- SHAP (optionnel) ---
    try:
        import shap
        if len(X_imp) > sample:
            X_sample = X_imp.sample(sample, random_state=SEED).reset_index(drop=True)
        else:
            X_sample = X_imp.copy().reset_index(drop=True)
        # top 40 features pour limiter le coût
        cols = imp["feature"].head(40).tolist()
        explainer = shap.Explainer(model.predict, X_sample[cols])
        sv = explainer(X_sample[cols])
        fig = plt.figure(figsize=(8, 10))
        shap.summary_plot(sv, X_sample[cols], max_display=max_display, show=False)
        plt.title(f"SHAP — {outcome}")
        plt.tight_layout()
        plt.savefig(OUT / f"shap_summary_{outcome}.png",
                    dpi=120, bbox_inches="tight")
        plt.close()
        mean_abs = np.abs(sv.values).mean(axis=0)
        pd.DataFrame({"feature": cols, "mean_abs_shap": mean_abs}) \
          .sort_values("mean_abs_shap", ascending=False) \
          .to_csv(OUT / f"shap_importances_{outcome}.csv", index=False)
        print(f"[shap] shap_summary_{outcome}.png produit.")
    except Exception as exc:
        print(f"[shap] SHAP skipped ({exc}) — permutation importance suffit.")

    return imp
