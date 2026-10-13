"""
ANNITIA Challenge 2026 — Orchestrateur v3 (anti-overfitting).
python -m src.run_pipeline
MIT License.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .config import (
    TRAIN_CSV, TEST_CSV, SAMPLE_SUB_CSV, OUT,
    W_HEPATIC, W_DEATH, SEED,
)
from .features import build_features, align_columns, impute_consistent, select_features_by_cindx
from .targets  import prepare_survival_targets
from .models   import multi_seed_ensemble
from .leakage  import run_all_checks
from .explain  import fit_final_model, shap_report


MULTI_SEEDS = [42, 123, 456, 789, 2026]


def _load():
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    sample = pd.read_csv(SAMPLE_SUB_CSV)
    print(f"[data] train={train.shape}  test={test.shape}")
    return train, test, sample


def run_endpoint(train_raw, test_raw, outcome: str):
    print("\n" + "=" * 78)
    print(f" Endpoint: {outcome.upper()}")
    print("=" * 78)

    # 1) Cibles
    df_valid, y, mask = prepare_survival_targets(train_raw, outcome=outcome)
    t = y["Time_years"].astype(float)
    e = y[y.dtype.names[0]].astype(int)
    print(f"  patients: {len(df_valid)}  events: {e.sum()} ({100*e.mean():.1f}%)")

    # 2) Features
    X_train = build_features(df_valid, outcome=outcome, mask_post_event=True)
    X_test  = build_features(test_raw, outcome=None, mask_post_event=False)
    X_train, X_test = align_columns(X_train, X_test)
    print(f"  features brutes: {X_train.shape[1]}")

    # 3) Imputation cohérente (médianes du train appliquées au test)
    X_train, X_test, _ = impute_consistent(X_train, X_test)

    # 4) Sélection de features par C-index univarié
    selected = select_features_by_cindx(X_train, t, e, min_abs_delta=0.04)
    print(f"  features sélectionnées: {len(selected)} (seuil |C-0.5| > 0.04)")
    X_train = X_train[selected]
    X_test  = X_test[selected]

    # 5) Checks anti-fuite
    run_all_checks(X_train, X_test, t)

    # 6) Multi-seed ensemble
    test_pred, mean_oof, all_oof = multi_seed_ensemble(
        X_train, t, e, X_test, seeds=MULTI_SEEDS, verbose=True,
    )

    # 7) Importance
    print(f"\n[explain] permutation importance for {outcome}...")
    final_model, X_imp, y_full = fit_final_model(X_train, t, e, seed=SEED)
    shap_report(final_model, X_imp, t, e, outcome=outcome)

    return {
        "outcome": outcome,
        "oof_c_index": float(mean_oof),
        "test_pred": test_pred,
    }


def main():
    train, test, sample = _load()

    res_hep = run_endpoint(train, test, outcome="hepatic")
    res_dth = run_endpoint(train, test, outcome="death")

    sub = pd.DataFrame({
        "trustii_id":         test["trustii_id"].values,
        "risk_hepatic_event": res_hep["test_pred"],
        "risk_death":         res_dth["test_pred"],
    })
    sub = sub[list(sample.columns)]
    sub = sub.sort_values("trustii_id").reset_index(drop=True)
    assert sub[["risk_hepatic_event", "risk_death"]].notna().all().all()

    path = OUT / "submission.csv"
    sub.to_csv(path, index=False)
    print(f"\n[submission] {path} ({len(sub)} lignes)")

    est_score = W_HEPATIC * res_hep["oof_c_index"] + W_DEATH * res_dth["oof_c_index"]
    summary = {
        "seed": SEED,
        "multi_seeds": MULTI_SEEDS,
        "c_index_hepatic_oof": res_hep["oof_c_index"],
        "c_index_death_oof":   res_dth["oof_c_index"],
        "estimated_weighted_score": est_score,
    }
    (OUT / "cv_scores.json").write_text(json.dumps(summary, indent=2))
    print(f"\n*** Score pondéré OOF estimé : {est_score:.4f} ***")
    print(f"    (hep={res_hep['oof_c_index']:.4f}×0.7 + dth={res_dth['oof_c_index']:.4f}×0.3)")


if __name__ == "__main__":
    main()
