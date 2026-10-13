"""Stacking meta-learner : remplace l'agrégation par poids."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


def stack_predictions(
    oof_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    e: np.ndarray,
    t: np.ndarray,
    ranked: bool = True,
):
    """
    Apprend un Cox PH sur les OOF des modèles de base + leurs interactions.
    Retourne les prédictions finales (OOF et test).
    """
    keys = sorted(oof_preds.keys())

    def _stack_features(preds_dict):
        cols = []
        for k in keys:
            v = preds_dict[k]
            if ranked:
                v = (np.argsort(np.argsort(v)) / max(1, len(v) - 1))
            cols.append(v)
        Z = np.column_stack(cols)
        # interactions pairwise
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                Z = np.column_stack([Z, cols[i] * cols[j]])
        return Z

    Z_oof = _stack_features(oof_preds)
    Z_test = _stack_features(test_preds)

    y = Surv.from_arrays(event=e.astype(bool), time=t)
    meta = CoxPHSurvivalAnalysis(alpha=0.5)   # léger ridge
    meta.fit(Z_oof, y)

    oof_final = meta.predict(Z_oof)
    test_final = meta.predict(Z_test)
    c_stack, *_ = concordance_index_censored(e.astype(bool), t, oof_final)
    return oof_final, test_final, c_stack, meta