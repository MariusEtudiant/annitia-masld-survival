"""
ANNITIA v8 — Push beyond 0.9128.
Strategy:
1. Improve v3 (GBSA) with Optuna hyperparameter optimization
2. Improve v6c (LightGBM) by adding calendar features
3. Add v7 (XGBoost classification) for diversity
4. Optimal 3-model blend
"""
import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from .config import (TRAIN_CSV, TEST_CSV, SAMPLE_SUB_CSV, OUT,
                     W_HEPATIC, W_DEATH, BIOMARKER_PREFIXES)
from .targets import prepare_survival_targets

warnings.filterwarnings("ignore")


def rank01(x):
    return np.argsort(np.argsort(x)) / max(1, len(x) - 1)


# ---------------------------------------------------------------------------
# FEATURES (from v3)
# ---------------------------------------------------------------------------
def compute_scores(df):
    df = df.copy()
    age_cols = sorted([c for c in df.columns if c.startswith("Age_v")],
                      key=lambda c: int(c.split("_v")[1]))
    for age_col in age_cols:
        suf = age_col.replace("Age", "")
        alt, ast_, plt_, age = df.get(f"alt{suf}"), df.get(f"ast{suf}"), df.get(f"plt{suf}"), df.get(age_col)
        if all(x is not None for x in [alt, ast_, plt_, age]):
            sa, sp = alt.clip(lower=1.0), plt_.clip(lower=1.0)
            df[f"fib4{suf}"] = (age * ast_) / (sp * np.sqrt(sa))
            df[f"apri{suf}"] = (ast_ / 40.0) * 100.0 / sp
    return df


def biomarker_stats(df):
    df = compute_scores(df)
    n = len(df)
    out = pd.DataFrame(index=df.index)
    bio_list = [
        ("fibs_stiffness_med_BM_1", "fibs"), ("fib4", "fib4"),
        ("aixp_aix_result_BM_3", "aixp"), ("fibrotest_BM_2", "ft"),
        ("plt", "plt"), ("apri", "apri"), ("ast", "ast"), ("alt", "alt"),
        ("ggt", "ggt"), ("gluc_fast", "gluc"), ("bilirubin", "bili"),
        ("BMI", "bmi"), ("triglyc", "trig"), ("chol", "chol"),
    ]
    for prefix, tag in bio_list:
        cols = sorted([c for c in df.columns if c.startswith(prefix + "_v")],
                      key=lambda c: int(c.split("_v")[-1]))
        if not cols:
            continue
        V = df[cols].to_numpy(dtype=float)
        out[f"{tag}_mean"] = np.nanmean(V, axis=1)
        out[f"{tag}_max"] = np.nanmax(V, axis=1)
        out[f"{tag}_min"] = np.nanmin(V, axis=1)
        out[f"{tag}_std"] = np.nanstd(V, axis=1)
        out[f"{tag}_median"] = np.nanmedian(V, axis=1)
        last = np.full(n, np.nan); first = np.full(n, np.nan)
        for i in range(n):
            valid = np.where(np.isfinite(V[i]))[0]
            if len(valid):
                last[i] = V[i, valid[-1]]; first[i] = V[i, valid[0]]
        out[f"{tag}_last"] = last
        out[f"{tag}_first"] = first
        out[f"{tag}_delta"] = last - first
        # NEW: quantiles
        out[f"{tag}_q25"] = np.nanquantile(V, 0.25, axis=1)
        out[f"{tag}_q75"] = np.nanquantile(V, 0.75, axis=1)
        out[f"{tag}_iqr"] = out[f"{tag}_q75"] - out[f"{tag}_q25"]
    return out


def calendar_features(df):
    out = pd.DataFrame(index=df.index)
    age_cols = sorted([c for c in df.columns if c.startswith("Age_v")],
                      key=lambda c: int(c.split("_v")[1]))
    A = df[age_cols].to_numpy(dtype=float)
    n_visits = np.sum(np.isfinite(A), axis=1).astype(float)
    out["age_baseline"] = df["Age_v1"].values
    out["last_observed_age"] = np.nanmax(A, axis=1)
    out["follow_up_duration"] = out["last_observed_age"] - out["age_baseline"]
    out["n_visits"] = n_visits
    out["visit_density"] = n_visits / (out["follow_up_duration"] + 1e-3)
    intervals_v1_v2 = np.full(len(df), np.nan)
    last_interval = np.full(len(df), np.nan)
    mean_interval = np.full(len(df), np.nan)
    std_interval = np.full(len(df), np.nan)
    max_interval = np.full(len(df), np.nan)
    for i in range(len(df)):
        valid = np.where(np.isfinite(A[i]))[0]
        if len(valid) >= 2:
            diffs = np.diff(A[i, valid])
            intervals_v1_v2[i] = diffs[0]
            last_interval[i] = diffs[-1]
            mean_interval[i] = np.mean(diffs)
            std_interval[i] = np.std(diffs) if len(diffs) > 1 else 0.0
            max_interval[i] = np.max(diffs)
    out["interval_v1_v2"] = intervals_v1_v2
    out["last_interval"] = last_interval
    out["mean_interval"] = mean_interval
    out["std_interval"] = std_interval
    out["max_interval"] = max_interval
    return out


def n_obs_per_biomarker(df):
    out = pd.DataFrame(index=df.index)
    age_cols = [c for c in df.columns if c.startswith("Age_v")]
    n_visits_total = np.sum(np.isfinite(df[age_cols].to_numpy(dtype=float)), axis=1).astype(float)
    n_visits_total = np.maximum(n_visits_total, 1)
    for pref in BIOMARKER_PREFIXES:
        cols = sorted([c for c in df.columns if c.startswith(pref + "_v")],
                      key=lambda c: int(c.split("_v")[-1]))
        if not cols:
            continue
        V = df[cols].to_numpy(dtype=float)
        n_obs = np.sum(np.isfinite(V), axis=1).astype(float)
        out[f"{pref}_n_obs"] = n_obs
        out[f"{pref}_ratio_obs"] = n_obs / n_visits_total
        out[f"{pref}_has_any"] = (n_obs > 0).astype(float)
    return out


def static_features(df):
    out = pd.DataFrame(index=df.index)
    for col in ["gender", "T2DM", "Hypertension", "Dyslipidaemia",
                "bariatric_surgery", "bariatric_surgery_age"]:
        if col in df.columns:
            out[col] = df[col].values
    return out


def build_features(df, include_calendar=True):
    parts = [biomarker_stats(df), n_obs_per_biomarker(df), static_features(df)]
    if include_calendar:
        parts.append(calendar_features(df))
    X = pd.concat(parts, axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def align_cols(X, X_test):
    all_cols = sorted(set(X.columns) | set(X_test.columns))
    for c in all_cols:
        if c not in X.columns: X[c] = np.nan
        if c not in X_test.columns: X_test[c] = np.nan
    return X[all_cols], X_test[all_cols]


# ---------------------------------------------------------------------------
# MODEL 1: GBSA (v3 improved with more configs)
# ---------------------------------------------------------------------------
def run_gbsa_multiseed(X, t, e, X_test, params, seeds, n_splits=5):
    oof_all = np.zeros(len(X))
    test_all = np.zeros(len(X_test))
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        test_p = np.zeros(len(X_test))
        for fold, (tr, te) in enumerate(cv.split(X, e)):
            imp = SimpleImputer(strategy="median")
            X_tr = pd.DataFrame(imp.fit_transform(X.iloc[tr]), columns=X.columns)
            X_te = pd.DataFrame(imp.transform(X.iloc[te]), columns=X.columns)
            X_ts = pd.DataFrame(imp.transform(X_test), columns=X.columns)
            model = GradientBoostingSurvivalAnalysis(**params, random_state=seed)
            y_tr = Surv.from_arrays(event=e[tr].astype(bool), time=t[tr])
            model.fit(X_tr, y_tr)
            oof[te] = model.predict(X_te)
            test_p += model.predict(X_ts) / n_splits
        oof_all += oof
        test_all += test_p
    oof_all /= len(seeds); test_all /= len(seeds)
    c, *_ = concordance_index_censored(e.astype(bool), t, oof_all)
    return c, oof_all, test_all


# ---------------------------------------------------------------------------
# MODEL 2: LightGBM multi-horizon (v6c improved WITH calendar)
# ---------------------------------------------------------------------------
def fit_lgbm(X_tr, y_tr, X_val, y_val, seed=42):
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.02,
        "num_leaves": 15, "max_depth": 4, "min_data_in_leaf": 20,
        "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 5,
        "lambda_l1": 0.1, "lambda_l2": 1.0, "verbose": -1, "seed": seed,
    }
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=2000,
                     valid_sets=[dval],
                     callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])


def cv_horizon(X, t, e, X_test, horizon, n_splits=5, seed=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); test_p = np.zeros(len(X_test))
    for fold, (tr, te) in enumerate(cv.split(X, e)):
        y_tr_full = np.full(len(tr), np.nan)
        for j_l, j_g in enumerate(tr):
            if e[j_g] == 1 and t[j_g] <= horizon: y_tr_full[j_l] = 1.0
            elif t[j_g] > horizon: y_tr_full[j_l] = 0.0
        usable = ~np.isnan(y_tr_full)
        if usable.sum() < 30 or y_tr_full[usable].sum() < 5: continue
        tr_u = tr[usable]; y_u = y_tr_full[usable]
        rng = np.random.default_rng(seed + fold)
        ip = np.where(y_u == 1)[0]; inn = np.where(y_u == 0)[0]
        rng.shuffle(ip); rng.shuffle(inn)
        nvp = max(2, int(len(ip)*0.15)); nvn = max(5, int(len(inn)*0.15))
        vl = np.concatenate([ip[:nvp], inn[:nvn]])
        tl = np.setdiff1d(np.arange(len(tr_u)), vl)
        model = fit_lgbm(X.iloc[tr_u[tl]], y_u[tl], X.iloc[tr_u[vl]], y_u[vl], seed+fold)
        oof[te] = model.predict(X.iloc[te])
        test_p += model.predict(X_test) / n_splits
    c, *_ = concordance_index_censored(e.astype(bool), t, oof)
    return c, oof, test_p


def multi_horizon_multiseed(X, t, e, X_test, horizons, seeds):
    h_oofs, h_tests = [], []
    for h in horizons:
        oof_seeds, test_seeds = [], []
        for s in seeds:
            c, o, tp = cv_horizon(X, t, e, X_test, h, seed=s)
            oof_seeds.append(o); test_seeds.append(tp)
        h_oofs.append(np.mean([rank01(o) for o in oof_seeds], axis=0))
        h_tests.append(np.mean([rank01(o) for o in test_seeds], axis=0))
        c_h, *_ = concordance_index_censored(e.astype(bool), t, h_oofs[-1])
        print(f"    h={h:5.1f}: C={c_h:.4f}")
    blend_oof = np.mean([rank01(o) for o in h_oofs], axis=0)
    blend_test = np.mean([rank01(o) for o in h_tests], axis=0)
    c_b, *_ = concordance_index_censored(e.astype(bool), t, blend_oof)
    print(f"    BLEND: C={c_b:.4f}")
    return c_b, blend_oof, blend_test


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    sample = pd.read_csv(SAMPLE_SUB_CSV)

    seeds_5 = [42, 123, 456, 789, 2024]
    seeds_10 = list(range(0, 1000, 100))

    for outcome in ["hepatic", "death"]:
        print(f"\n{'='*70}\n{outcome.upper()}\n{'='*70}")

        df_v, y, _ = prepare_survival_targets(train, outcome=outcome)
        t = y["Time_years"].astype(float)
        e = y[y.dtype.names[0]].astype(int)
        print(f"  patients={len(df_v)}, events={e.sum()}")

        X_cal = build_features(df_v, include_calendar=True)
        X_cal_test = build_features(test, include_calendar=True)
        X_cal, X_cal_test = align_cols(X_cal, X_cal_test)

        X_nocal = build_features(df_v, include_calendar=False)
        X_nocal_test = build_features(test, include_calendar=False)
        X_nocal, X_nocal_test = align_cols(X_nocal, X_nocal_test)

        print(f"  features(+cal)={X_cal.shape[1]}, features(-cal)={X_nocal.shape[1]}")

        # ----- GBSA grid (v3 improved) -----
        print("\n  --- GBSA grid ---")
        gbsa_configs = {
            "d1_n100_lr0.05_msl15": dict(n_estimators=100, max_depth=1, learning_rate=0.05,
                min_samples_leaf=15, subsample=0.8, max_features=0.7),
            "d2_n200_lr0.02_msl20": dict(n_estimators=200, max_depth=2, learning_rate=0.02,
                min_samples_leaf=20, subsample=0.8, max_features=0.7),
            "d3_n200_lr0.02_msl20": dict(n_estimators=200, max_depth=3, learning_rate=0.02,
                min_samples_leaf=20, subsample=0.7, max_features=0.5),
            "d1_n300_lr0.02_msl15": dict(n_estimators=300, max_depth=1, learning_rate=0.02,
                min_samples_leaf=15, subsample=0.8, max_features=0.5),
            "d2_n300_lr0.01_msl20": dict(n_estimators=300, max_depth=2, learning_rate=0.01,
                min_samples_leaf=20, subsample=0.8, max_features=0.7),
            "d4_n200_lr0.01_msl25": dict(n_estimators=200, max_depth=4, learning_rate=0.01,
                min_samples_leaf=25, subsample=0.8, max_features=0.7),
        }
        best_gbsa_c, best_gbsa_cfg = 0, ""
        for name, params in gbsa_configs.items():
            c, _, _ = run_gbsa_multiseed(X_cal, t, e, X_cal_test, params, seeds=[42])
            print(f"    {name:40s} C={c:.4f}")
            if c > best_gbsa_c:
                best_gbsa_c = c; best_gbsa_cfg = name

        print(f"  Best GBSA: {best_gbsa_cfg} = {best_gbsa_c:.4f}")
        # Run best with 10 seeds
        c_gbsa, oof_gbsa, test_gbsa = run_gbsa_multiseed(
            X_cal, t, e, X_cal_test, gbsa_configs[best_gbsa_cfg], seeds_10)
        print(f"  GBSA 10-seed: C={c_gbsa:.4f}")

        # ----- LightGBM multi-horizon (v6c with calendar) -----
        print("\n  --- LightGBM multi-horizon ---")
        # Without calendar (original v6c)
        print("  Without calendar:")
        c_lgb_nocal, oof_lgb_nocal, test_lgb_nocal = multi_horizon_multiseed(
            X_nocal, t, e, X_nocal_test,
            horizons=[5, 8, 11, 14], seeds=seeds_5)
        # With calendar (new)
        print("  With calendar:")
        c_lgb_cal, oof_lgb_cal, test_lgb_cal = multi_horizon_multiseed(
            X_cal, t, e, X_cal_test,
            horizons=[5, 8, 11, 14], seeds=seeds_5)

        # ----- Coxnet (for diversity) -----
        print("\n  --- Coxnet ---")
        oof_cox_all = np.zeros(len(X_cal))
        test_cox_all = np.zeros(len(X_cal_test))
        for seed in seeds_5:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            oof_c = np.zeros(len(X_cal)); test_c = np.zeros(len(X_cal_test))
            for fold, (tr, te) in enumerate(cv.split(X_cal, e)):
                pipe = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("cox", CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                        max_iter=10000, fit_baseline_model=False)),
                ])
                y_tr = Surv.from_arrays(event=e[tr].astype(bool), time=t[tr])
                try:
                    pipe.fit(X_cal.iloc[tr], y_tr)
                    a = pipe.named_steps["cox"].alphas_[len(pipe.named_steps["cox"].alphas_)//2]
                    oof_c[te] = pipe.predict(X_cal.iloc[te], alpha=a)
                    test_c += pipe.predict(X_cal_test, alpha=a) / 5
                except: continue
            oof_cox_all += oof_c; test_cox_all += test_c
        oof_cox_all /= len(seeds_5); test_cox_all /= len(seeds_5)
        c_cox, *_ = concordance_index_censored(e.astype(bool), t, oof_cox_all)
        print(f"  Coxnet: C={c_cox:.4f}")

        # ----- MEGA BLEND SEARCH -----
        print("\n  --- Blend search ---")
        models = {
            "gbsa": (oof_gbsa, test_gbsa),
            "lgb_nocal": (oof_lgb_nocal, test_lgb_nocal),
            "lgb_cal": (oof_lgb_cal, test_lgb_cal),
            "cox": (oof_cox_all, test_cox_all),
        }
        best_blend_c, best_blend_test, best_blend_name = 0, None, ""
        # 2-model blends
        model_names = list(models.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    oof_b = w * rank01(models[m1][0]) + (1-w) * rank01(models[m2][0])
                    c_b, *_ = concordance_index_censored(e.astype(bool), t, oof_b)
                    if c_b > best_blend_c:
                        best_blend_c = c_b
                        test_b = w * rank01(models[m1][1]) + (1-w) * rank01(models[m2][1])
                        best_blend_test = test_b
                        best_blend_name = f"{m1}={w:.1f}+{m2}={1-w:.1f}"
        # 3-model blends
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                for k in range(j+1, len(model_names)):
                    m1, m2, m3 = model_names[i], model_names[j], model_names[k]
                    for w1 in [0.2, 0.4, 0.5, 0.6, 0.7]:
                        for w2 in [0.1, 0.2, 0.3]:
                            w3 = 1.0 - w1 - w2
                            if w3 < 0.05: continue
                            oof_b = w1*rank01(models[m1][0]) + w2*rank01(models[m2][0]) + w3*rank01(models[m3][0])
                            c_b, *_ = concordance_index_censored(e.astype(bool), t, oof_b)
                            if c_b > best_blend_c:
                                best_blend_c = c_b
                                test_b = w1*rank01(models[m1][1]) + w2*rank01(models[m2][1]) + w3*rank01(models[m3][1])
                                best_blend_test = test_b
                                best_blend_name = f"{m1}={w1:.1f}+{m2}={w2:.1f}+{m3}={w3:.1f}"

        print(f"  Best blend: {best_blend_name} = {best_blend_c:.4f}")

        # Store results
        if outcome == "hepatic":
            test_h_gbsa, test_h_blend = test_gbsa, best_blend_test
            oof_h_gbsa, oof_h_blend = oof_gbsa, None
            c_h_gbsa, c_h_blend = c_gbsa, best_blend_c
            models_h = models
        else:
            test_d_gbsa, test_d_blend = test_gbsa, best_blend_test
            c_d_gbsa, c_d_blend = c_gbsa, best_blend_c
            models_d = models

    # ----- SAVE SUBMISSIONS -----
    print(f"\n{'='*70}\nSAVING\n{'='*70}")
    def save_sub(name, h_pred, d_pred):
        sub = pd.DataFrame({
            "trustii_id": test["trustii_id"].values,
            "risk_hepatic_event": h_pred,
            "risk_death": d_pred,
        })
        sub = sub[list(sample.columns)].sort_values("trustii_id").reset_index(drop=True)
        path = OUT / f"submission_{name}.csv"
        sub.to_csv(path, index=False)
        print(f"  {name}")

    # v8 = best blend for each endpoint
    save_sub("v8_blend", test_h_blend, test_d_blend)
    save_sub("v8_gbsa_only", test_h_gbsa, test_d_gbsa)

    # Blend v8 with existing v3
    v3_path = OUT / "submission_v3_gbsa_only.csv"
    if v3_path.exists():
        v3 = pd.read_csv(v3_path).sort_values("trustii_id").reset_index(drop=True)
        for w in [0.3, 0.5, 0.7]:
            h_blend = w * rank01(test_h_blend) + (1-w) * rank01(v3["risk_hepatic_event"].values)
            d_blend = w * rank01(test_d_blend) + (1-w) * rank01(v3["risk_death"].values)
            save_sub(f"v8_x_v3_{int(w*100)}_{int((1-w)*100)}", h_blend, d_blend)

    # Blend with v6c_blend30 (current best LB=0.9128)
    v6c_path = OUT / "submission_v6c_blend30.csv"
    if v6c_path.exists():
        v6c = pd.read_csv(v6c_path).sort_values("trustii_id").reset_index(drop=True)
        for w in [0.3, 0.5, 0.7]:
            h_blend = w * rank01(test_h_blend) + (1-w) * rank01(v6c["risk_hepatic_event"].values)
            d_blend = w * rank01(test_d_blend) + (1-w) * rank01(v6c["risk_death"].values)
            save_sub(f"v8_x_v6cb30_{int(w*100)}_{int((1-w)*100)}", h_blend, d_blend)

    print(f"\n*** OOF Summary ***")
    print(f"  Hepatic GBSA: {c_h_gbsa:.4f}")
    print(f"  Hepatic best blend: {c_h_blend:.4f}")
    print(f"  Death GBSA: {c_d_gbsa:.4f}")
    print(f"  Death best blend: {c_d_blend:.4f}")


if __name__ == "__main__":
    main()
