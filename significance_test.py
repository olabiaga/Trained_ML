"""
Paired significance tests between the three tuned models.

Uses the SAME 15 folds as train.py (GroupKFold by event, 5 folds x 3 repeats, random_state = repeat),
so each fold gives one paired score per model.

Tests (per metric):
  - Friedman test across the 3 models
  - Wilcoxon signed-rank test, Random Forest vs each other model (Holm-corrected)
  - Corrected resampled t-test (Nadeau & Bengio, 2003), which accounts for the overlap between
    training sets in repeated cross-validation; plain tests on CV folds are too optimistic.

Outputs: models/per_fold_results.csv, models/significance_tests.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from train import ALGORITHMS, FEATURES, N_REPEATS, N_SPLITS, add_features, load

OUT = Path(__file__).parent / "models"

# Best hyperparameters found by train.py (see models/tuning_log.csv)
BEST = {
    "Random Forest": {"max_features": 0.3, "min_samples_leaf": 1, "min_freq": 1},
    "Gradient Boosting": {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 2, "min_freq": 1},
    "Linear Regression": {"alpha": 0.3, "min_freq": 1},
}
METRICS = {  # name -> higher_is_better
    "R2 (line item)": True,
    "MAE line (PHP)": False,
    "MAPE event (%)": False,
}


def per_fold_scores(df):
    X, y, groups = df[FEATURES], np.log(df["Unit_Cost"]), df["Event_Title"]
    rows = []
    for rep in range(N_REPEATS):
        splits = GroupKFold(N_SPLITS, shuffle=True, random_state=rep).split(X, y, groups)
        for fold, (tr, te) in enumerate(splits):
            test = df.iloc[te]
            for name, params in BEST.items():
                m = ALGORITHMS[name]["build"](params).fit(X.iloc[tr], y.iloc[tr])
                pred = np.exp(m.predict(X.iloc[te])) * test["Qty"].values
                ev = test.assign(pred=pred).groupby("Event_Title")[["Subtotal", "pred"]].sum()
                rows.append({
                    "repeat": rep + 1, "fold": fold + 1, "model": name,
                    "n_train": len(tr), "n_test": len(te),
                    "R2 (line item)": r2_score(test["Subtotal"], pred),
                    "MAE line (PHP)": mean_absolute_error(test["Subtotal"], pred),
                    "MAPE event (%)": float(np.mean(np.abs(ev["pred"] - ev["Subtotal"]) / ev["Subtotal"]) * 100),
                })
    return pd.DataFrame(rows)


def corrected_ttest(diff, test_train_ratio):
    """Nadeau & Bengio (2003) corrected resampled t-test, two-sided."""
    k = len(diff)
    var = np.var(diff, ddof=1)
    if var == 0:
        return np.nan, 1.0
    t = diff.mean() / np.sqrt((1 / k + test_train_ratio) * var)
    return t, 2 * stats.t.sf(abs(t), df=k - 1)


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    df = add_features(load())
    folds = per_fold_scores(df)
    OUT.mkdir(exist_ok=True)
    folds.to_csv(OUT / "per_fold_results.csv", index=False, float_format="%.4f")
    ratio = (folds["n_test"] / folds["n_train"]).mean()

    results = []
    for metric, higher_better in METRICS.items():
        wide = folds.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
        fr = stats.friedmanchisquare(*[wide[m] for m in BEST])
        results.append({"metric": metric, "comparison": "All 3 models", "test": "Friedman",
                        "statistic": fr.statistic, "p_value": fr.pvalue, "p_holm": np.nan,
                        "mean_diff": np.nan, "rf_better_folds": ""})
        comps = []
        for other in ("Gradient Boosting", "Linear Regression"):
            d = (wide["Random Forest"] - wide[other]).values
            if not higher_better:
                d = -d  # positive = Random Forest better
            w = stats.wilcoxon(d)
            t, p_t = corrected_ttest(d, ratio)
            comps.append((other, d, w, t, p_t))
        p_holm_w = holm(np.array([c[2].pvalue for c in comps]))
        p_holm_t = holm(np.array([c[4] for c in comps]))
        for (other, d, w, t, p_t), pw, pt in zip(comps, p_holm_w, p_holm_t):
            better = f"{int((d > 0).sum())}/{len(d)}"
            results.append({"metric": metric, "comparison": f"RF vs {other}", "test": "Wilcoxon signed-rank",
                            "statistic": w.statistic, "p_value": w.pvalue, "p_holm": pw,
                            "mean_diff": d.mean(), "rf_better_folds": better})
            results.append({"metric": metric, "comparison": f"RF vs {other}", "test": "Corrected resampled t",
                            "statistic": t, "p_value": p_t, "p_holm": pt,
                            "mean_diff": d.mean(), "rf_better_folds": better})

    res = pd.DataFrame(results)
    res.to_csv(OUT / "significance_tests.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:,.4f}")
    print(f"15 paired folds per model | mean test/train ratio = {ratio:.3f}\n")
    print("Mean per-fold scores:")
    print(folds.groupby("model")[list(METRICS)].agg(["mean", "std"]).round(3).to_string())
    print("\n(mean_diff > 0 means Random Forest is better)")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
