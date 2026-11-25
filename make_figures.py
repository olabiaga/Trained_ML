"""
Create the four thesis figures (PNG, 300 dpi) in figures/.

  fig1_model_comparison.png   R² and accuracy of the three models (cross-validated)
  fig2_feature_importance.png Random Forest feature importance
  fig3_extrapolation.png      Forecasting limitation: flat prediction beyond the training years
  fig4_actual_vs_predicted.png Random Forest out-of-fold predictions vs actual line-item cost

Style: one blue accent on what the title is about, everything else grey; values labelled on marks.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from train import ALGORITHMS, FEATURES, add_features, cv_predict, load, load_h5

ROOT = Path(__file__).parent
OUT = ROOT / "figures"
MODEL = ROOT / "models" / "best_model.h5"

ACCENT = "#2a78d6"   # reference palette categorical slot 1
GREY = "#b9b8b0"
REF = "#8a8982"
INK = "#0b0b0b"
INK2 = "#52514e"
PESO = "₱"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": REF, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def title(fig, main, sub):
    fig.text(0.01, 1.04, main, fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.01, 0.995, sub, fontsize=10, color=INK2, ha="left", va="bottom")


def fig1_model_comparison():
    m = pd.read_csv(ROOT / "models" / "comparison_matrix.csv", index_col=0)
    models = ["Random Forest", "Gradient Boosting", "Linear Regression"]
    panels = [("R2 (line item)", "R² (line item)", 1.0, "{:.3f}"),
              ("Accuracy event (100-MAPE %)", "Accuracy, event total (%)", 100, "{:.1f}%")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (row, label, top_val, fmt) in zip(axes, panels):
        # round half up at 4 dp first so labels match the tables (0.4805 -> 0.481)
        vals = [float(np.round(m.loc[row, k], 4)) + 1e-9 for k in models]
        colors = [ACCENT if k == "Random Forest" else GREY for k in models]
        bars = ax.bar(range(3), vals, color=colors, width=0.6, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top_val * 0.02, fmt.format(v),
                    ha="center", va="bottom", color=INK, fontsize=10)
        ax.set_xticks(range(3), [k.replace(" ", "\n") for k in models])
        ax.set_ylim(0, top_val)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(label, loc="left", fontsize=11, color=INK)
    title(fig, "Random Forest scored highest on both measures",
          "Cross-validated on unseen events (Group K-Fold, 5 folds × 3 repeats); "
          "its lead over Gradient Boosting is not statistically significant")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_model_comparison.png")
    plt.close(fig)


def grouped_importance(model):
    pre, rf = model.named_steps["pre"], model.named_steps["m"]
    imp = pd.Series(rf.feature_importances_, index=pre.get_feature_names_out())

    def group(name):
        kind, rest = name.split("__", 1)
        return rest.split("_")[0] if kind == "cat" else rest
    g = imp.groupby(group).sum() * 100
    names = {"Qty_log": "Quantity", "Category": "Category", "Is_Bulk": "Lump-sum item (Qty = 1)",
             "Item": "Item name", "College": "College", "Year": "Year", "Month": "Month", "Semester": "Semester"}
    return g.rename(index=names).round(1).sort_values()  # rounded so shown values add up


def fig2_feature_importance(model):
    g = grouped_importance(model)
    drivers = {"Quantity", "Category"}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = [ACCENT if k in drivers else GREY for k in g.index]
    ax.barh(g.index, g.values, color=colors, height=0.65, edgecolor="white", linewidth=2)
    for y, v in enumerate(g.values):
        ax.text(v + 0.6, y, f"{v:.1f}%", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, g.max() * 1.12)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    share = g[list(drivers)].sum()
    dates = g[["Year", "Month", "Semester"]].sum()
    title(fig, "Quantity and category drive most of the predicted cost",
          f"{share:.1f}% of importance from quantity and category; only {dates:.1f}% from year, month and semester")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_feature_importance.png")
    plt.close(fig)


def fig3_extrapolation(model, df):
    last = int(df["Year"].max())
    years = [2021, 2023, 2026, 2028, 2030, 2035]
    base = dict(Semester="1st Semester", College="CCS", Event_Title="X", Category="Food & Catering",
                Item="Snacks for Participants", Qty=100)
    rows = pd.DataFrame([{**base, "Date": pd.Timestamp(f"{y}-10-15")} for y in years])
    cost = np.exp(model.predict(add_features(rows)[FEATURES]))
    s = pd.Series(cost, index=years)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    inside, beyond = s[s.index <= last], s[s.index >= last]
    ax.plot(inside.index, inside.values, color=GREY, lw=2.5, marker="o", ms=7)
    ax.plot(beyond.index, beyond.values, color=ACCENT, lw=2.5, marker="o", ms=7)
    ax.axvline(last, color=REF, ls="--", lw=1)
    ax.text(last + 0.2, s.max() * 1.12, "Last year in training data", color=INK2, fontsize=9, va="top")
    offsets = {years[0]: (0, 10, "center"), last: (-8, 10, "right"), years[-1]: (0, 10, "right")}
    for yr, (dx, dy, ha) in offsets.items():
        ax.annotate(f"{PESO}{s[yr]:.2f}", (yr, s[yr]), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, color=INK, fontsize=10)
    ax.text(2030.5, s[last] * 0.86, "Future years: no change", color=ACCENT, ha="center", fontsize=10)
    ax.set_xticks(years)
    ax.set_ylim(0, s.max() * 1.15)
    ax.set_ylabel(f"Predicted unit cost ({PESO})")
    ax.grid(axis="y", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)
    title(fig, "Random Forest cannot extrapolate: the forecast stays flat after the training data ends",
          f"Snacks for Participants, 100 pcs, CCS: unit cost stays at {PESO}{s[last]:.2f} from {last} to {years[-1]}")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_extrapolation.png")
    plt.close(fig)


def fig4_actual_vs_predicted(df):
    params = {"max_features": 0.3, "min_samples_leaf": 1, "min_freq": 1}
    unit = cv_predict(ALGORITHMS["Random Forest"]["build"], params, df).mean(axis=0)
    pred = unit * df["Qty"].values
    actual = df["Subtotal"].values
    r2 = r2_score(actual, pred)

    fig, ax = plt.subplots(figsize=(6, 5.6))
    lo, hi = 400, 70000
    ax.plot([lo, hi], [lo, hi], color=REF, ls="--", lw=1, zorder=1)
    ax.text(hi * 0.9, lo * 1.25, "Dashed line = perfect prediction", color=INK2, fontsize=9, ha="right")
    ax.scatter(actual, pred, s=36, color=ACCENT, alpha=0.65, edgecolor="white", linewidth=0.8, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [500, 1000, 5000, 10000, 50000]
    ax.set_xticks(ticks, [f"{PESO}{t:,}" for t in ticks])
    ax.set_yticks(ticks, [f"{PESO}{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Actual line-item cost (log scale)")
    ax.set_ylabel("Predicted line-item cost (log scale)")
    ax.grid(color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    title(fig, "Predictions follow actual costs on unseen events",
          f"Random Forest, out-of-fold predictions for {len(df)} line items (R² = {r2:.3f}, averaged over 3 repeats)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_actual_vs_predicted.png")
    plt.close(fig)
    return r2


def main():
    OUT.mkdir(exist_ok=True)
    df = add_features(load())
    model, _ = load_h5(MODEL)
    fig1_model_comparison()
    fig2_feature_importance(model)
    fig3_extrapolation(model, df)
    r2 = fig4_actual_vs_predicted(df)
    print(f"Saved 4 figures to {OUT} (fig4 R2 = {r2:.3f})")


if __name__ == "__main__":
    main()
