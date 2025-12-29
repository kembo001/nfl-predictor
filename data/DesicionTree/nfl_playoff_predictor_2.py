"""
NFL Playoff Predictor (Decision Tree / Random Forest) — UPDATED, LESS LEAKY, MORE TRUSTWORTHY EVAL

What’s changed vs your version:
- No time leakage from preprocessing: missing-value imputation is inside a Pipeline.
- Proper “walk-forward by season” backtest (train on seasons < t, test on season t).
- Evaluation focuses on what matters for this problem:
  - Champion rank among playoff teams each year
  - Top-1 / Top-3 hit rates
  - Mean reciprocal rank
  - Brier score + Log loss (probability quality)
- Adds a simple seed-only baseline for comparison.
- 2025 playoff predictions (based on 2024 season teams) train ONLY on seasons <= 2023.

Assumes your CSV has at least:
- season (int)
- team (str)
- conference (str)
- champion (0/1)
- playoff_seed (int)
- plus the feature columns below
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

from sklearn.inspection import permutation_importance


# ============================================================
# 1) LOAD + FEATURES
# ============================================================

DEFAULT_FEATURE_COLS = [
    # Offensive metrics
    "passing_epa",
    "rushing_epa",
    "total_offensive_epa",

    # Defensive metrics
    "defensive_epa",
    "defensive_pass_epa",
    "defensive_rush_epa",

    # Overall team performance
    "win_pct",
    "point_differential",
    "net_epa",

    # Momentum/hot streak metrics
    "last_5_win_pct",
    "momentum_residual",

    # Pass rush and protection
    "pass_rush_rating",
    "sack_rate",
    "pass_block_rating",

    # Playoff seeding
    "playoff_seed",
]


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if "season" not in df.columns:
        raise ValueError("Expected a 'season' column in the CSV.")
    if "champion" not in df.columns:
        raise ValueError("Expected a 'champion' column in the CSV (0/1).")

    df["season"] = df["season"].astype(int)
    df["champion"] = df["champion"].astype(int)

    print(f"Loaded {len(df)} playoff team rows from {df['season'].min()}–{df['season'].max()}")
    return df


def get_feature_cols(df: pd.DataFrame, requested_cols=None):
    requested_cols = requested_cols or DEFAULT_FEATURE_COLS
    available = [c for c in requested_cols if c in df.columns]
    missing = [c for c in requested_cols if c not in df.columns]
    if missing:
        print(f"WARNING: missing {len(missing)} requested features: {missing}")
    print(f"Using {len(available)} features: {available}")
    return available


# ============================================================
# 2) MODELS
# ============================================================

def make_dt_pipeline(max_depth=5, random_state=42) -> Pipeline:
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
    )
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("dt", dt),
    ])
    return pipe


def make_rf_pipeline(n_estimators=500, max_depth=10, random_state=42) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", rf),
    ])
    return pipe


# ============================================================
# 3) EVALUATION HELPERS
# ============================================================

def champion_rank(proba: np.ndarray, y_true: np.ndarray) -> int:
    """
    Returns 1 if the champion got the highest probability in that season, 2 if second, etc.
    Assumes exactly one champion in y_true.
    """
    champ_idx = np.where(y_true == 1)[0]
    if len(champ_idx) != 1:
        # Some datasets include multiple champs (bad data) or none (incomplete year)
        return -1

    champ_idx = champ_idx[0]
    ordering = np.argsort(proba)[::-1]  # highest proba first
    return int(np.where(ordering == champ_idx)[0][0] + 1)


def seed_baseline_rank(df_season: pd.DataFrame) -> int:
    """
    Baseline: rank teams by playoff_seed ascending (1 is best).
    """
    if "playoff_seed" not in df_season.columns:
        return -1
    y_true = df_season["champion"].to_numpy().astype(int)
    champ_idx = np.where(y_true == 1)[0]
    if len(champ_idx) != 1:
        return -1
    champ_idx = champ_idx[0]

    ordering = np.argsort(df_season["playoff_seed"].to_numpy())  # low seed first
    return int(np.where(ordering == champ_idx)[0][0] + 1)


def print_holdout_eval(model: Pipeline, df: pd.DataFrame, feature_cols, train_end_season: int):
    """
    Train on seasons <= train_end_season, test on seasons > train_end_season
    Prints standard classification metrics (useful but not the main signal here).
    """
    train = df[df["season"] <= train_end_season].copy()
    test = df[df["season"] > train_end_season].copy()

    X_train, y_train = train[feature_cols], train["champion"]
    X_test, y_test = test[feature_cols], test["champion"]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print(f"HOLDOUT EVAL: Train <= {train_end_season} | Test > {train_end_season}")
    print("=" * 60)
    print(f"Test samples: {len(test)} | Champions in test: {(y_test==1).sum()}")

    print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
    print("\nClassification report (warning: class imbalance makes this noisy):")
    print(classification_report(y_test, pred, target_names=["Non-Champion", "Champion"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))

    # Probability quality
    try:
        ll = log_loss(y_test, np.vstack([1 - proba, proba]).T, labels=[0, 1])
        bs = brier_score_loss(y_test, proba)
        print(f"\nLog loss: {ll:.4f}")
        print(f"Brier score: {bs:.4f}")
    except Exception as e:
        print(f"\nCould not compute log loss / brier: {e}")


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols,
    model_factory,
    start_season: int,
    end_season: int,
):
    """
    Walk-forward evaluation:
      For each season t in [start_season, end_season]:
        train on seasons < t
        test on season == t
        compute champion rank metrics + prob metrics
    """
    seasons = sorted(df["season"].unique())
    seasons = [s for s in seasons if start_season <= s <= end_season]

    rows = []

    for t in seasons:
        train = df[df["season"] < t].copy()
        test = df[df["season"] == t].copy()

        # Need at least some training data + exactly one champion in test
        if len(train) == 0:
            continue
        if test["champion"].sum() != 1:
            continue

        X_train, y_train = train[feature_cols], train["champion"].astype(int)
        X_test, y_test = test[feature_cols], test["champion"].astype(int)

        model = model_factory()
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        rank = champion_rank(proba, y_test.to_numpy())
        seed_rank = seed_baseline_rank(test)

        # Probability quality metrics for that season (only 1 positive, so log_loss can be sensitive)
        # Still useful in aggregate.
        try:
            ll = log_loss(y_test, np.vstack([1 - proba, proba]).T, labels=[0, 1])
        except Exception:
            ll = np.nan
        try:
            bs = brier_score_loss(y_test, proba)
        except Exception:
            bs = np.nan

        rows.append({
            "season": t,
            "n_playoff_teams": len(test),
            "champion_rank_model": rank,
            "champion_rank_seed_baseline": seed_rank,
            "log_loss": ll,
            "brier": bs,
        })

    results = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)

    if len(results) == 0:
        print("No seasons evaluated in walk-forward backtest. Check your data.")
        return results

    # Aggregate summary
    model_ranks = results["champion_rank_model"].to_numpy()
    seed_ranks = results["champion_rank_seed_baseline"].to_numpy()

    def summarize_ranks(name, ranks):
        ranks = ranks[ranks > 0]
        top1 = np.mean(ranks == 1)
        top3 = np.mean(ranks <= 3)
        mrr = np.mean(1.0 / ranks)
        mean_rank = np.mean(ranks)
        print(f"\n{name}:")
        print(f"  Seasons evaluated: {len(ranks)}")
        print(f"  Mean champion rank: {mean_rank:.2f} (lower is better)")
        print(f"  Top-1 hit rate: {top1:.1%}")
        print(f"  Top-3 hit rate: {top3:.1%}")
        print(f"  Mean reciprocal rank: {mrr:.3f}")

    print("\n" + "=" * 60)
    print(f"WALK-FORWARD BACKTEST: {results['season'].min()}–{results['season'].max()}")
    print("=" * 60)

    summarize_ranks("MODEL", model_ranks)
    summarize_ranks("SEED BASELINE", seed_ranks)

    # Probability metrics
    ll_mean = np.nanmean(results["log_loss"].to_numpy())
    bs_mean = np.nanmean(results["brier"].to_numpy())
    print(f"\nProbability quality (averaged over seasons):")
    print(f"  Mean log loss: {ll_mean:.4f}")
    print(f"  Mean Brier score: {bs_mean:.4f}")

    return results


# ============================================================
# 4) INTERPRETABILITY PLOTS
# ============================================================

def plot_rf_feature_importance(rf_pipe: Pipeline, feature_cols, outpath="feature_importance_rf.png"):
    rf = rf_pipe.named_steps["rf"]
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Random Forest (impurity) Feature Importance")
    plt.bar(range(len(feature_cols)), importances[order])
    plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in order], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.show()

    print("\nTop 10 impurity importances:")
    for i in range(min(10, len(feature_cols))):
        idx = order[i]
        print(f"  {i+1:2d}. {feature_cols[idx]}: {importances[idx]:.4f}")


def plot_permutation_importance_on_season(
    rf_pipe: Pipeline,
    df: pd.DataFrame,
    feature_cols,
    season: int,
    outpath="permutation_importance.png",
    n_repeats=30,
    random_state=42,
):
    """
    Permutation importance is more reliable than impurity importance with correlated features.
    We compute it on a labeled season (must have champion labels).
    """
    df_season = df[df["season"] == season].copy()
    if df_season["champion"].sum() != 1:
        print(f"Skipping permutation importance: season {season} does not have exactly one champion label.")
        return

    X = df_season[feature_cols]
    y = df_season["champion"].astype(int)

    r = permutation_importance(
        rf_pipe,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_log_loss",  # focuses on probability quality
    )

    importances = r.importances_mean
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f"Permutation Importance (season {season}, scoring=neg_log_loss)")
    plt.bar(range(len(feature_cols)), importances[order])
    plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in order], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.show()

    print(f"\nTop 10 permutation importances (season {season}):")
    for i in range(min(10, len(feature_cols))):
        idx = order[i]
        print(f"  {i+1:2d}. {feature_cols[idx]}: {importances[idx]:.6f}")


def visualize_decision_tree(dt_pipe: Pipeline, feature_cols, outpath="decision_tree.png"):
    dt = dt_pipe.named_steps["dt"]
    plt.figure(figsize=(22, 12))
    plot_tree(
        dt,
        feature_names=feature_cols,
        class_names=["Non-Champion", "Champion"],
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title("Decision Tree (trained model)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.show()


# ============================================================
# 5) PREDICT A SEASON (NO LEAKAGE)
# ============================================================

def predict_playoffs_for_season(df: pd.DataFrame, feature_cols, target_season: int, rf_factory):
    """
    Predict champion probabilities for playoff teams in target_season
    using a model trained ONLY on seasons < target_season.
    """
    train = df[df["season"] < target_season].copy()
    test = df[df["season"] == target_season].copy()

    if len(test) == 0:
        print(f"No rows found for season {target_season}.")
        return None
    if len(train) == 0:
        print(f"Not enough training data before season {target_season}.")
        return None

    model = rf_factory()
    model.fit(train[feature_cols], train["champion"].astype(int))

    proba = model.predict_proba(test[feature_cols])[:, 1]
    out = test.copy()
    out["champion_probability"] = proba

    cols = ["team", "conference", "playoff_seed", "win_pct", "point_differential", "champion_probability"]
    cols = [c for c in cols if c in out.columns]

    out = out[cols].sort_values("champion_probability", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print(f"PLAYOFF PREDICTIONS for season {target_season} (trained on seasons < {target_season})")
    print("=" * 60)

    for i, row in out.iterrows():
        team = str(row.get("team", "??")).ljust(4)
        conf = str(row.get("conference", ""))
        seed = int(row.get("playoff_seed", -1)) if "playoff_seed" in out.columns else -1
        winp = row.get("win_pct", np.nan)
        pdiff = row.get("point_differential", np.nan)
        prob = row["champion_probability"]

        seed_str = f"Seed {seed:2d}" if seed != -1 else "Seed ??"
        win_str = f"Win%: {winp:.3f}" if pd.notna(winp) else "Win%: ---"
        pd_str = f"PD: {pdiff:+.0f}" if pd.notna(pdiff) else "PD: ---"

        print(f"{i+1:2d}. {team} ({conf}) {seed_str} | {win_str} | {pd_str} | Prob: {prob:.1%}")

    return out


# ============================================================
# 6) MAIN
# ============================================================

def main():
    DATA_PATH = "../all_seasons_data.csv"

    # For your stated goal:
    # "Use 2024 season data (regular season) to predict 2025 playoffs"
    # => your playoff-team rows for that postseason are season=2024, and you must train on <=2023.
    PREDICT_SEASON = 2024

    df = load_data(DATA_PATH)
    feature_cols = get_feature_cols(df, DEFAULT_FEATURE_COLS)

    print("\nTarget distribution overall:")
    print(f"  Non-Champions: {(df['champion']==0).sum()} ({(df['champion']==0).mean():.1%})")
    print(f"  Champions:     {(df['champion']==1).sum()} ({(df['champion']==1).mean():.1%})")

    # Factories so each backtest season gets a fresh model
    dt_factory = lambda: make_dt_pipeline(max_depth=5)
    rf_factory = lambda: make_rf_pipeline(n_estimators=500, max_depth=10)

    # 1) Holdout sanity check (still useful, but small if you only have 2023–2024 in test)
    print_holdout_eval(
        model=rf_factory(),
        df=df,
        feature_cols=feature_cols,
        train_end_season=2022,  # matches your original idea: test on 2023+
    )

    # 2) Walk-forward backtest (this is the trust test)
    # Use a reasonable range (adjust if you want). Needs enough prior seasons to train.
    backtest = walk_forward_backtest(
        df=df,
        feature_cols=feature_cols,
        model_factory=rf_factory,
        start_season=max(df["season"].min() + 5, 2005),
        end_season=df["season"].max(),
    )

    # Save backtest table for inspection
    if len(backtest) > 0:
        backtest_path = "walk_forward_backtest_results.csv"
        backtest.to_csv(backtest_path, index=False)
        print(f"\nSaved walk-forward results to: {backtest_path}")

    # 3) Train a model on seasons < PREDICT_SEASON and predict that season’s playoff teams
    # This avoids training on the same season you’re “predicting.”
    preds = predict_playoffs_for_season(
        df=df,
        feature_cols=feature_cols,
        target_season=PREDICT_SEASON,
        rf_factory=rf_factory,
    )

    # 4) Optional interpretability:
    # Train an RF on all data up to the year before prediction and plot importances.
    train_df = df[df["season"] < PREDICT_SEASON].copy()
    rf_for_plots = rf_factory()
    rf_for_plots.fit(train_df[feature_cols], train_df["champion"].astype(int))

    plot_rf_feature_importance(rf_for_plots, feature_cols, outpath="feature_importance_rf.png")

    # Permutation importance on the most recent *labeled* season before prediction.
    # If you’re predicting season=2024, then season=2023 is the last fully “past” labeled season.
    last_labeled = PREDICT_SEASON - 1
    if last_labeled in df["season"].unique():
        plot_permutation_importance_on_season(
            rf_pipe=rf_for_plots,
            df=df,
            feature_cols=feature_cols,
            season=last_labeled,
            outpath="permutation_importance.png",
        )

    # Optional: decision tree visualization (trained on pre-prediction seasons)
    dt_for_plot = dt_factory()
    dt_for_plot.fit(train_df[feature_cols], train_df["champion"].astype(int))
    visualize_decision_tree(dt_for_plot, feature_cols, outpath="decision_tree.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
