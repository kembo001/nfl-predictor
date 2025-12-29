"""
NFL Playoff Model v16.2 — v16.1 + stability fixes

Changes vs v16.1:
- Use XGBoost(full) as ensemble model (more stable than baseline-only XGB on tiny samples)
- Regularized Platt scaling with safety guards (avoids huge/unstable params)
- Cap baseline adjustment magnitude (prevents big flips away from offset)
- Toss-up shrink toward offset (seed gap <= 1 only) to reduce noise in ambiguous games
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
import warnings

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Note: XGBoost not installed. Skipping gradient boosting.")


# ============================================================
# DATA LOADING
# ============================================================

def load_all_data():
    games_df = pd.read_csv("../nfl_playoff_results_2000_2024_with_epa.csv")
    team_df = pd.read_csv("../df_with_upsets_merged.csv")
    spread_df = pd.read_csv("../csv/nfl_biggest_playoff_upsets_2000_2024.csv")

    print(f"Loaded {len(games_df)} playoff games")
    print(f"Loaded {len(team_df)} team-season records")
    print(f"Loaded {len(spread_df)} games with spread data")

    return games_df, team_df, spread_df


def merge_spread_data_safe(games_df, spread_df):
    games_df = games_df.copy()
    spread_lookup = {}

    for _, row in spread_df.iterrows():
        season, game_type = row["season"], row["game_type"]
        underdog = row["underdog"]
        magnitude = abs(row["spread_magnitude"])
        teams = {row["winner"], row["loser"]}

        matching = games_df[
            (games_df["season"] == season)
            & (games_df["game_type"] == game_type)
            & (games_df["home_team"].isin(teams))
            & (games_df["away_team"].isin(teams))
        ]

        if len(matching) > 0:
            game = matching.iloc[0]
            home = game["home_team"]
            home_spread = magnitude if home == underdog else -magnitude
            key = (season, game_type, game["away_team"], home)
            spread_lookup[key] = home_spread

    games_df["home_spread"] = games_df.apply(
        lambda r: spread_lookup.get((r["season"], r["game_type"], r["away_team"], r["home_team"]), np.nan),
        axis=1,
    )

    matched = games_df["home_spread"].notna().sum()
    print(f"Matched spread data for {matched}/{len(games_df)} games")
    return games_df


def spread_sanity_check(games_df):
    df = games_df.dropna(subset=["home_spread"]).copy()
    if len(df) < 10:
        return
    df["home_win"] = (df["winner"] == df["home_team"]).astype(int)
    corr = np.corrcoef(df["home_spread"].values, df["home_win"].values)[0, 1]
    ok = "✓" if corr < 0 else "✗"
    print(f"Spread sanity check: corr(home_spread, home_win) = {corr:+.3f} {ok}")


# ============================================================
# BASELINES
# ============================================================

def compute_historical_baselines(games_df, max_season):
    train_games = games_df[games_df["season"] <= max_season]
    home_games = train_games[train_games["location"] == "Home"]
    if len(home_games) == 0:
        return {"home_win_rate": 0.55}
    home_wins = (home_games["winner"] == home_games["home_team"]).sum()
    return {"home_win_rate": home_wins / len(home_games)}


def baseline_probability(home_seed, away_seed, is_neutral, baselines):
    seed_diff = away_seed - home_seed
    if is_neutral:
        base_prob = 0.50 + (seed_diff * 0.03)
    else:
        base_prob = baselines["home_win_rate"] + (seed_diff * 0.02)
    return np.clip(base_prob, 0.20, 0.85)


def fit_expected_win_pct_model(team_df, max_season):
    train_data = team_df[team_df["season"] <= max_season].dropna(subset=["win_pct", "net_epa"])
    if len(train_data) < 20:
        return {"intercept": 0.5, "slope": 2.0}
    X = sm.add_constant(train_data["net_epa"].values.reshape(-1, 1))
    model = sm.OLS(train_data["win_pct"].values, X).fit()
    return {"intercept": model.params[0], "slope": model.params[1]}


# ============================================================
# Z-SCORES
# ============================================================

def compute_season_zscores(team_df):
    team_df = team_df.copy()

    higher_is_better = [
        "passing_epa", "rushing_epa", "total_offensive_epa", "net_epa",
        "point_differential", "win_pct", "pass_rush_rating",
        "pressure_rate", "sack_rate", "pass_block_rating", "protection_rate",
    ]
    lower_is_better = [
        "defensive_epa", "defensive_pass_epa", "defensive_rush_epa", "sacks_allowed_rate"
    ]

    for stat in higher_is_better:
        if stat in team_df.columns:
            team_df[f"z_{stat}"] = team_df.groupby("season")[stat].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )

    for stat in lower_is_better:
        if stat in team_df.columns:
            team_df[f"z_{stat}"] = team_df.groupby("season")[stat].transform(
                lambda x: -(x - x.mean()) / (x.std() + 1e-9)
            )

    return team_df


# ============================================================
# MATCHUP FEATURES
# ============================================================

def compute_matchup_features(away_data, home_data):
    features = {}

    def get_z(data, stat, default=0.0):
        z_col = f"z_{stat}"
        return float(data[z_col]) if (z_col in data.index and pd.notna(data[z_col])) else float(default)

    home_pass_off = get_z(home_data, "passing_epa")
    home_rush_off = get_z(home_data, "rushing_epa")
    away_pass_off = get_z(away_data, "passing_epa")
    away_rush_off = get_z(away_data, "rushing_epa")

    home_pass_def = get_z(home_data, "defensive_pass_epa")
    home_rush_def = get_z(home_data, "defensive_rush_epa")
    away_pass_def = get_z(away_data, "defensive_pass_epa")
    away_rush_def = get_z(away_data, "defensive_rush_epa")

    home_pass_edge = home_pass_off - away_pass_def
    home_rush_edge = home_rush_off - away_rush_def
    away_pass_edge = away_pass_off - home_pass_def
    away_rush_edge = away_rush_off - home_rush_def

    features["delta_pass_edge"] = home_pass_edge - away_pass_edge
    features["delta_rush_edge"] = home_rush_edge - away_rush_edge

    home_ol = get_z(home_data, "protection_rate")
    away_ol = get_z(away_data, "protection_rate")
    home_dl = get_z(home_data, "pressure_rate")
    away_dl = get_z(away_data, "pressure_rate")

    home_ol_exposure = float(np.maximum(0, away_dl - home_ol))
    away_ol_exposure = float(np.maximum(0, home_dl - away_ol))

    features["home_ol_exposure"] = home_ol_exposure
    features["away_ol_exposure"] = away_ol_exposure
    features["delta_ol_exposure"] = away_ol_exposure - home_ol_exposure
    features["total_ol_exposure"] = home_ol_exposure + away_ol_exposure

    home_pass_d_exposure = float(np.maximum(0, away_pass_off - home_pass_def))
    away_pass_d_exposure = float(np.maximum(0, home_pass_off - away_pass_def))

    features["home_pass_d_exposure"] = home_pass_d_exposure
    features["away_pass_d_exposure"] = away_pass_d_exposure
    features["delta_pass_d_exposure"] = away_pass_d_exposure - home_pass_d_exposure
    features["total_pass_d_exposure"] = home_pass_d_exposure + away_pass_d_exposure

    features["delta_rush_d_exposure"] = float(
        np.maximum(0, home_rush_off - away_rush_def) - np.maximum(0, away_rush_off - home_rush_def)
    )

    features["home_ol_z"] = home_ol
    features["away_ol_z"] = away_ol
    features["home_dl_z"] = home_dl
    features["away_dl_z"] = away_dl

    return features


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_interaction_features(features_dict):
    interactions = {}
    if "delta_pass_edge" in features_dict and "delta_ol_exposure" in features_dict:
        interactions["pass_x_ol"] = float(features_dict["delta_pass_edge"] * features_dict["delta_ol_exposure"])
    if "seed_diff" in features_dict and "delta_net_epa" in features_dict:
        interactions["seed_x_epa"] = float(features_dict["seed_diff"] * features_dict["delta_net_epa"])
    return interactions


def create_spread_features(home_spread):
    if pd.isna(home_spread):
        return {
            "has_spread": 0,
            "spread_magnitude": 0.0,
            "spread_confidence": 0.0,
            "is_close_game": 0,
        }
    spread = abs(float(home_spread))
    return {
        "has_spread": 1,
        "spread_magnitude": float(spread),
        "spread_confidence": float(1.0 / (spread + 3.0)),
        "is_close_game": 1 if spread <= 3.0 else 0,
    }


def create_round_features(game_type):
    return {"is_superbowl": 1 if game_type == "SB" else 0}


def create_tossup_features(seed_diff, home_spread):
    abs_seed_diff = abs(int(seed_diff))
    has_spread = pd.notna(home_spread)

    if has_spread:
        abs_spread = abs(float(home_spread))
        close_spread = (abs_spread <= 3.5)
    else:
        close_spread = True  # missing spread => treat as uncertain/close

    is_tossup = 1 if (abs_seed_diff <= 1 and close_spread) else 0

    return {
        "is_tossup": int(is_tossup),
        "is_big_favorite": 1 if abs_seed_diff >= 4 else 0,
        "is_medium_gap": 1 if abs_seed_diff in (2, 3) else 0,
        "is_close_seeds": 1 if abs_seed_diff <= 1 else 0,
        "tossup_baseline": 1 if ((not has_spread) and (abs_seed_diff <= 1)) else 0,
    }


# ============================================================
# SPREAD MODEL
# ============================================================

def fit_spread_model_v14(features_df, train_seasons, baselines, alpha=0.5,
                         clamp_band=(-0.12, -0.04),
                         prior_slope_per_point=-0.073,
                         prior_strength_k=60):
    train_df = features_df[
        (features_df["season"].isin(train_seasons)) & (features_df["home_spread"].notna())
    ].copy()

    n = len(train_df)
    if n < 20:
        return None

    spread_td = train_df["home_spread"].values / 7.0
    X = sm.add_constant(spread_td)
    y = train_df["home_wins"].values

    glm = sm.GLM(y, X, family=sm.families.Binomial())
    model = glm.fit_regularized(alpha=alpha, L1_wt=0.0)

    intercept_hat = float(model.params[0])
    slope_td_hat = float(model.params[1])
    slope_per_point_hat = slope_td_hat / 7.0

    slope_per_point_clamped = float(np.clip(slope_per_point_hat, clamp_band[0], clamp_band[1]))

    w_prior = float(n / (n + prior_strength_k))
    prior_intercept = float(logit(np.clip(baselines["home_win_rate"], 0.05, 0.95)))

    slope_per_point_final = w_prior * slope_per_point_clamped + (1 - w_prior) * prior_slope_per_point
    intercept_final = w_prior * intercept_hat + (1 - w_prior) * prior_intercept

    return {
        "n": n,
        "alpha": alpha,
        "w_prior": w_prior,
        "intercept": intercept_final,
        "slope_per_point": slope_per_point_final,
    }


def tune_spread_alpha_v14(features_df, train_seasons, val_seasons, baselines,
                          alphas=(0.1, 0.25, 0.5, 1.0, 2.0),
                          verbose=True):
    if verbose:
        print("\nTuning spread alpha...")

    best_alpha, best_ll, best_model = None, float("inf"), None

    val_df = features_df[
        (features_df["season"].isin(val_seasons)) & (features_df["home_spread"].notna())
    ].copy()

    if len(val_df) < 5:
        if verbose:
            print("  Not enough spread games, using default α=0.5")
        return fit_spread_model_v14(features_df, train_seasons, baselines, alpha=0.5)

    for a in alphas:
        smod = fit_spread_model_v14(features_df, train_seasons, baselines, alpha=a)
        if smod is None:
            continue

        probs = expit(np.clip(smod["intercept"] + smod["slope_per_point"] * val_df["home_spread"].values, -4, 4))
        y = val_df["home_wins"].values
        ll = -np.mean(np.log(np.clip(np.where(y == 1, probs, 1 - probs), 0.01, 0.99)))

        if verbose:
            print(f"  α={a:<4}  val_ll={ll:.4f}  slope/pt={smod['slope_per_point']:+.4f}")

        if ll < best_ll:
            best_ll, best_alpha, best_model = ll, a, smod

    if verbose and best_model:
        print(f"  Best α={best_alpha}")
    return best_model


def print_spread_model(spread_model):
    if spread_model is None:
        print("Spread Model: None")
        return

    print("\nSpread Model:")
    print(f"  n={spread_model['n']}, α={spread_model['alpha']}")
    print(f"  logit(P(home)) = {spread_model['intercept']:.3f} + {spread_model['slope_per_point']:+.4f} * spread")
    print("  Implied probs:")
    for pts in [-10, -7, -3, 0, 3, 7, 10]:
        p = expit(np.clip(spread_model["intercept"] + spread_model["slope_per_point"] * pts, -4, 4))
        print(f"    {pts:+3d}: {p:.1%}")


def get_spread_offset_logit(home_spread, spread_model):
    if spread_model is None or pd.isna(home_spread):
        return np.nan
    offset = spread_model["intercept"] + spread_model["slope_per_point"] * float(home_spread)
    return float(np.clip(offset, -4, 4))


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_game_features_v16(games_df, team_df, epa_model, baselines, spread_model):
    team_df = compute_season_zscores(team_df)
    rows = []

    for _, game in games_df.iterrows():
        season = int(game["season"])
        is_neutral = (game["location"] == "Neutral")

        orig_away, orig_home = game["away_team"], game["home_team"]

        away_row = team_df[(team_df["team"] == orig_away) & (team_df["season"] == season)]
        home_row = team_df[(team_df["team"] == orig_home) & (team_df["season"] == season)]
        if len(away_row) == 0 or len(home_row) == 0:
            continue

        away_data = away_row.iloc[0]
        home_data = home_row.iloc[0]

        if pd.isna(game.get("away_offensive_epa")) or pd.isna(game.get("home_offensive_epa")):
            continue

        away_seed = int(away_data["playoff_seed"])
        home_seed = int(home_data["playoff_seed"])

        home_spread = game.get("home_spread", np.nan)
        away, home = orig_away, orig_home

        if is_neutral and away_seed < home_seed:
            away, home = orig_home, orig_away
            away_data, home_data = home_data, away_data
            away_seed, home_seed = home_seed, away_seed
            if pd.notna(home_spread):
                home_spread = -float(home_spread)

        away_games = float(away_data["wins"] + away_data["losses"])
        home_games = float(home_data["wins"] + home_data["losses"])
        away_pd_pg = float(away_data["point_differential"] / max(away_games, 1))
        home_pd_pg = float(home_data["point_differential"] / max(home_games, 1))
        away_net_epa = float(away_data.get("net_epa", 0) or 0)
        home_net_epa = float(home_data.get("net_epa", 0) or 0)

        matchup = compute_matchup_features(away_data, home_data)

        away_exp = float(epa_model["intercept"] + epa_model["slope"] * away_net_epa)
        home_exp = float(epa_model["intercept"] + epa_model["slope"] * home_net_epa)
        away_win_pct = float(away_data["win_pct"])
        home_win_pct = float(home_data["win_pct"])

        season_teams = team_df[team_df["season"] == season].copy()
        season_teams["pd_pg"] = season_teams["point_differential"] / (
            (season_teams["wins"] + season_teams["losses"]).clip(lower=1)
        )
        season_teams["qrank"] = season_teams["pd_pg"].rank(ascending=False)

        away_q = season_teams[season_teams["team"] == away]["qrank"].values
        home_q = season_teams[season_teams["team"] == home]["qrank"].values
        away_quality_rank = int(away_q[0]) if len(away_q) else len(season_teams) // 2
        home_quality_rank = int(home_q[0]) if len(home_q) else len(season_teams) // 2

        away_mom = float(away_data.get("momentum_residual", 0) or 0)
        home_mom = float(home_data.get("momentum_residual", 0) or 0)

        baseline_prob = float(baseline_probability(home_seed, away_seed, is_neutral, baselines))
        baseline_logit = float(logit(np.clip(baseline_prob, 0.01, 0.99)))
        spread_offset = get_spread_offset_logit(home_spread, spread_model)

        if pd.notna(spread_offset):
            offset_logit = float(spread_offset)
            offset_source = "spread"
            spread_prob = float(expit(spread_offset))
        else:
            offset_logit = float(baseline_logit)
            offset_source = "baseline"
            spread_prob = np.nan

        actual_winner = game["winner"]
        home_wins = 1 if actual_winner == home else 0

        seed_diff = away_seed - home_seed

        row = {
            "season": season,
            "game_type": game["game_type"],
            "away_team": away,
            "home_team": home,
            "orig_away_team": orig_away,
            "orig_home_team": orig_home,
            "winner": actual_winner,
            "away_score": game["away_score"],
            "home_score": game["home_score"],
            "home_wins": home_wins,
            "is_neutral": 1 if is_neutral else 0,
            "away_seed": away_seed,
            "home_seed": home_seed,
            "seed_diff": seed_diff,
            "seed_diff_sq": seed_diff ** 2,
            "seed_diff_abs": abs(seed_diff),

            "delta_net_epa": home_net_epa - away_net_epa,
            "delta_pd_pg": home_pd_pg - away_pd_pg,
            "delta_vulnerability": (away_win_pct - away_exp) - (home_win_pct - home_exp),
            "delta_underseeded": (away_seed - away_quality_rank) - (home_seed - home_quality_rank),
            "delta_momentum": home_mom - away_mom,

            "baseline_prob": baseline_prob,
            "baseline_logit": baseline_logit,
            "home_spread": home_spread,
            "spread_offset": spread_offset,
            "spread_prob": spread_prob,
            "offset_logit": offset_logit,
            "offset_source": offset_source,
        }

        row.update(matchup)
        row.update(create_interaction_features(row))
        row.update(create_spread_features(home_spread))
        row.update(create_round_features(game["game_type"]))
        row.update(create_tossup_features(seed_diff, home_spread))

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# RIDGE MODEL (baseline-only)
# ============================================================

def train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=1.0):
    train_df = features_df[
        (features_df["season"].isin(train_seasons)) &
        (features_df["offset_source"] == "baseline")
    ].dropna(subset=feature_cols + ["offset_logit"])

    if len(train_df) < 30:
        return None

    X_raw = train_df[feature_cols].values
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0) + 1e-9
    X_scaled = (X_raw - mu) / sd

    X = sm.add_constant(X_scaled)
    y = train_df["home_wins"].values
    offset = train_df["offset_logit"].values

    try:
        res = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit_regularized(alpha=alpha, L1_wt=0.0)
    except Exception:
        try:
            res = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit()
        except Exception:
            return None

    return {
        "model": res,
        "feature_cols": feature_cols,
        "mu": mu,
        "sd": sd,
        "train_samples": len(train_df),
        "alpha": alpha,
    }


def predict_raw(model_dict, game_features):
    if model_dict is None:
        return None

    try:
        X_raw = np.array([[game_features[col] for col in model_dict["feature_cols"]]], dtype=float)
    except KeyError:
        return None

    offset_logit = float(game_features["offset_logit"])
    if np.any(np.isnan(X_raw)):
        p = float(expit(np.clip(offset_logit, -4, 4)))
        return {"raw_prob": p, "adjustment": 0.0, "offset_logit": offset_logit}

    X_scaled = (X_raw - model_dict["mu"]) / model_dict["sd"]
    X = sm.add_constant(X_scaled, has_constant="add")

    p = float(model_dict["model"].predict(X, offset=np.array([offset_logit]))[0])
    p = float(np.clip(p, 0.01, 0.99))

    adj = float(logit(p) - offset_logit)
    return {"raw_prob": p, "adjustment": adj, "offset_logit": offset_logit}


# ============================================================
# XGBOOST + REGULARIZED PLATT
# ============================================================

def _fit_platt_regularized(raw_probs, y, alpha=1.0, shrinkage=0.25):
    """
    Robust Platt:
    - requires both classes in y
    - uses regularized GLM for stability
    - shrinks toward identity (a=1,b=0)
    """
    y = np.asarray(y).astype(int)
    if len(y) < 25:
        return 1.0, 0.0, False
    if y.min() == y.max():
        return 1.0, 0.0, False

    raw_probs = np.clip(np.asarray(raw_probs), 0.02, 0.98)
    raw_logits = logit(raw_probs)

    if np.std(raw_logits) < 1e-6:
        return 1.0, 0.0, False

    X = sm.add_constant(raw_logits, has_constant="add")

    try:
        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit_regularized(alpha=alpha, L1_wt=0.0)
        params = np.asarray(fit.params)
        if params.shape[0] < 2:
            return 1.0, 0.0, False
        b_raw, a_raw = float(params[0]), float(params[1])
    except Exception:
        return 1.0, 0.0, False

    # shrink toward identity
    a = 1.0 + shrinkage * (a_raw - 1.0)
    b = 0.0 + shrinkage * b_raw

    # clamp to keep it sane
    a = float(np.clip(a, 0.6, 1.8))
    b = float(np.clip(b, -1.5, 1.5))

    return a, b, True


def train_xgboost_calibrated(features_df, feature_cols, train_seasons, calib_seasons,
                             n_estimators=500, max_depth=2, learning_rate=0.03):
    if not HAS_XGBOOST:
        return None

    train_df = features_df[features_df["season"].isin(train_seasons)].dropna(subset=feature_cols)
    if len(train_df) < 30:
        return None

    X_train = train_df[feature_cols].values
    y_train = train_df["home_wins"].values

    try:
        xgb = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
        xgb.fit(X_train, y_train)
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        return None

    calib_df = features_df[features_df["season"].isin(calib_seasons)].dropna(subset=feature_cols)
    if len(calib_df) < 25:
        print("  Warning: Not enough calibration data for XGBoost Platt scaling (need ~25+)")
        return {
            "model": xgb,
            "feature_cols": feature_cols,
            "train_samples": len(train_df),
            "calib_samples": len(calib_df),
            "type": "xgboost",
            "calibrated": False,
            "platt_a": 1.0,
            "platt_b": 0.0,
        }

    X_calib = calib_df[feature_cols].values
    y_calib = calib_df["home_wins"].values
    raw_probs = xgb.predict_proba(X_calib)[:, 1]

    a, b, ok = _fit_platt_regularized(raw_probs, y_calib, alpha=1.0, shrinkage=0.25)
    if ok:
        print(f"  XGBoost Platt (reg): a={a:.3f}, b={b:.3f} (n={len(calib_df)})")
    else:
        print("  Warning: XGBoost Platt unstable; using identity")

    return {
        "model": xgb,
        "feature_cols": feature_cols,
        "train_samples": len(train_df),
        "calib_samples": len(calib_df),
        "type": "xgboost",
        "calibrated": ok,
        "platt_a": a,
        "platt_b": b,
    }


def predict_xgboost_calibrated(model_dict, game_features):
    if model_dict is None or model_dict.get("type") != "xgboost":
        return None

    try:
        X = np.array([[game_features[col] for col in model_dict["feature_cols"]]], dtype=float)
        if np.any(np.isnan(X)):
            return None

        raw_prob = float(model_dict["model"].predict_proba(X)[0, 1])

        raw_logit = logit(np.clip(raw_prob, 0.02, 0.98))
        cal_logit = model_dict["platt_b"] + model_dict["platt_a"] * raw_logit
        cal_prob = float(expit(np.clip(cal_logit, -4, 4)))

        return {"raw_prob": raw_prob, "calibrated_prob": cal_prob}
    except Exception:
        return None


# ============================================================
# LAMBDA TUNING (keep λ_spread=0)
# ============================================================

def tune_source_lambdas(model_dict, features_df, tune_seasons, improve_gate=0.025, verbose=True):
    tune_df = features_df[
        (features_df["season"].isin(tune_seasons)) &
        (features_df["offset_source"] == "baseline")
    ].dropna(subset=model_dict["feature_cols"] + ["offset_logit"])

    best = {"spread": 0.0, "base": 1.0}
    if verbose:
        print("  λ_spread forced to 0.0 (trust market)")

    # baseline lambda
    if len(tune_df) < 10:
        return best

    best_lam, best_imp = 0.0, -float("inf")
    for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        model_lls, offset_lls = [], []
        for _, r in tune_df.iterrows():
            pr = predict_raw(model_dict, r)
            if pr is None:
                continue
            p = float(expit(np.clip(pr["offset_logit"] + lam * pr["adjustment"], -4, 4)))
            actual = int(r["home_wins"])
            model_lls.append(-np.log(np.clip(p if actual else 1 - p, 0.01, 0.99)))
            off_p = float(expit(float(r["offset_logit"])))
            offset_lls.append(-np.log(np.clip(off_p if actual else 1 - off_p, 0.01, 0.99)))

        if model_lls:
            imp = float(np.mean(offset_lls) - np.mean(model_lls))
            if imp > best_imp:
                best_imp, best_lam = imp, lam

    best["base"] = float(best_lam)
    if verbose:
        print(f"  λ_base = {best_lam:.1f} (Δll={best_imp:+.4f})")
    return best


# ============================================================
# ENSEMBLE + CAPS + TOSSUP SHRINK
# ============================================================

def predict_ensemble_v16(model_dict, game_features, lambda_params,
                         xgb_model=None, xgb_weight=0.35,
                         adj_cap=0.80,
                         tossup_shrink=0.35):
    """
    Spread games: return offset prob directly.
    Baseline games:
      - ridge produces adjustment, but we cap it (logit units)
      - optional XGB ensemble
      - if is_tossup==1, shrink toward offset (reduces noisy flips)
    """
    src = game_features["offset_source"]

    if src == "spread":
        p = float(expit(np.clip(float(game_features["offset_logit"]), -4, 4)))
        return {"home_prob": p, "predicted_winner": "home" if p > 0.5 else "away", "is_ensemble": False}

    pr = predict_raw(model_dict, game_features)
    if pr is None:
        return None

    lam = float(lambda_params["base"])

    # cap the adjustment to prevent huge flips (THIS FIXES LAC/HOU style swings)
    adj = float(np.clip(pr["adjustment"], -adj_cap, adj_cap))

    blended_logit = float(np.clip(pr["offset_logit"] + lam * adj, -4, 4))
    ridge_prob = float(expit(blended_logit))

    final_prob = ridge_prob
    xgb_pred = None

    if xgb_model is not None:
        xgb_pred = predict_xgboost_calibrated(xgb_model, game_features)
        if xgb_pred is not None:
            xgb_prob = float(xgb_pred["calibrated_prob"])
            final_prob = float(np.clip((1 - xgb_weight) * ridge_prob + xgb_weight * xgb_prob, 0.01, 0.99))

    # tossup shrink toward offset (seed gap <= 1 only)
    if int(game_features.get("is_tossup", 0)) == 1:
        off_p = float(expit(float(game_features["offset_logit"])))
        final_prob = float(np.clip((1 - tossup_shrink) * final_prob + tossup_shrink * off_p, 0.01, 0.99))

    return {
        "home_prob": final_prob,
        "predicted_winner": "home" if final_prob > 0.5 else "away",
        "ridge_prob": ridge_prob,
        "xgb_prob": None if xgb_pred is None else float(xgb_pred["calibrated_prob"]),
        "is_ensemble": xgb_pred is not None
    }


# ============================================================
# EVALUATION HELPERS
# ============================================================

def evaluate_predictions(results_df):
    out = {"n_games": len(results_df)}
    out["accuracy"] = float(results_df["correct"].mean()) if len(results_df) else 0.0
    probs = results_df.apply(
        lambda r: r["home_prob"] if int(r["actual_home_wins"]) == 1 else (1 - r["home_prob"]),
        axis=1,
    )
    out["log_loss"] = float(-np.mean(np.log(probs.clip(0.01, 0.99)))) if len(results_df) else np.nan
    return out


# ============================================================
# 2024 VALIDATION
# ============================================================

def validate_on_2024(model_dict, features_df, lambda_params,
                     xgb_model=None, xgb_weight=0.35):
    print("\n" + "=" * 140)
    ensemble_str = f" + XGB(full) ensemble (weight={xgb_weight:.1f})" if xgb_model else ""
    print(f"2024 PLAYOFF VALIDATION - MODEL v16.2 (λ_spread=0.0, λ_base={lambda_params['base']:.1f}{ensemble_str})")
    print("=" * 140)

    test_df = features_df[features_df["season"] == 2024].copy()
    round_names = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference", "SB": "Super Bowl"}

    rows = []
    for _, gf in test_df.iterrows():
        pred = predict_ensemble_v16(
            model_dict, gf, lambda_params,
            xgb_model=xgb_model, xgb_weight=xgb_weight,
            adj_cap=0.80,
            tossup_shrink=0.35
        )
        if pred is None:
            continue

        pred_team = gf["home_team"] if pred["predicted_winner"] == "home" else gf["away_team"]
        actual = gf["winner"]
        correct = (pred_team == actual)

        rows.append({
            "round": round_names.get(gf["game_type"], gf["game_type"]),
            "matchup": f"{gf['orig_away_team']} @ {gf['orig_home_team']}",
            "offset_source": gf["offset_source"],
            "home_prob": float(pred["home_prob"]),
            "offset_prob": float(expit(float(gf["offset_logit"]))),
            "predicted": pred_team,
            "actual": actual,
            "correct": bool(correct),
            "actual_home_wins": 1 if actual == gf["home_team"] else 0,
            "is_tossup": int(gf.get("is_tossup", 0)),
        })

    results_df = pd.DataFrame(rows)

    for rname in ["Wild Card", "Divisional", "Conference", "Super Bowl"]:
        rg = results_df[results_df["round"] == rname]
        if len(rg) == 0:
            continue
        print(f"\n{rname.upper()}")
        print(f"{'Matchup':<16} {'Src':<5} {'Final':>6} {'Off':>6} {'T':>2} {'Pred':<5} {'Act':<5} ✓")
        print("-" * 80)
        for _, row in rg.iterrows():
            src = "sprd" if row["offset_source"] == "spread" else "base"
            mark = "✓" if row["correct"] else "✗"
            toss = "T" if row["is_tossup"] else ""
            print(f"{row['matchup']:<16} {src:<5} {row['home_prob']:>5.0%} {row['offset_prob']:>5.0%} {toss:>2} {row['predicted']:<5} {row['actual']:<5} {mark}")

    print("\n" + "=" * 140)
    print("METRICS - MODEL v16.2")
    print("=" * 140)
    m = evaluate_predictions(results_df)
    print(f"\nAccuracy: {results_df['correct'].sum()}/{len(results_df)} ({m['accuracy']:.1%})")
    print(f"Log loss: {m['log_loss']:.4f}")

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("NFL PLAYOFF MODEL v16.2 - STABILITY PATCHES")
    print("=" * 100)

    games_df, team_df, spread_df = load_all_data()
    games_df = merge_spread_data_safe(games_df, spread_df)
    spread_sanity_check(games_df)

    train_seasons = list(range(2000, 2023))
    calib_seasons = [2021, 2022, 2023]
    tune_seasons = list(range(2015, 2024))
    spread_val_seasons = [2020, 2021, 2022, 2023]
    xgb_calib_seasons = [2021, 2022, 2023]

    baselines = compute_historical_baselines(games_df, max_season=2023)
    print(f"\nHistorical Baselines: Home win rate = {baselines['home_win_rate']:.1%}")

    epa_model = fit_expected_win_pct_model(team_df, max_season=2023)
    print(f"EPA -> Win%: {epa_model['intercept']:.3f} + {epa_model['slope']:.3f} * net_epa")

    temp_features = prepare_game_features_v16(games_df, team_df, epa_model, baselines, spread_model=None)

    spread_model = tune_spread_alpha_v14(
        temp_features,
        train_seasons=train_seasons,
        val_seasons=spread_val_seasons,
        baselines=baselines,
        alphas=(0.1, 0.25, 0.5, 1.0, 2.0)
    )
    print_spread_model(spread_model)

    print("\nPreparing v16.2 features...")
    features_df = prepare_game_features_v16(games_df, team_df, epa_model, baselines, spread_model)
    print(f"Total games: {len(features_df)}")

    feature_cols = [
        "delta_net_epa", "delta_pd_pg", "seed_diff",
        "delta_vulnerability", "delta_underseeded", "delta_momentum",
        "delta_pass_edge", "delta_rush_edge",
        "delta_ol_exposure", "delta_pass_d_exposure",
        "home_ol_exposure", "away_ol_exposure",
        "total_ol_exposure", "total_pass_d_exposure",
        "pass_x_ol", "seed_x_epa",
        "seed_diff_sq", "seed_diff_abs",
        "is_superbowl",
        "has_spread", "spread_magnitude", "spread_confidence", "is_close_game",
        "is_tossup", "tossup_baseline", "is_big_favorite", "is_medium_gap", "is_close_seeds",
    ]
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    print("\nTuning ridge alpha on 2023 (baseline only)...")
    best_alpha, best_ll = None, float("inf")
    for a in [0.5, 1.0, 2.0, 5.0]:
        md = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=a)
        if md is None:
            continue
        val_df = features_df[
            (features_df["season"] == 2023) & (features_df["offset_source"] == "baseline")
        ].dropna(subset=feature_cols + ["offset_logit"])

        preds = []
        for _, r in val_df.iterrows():
            pr = predict_raw(md, r)
            if pr is None:
                continue
            p = pr["raw_prob"]
            actual = int(r["home_wins"])
            preds.append(p if actual == 1 else (1 - p))

        if preds:
            ll = float(-np.mean(np.log(np.clip(preds, 0.01, 0.99))))
            print(f"  α={a}: ll={ll:.4f}")
            if ll < best_ll:
                best_ll, best_alpha = ll, a

    model_dict = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=best_alpha)
    print(f"\nBest α={best_alpha}, trained on {model_dict['train_samples']} BASELINE games")

    # XGB(full) for ensemble stability
    xgb_model_full = None
    if HAS_XGBOOST:
        print("\nTraining calibrated XGBoost (full) for ensemble...")
        xgb_model_full = train_xgboost_calibrated(features_df, feature_cols, train_seasons, xgb_calib_seasons)
        if xgb_model_full:
            print(f"  XGB(full) trained on {xgb_model_full['train_samples']} games; calibrated={xgb_model_full['calibrated']}")

    print("\nTuning λ (baseline only; spread fixed at 0)...")
    lambda_params = tune_source_lambdas(model_dict, features_df, tune_seasons, improve_gate=0.025, verbose=True)

    # modest default; you can grid search this like before if you want
    xgb_weight = 0.35

    results_2024 = validate_on_2024(
        model_dict, features_df, lambda_params,
        xgb_model=xgb_model_full, xgb_weight=xgb_weight
    )

    out_path = "../validation_results_2024_v16_2.csv"
    results_2024.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return results_2024


if __name__ == "__main__":
    main()
