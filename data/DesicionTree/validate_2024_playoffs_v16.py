"""
NFL Playoff Model v16 — v15.1 + XGBoost Calibration & Dynamic Ensemble

NEW vs v15.1:
- Calibrated XGBoost (Platt scaling to fix overconfidence)
- Dynamic ensemble weight tuning (optimize on validation set)
- Toss-up detector feature (games with seed_diff <= 1 and close spread)
- Seed-tier specific analysis (big favorites vs close games)
- Improved upset detection for close games

PRESERVED from v15.1:
- λ_spread=0.0 (trust market for spread games)
- Simplified feature set (22 features)
- XGBoost ensemble for baseline games
- Conservative calibration

NOTES:
- XGBoost was showing 98-99% probabilities (overconfident)
- Platt scaling brings XGBoost in line with ridge calibration
- Dynamic weight allows model to learn optimal blend
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
import warnings
warnings.filterwarnings("ignore")

# Optional XGBoost
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
# FEATURE ENGINEERING (v15.1 + v16 additions)
# ============================================================

def create_interaction_features(features_dict):
    """Simplified interactions (v15.1)"""
    interactions = {}
    
    if "delta_pass_edge" in features_dict and "delta_ol_exposure" in features_dict:
        interactions["pass_x_ol"] = float(
            features_dict["delta_pass_edge"] * features_dict["delta_ol_exposure"]
        )
    
    if "seed_diff" in features_dict and "delta_net_epa" in features_dict:
        interactions["seed_x_epa"] = float(
            features_dict["seed_diff"] * features_dict["delta_net_epa"]
        )
    
    return interactions


def create_spread_features(home_spread):
    """Better spread-derived features (v15.1)"""
    if pd.isna(home_spread):
        return {
            "spread_magnitude": 0.0,
            "spread_confidence": 0.5,
            "is_close_game": 0,
        }
    
    spread = abs(float(home_spread))
    
    return {
        "spread_magnitude": spread,
        "spread_confidence": 1.0 / (spread + 3.0),
        "is_close_game": 1 if spread <= 3.0 else 0,
    }


def create_round_features(game_type):
    """Only Super Bowl indicator (v15.1)"""
    return {"is_superbowl": 1 if game_type == "SB" else 0}


def create_tossup_features(seed_diff, home_spread):
    """
    NEW v16: Toss-up detector for close games.
    These games have different dynamics - need special handling.
    """
    abs_seed_diff = abs(seed_diff)
    abs_spread = abs(home_spread) if pd.notna(home_spread) else 7.0  # Default to medium spread
    
    features = {
        # Toss-up: small seed gap AND close spread (or baseline game)
        "is_tossup": 1 if abs_seed_diff <= 1 and abs_spread <= 3.5 else 0,
        
        # Seed tier indicators for potential tier-specific modeling
        "is_big_favorite": 1 if abs_seed_diff >= 4 else 0,  # 7v3, 6v2, 5v1, etc.
        "is_medium_gap": 1 if abs_seed_diff == 2 or abs_seed_diff == 3 else 0,
        "is_close_seeds": 1 if abs_seed_diff <= 1 else 0,
        
        # Interaction: toss-up game that's also baseline (no spread) = highest uncertainty
        "tossup_baseline": 0,  # Will be set later if baseline
    }
    
    return features


# ============================================================
# SPREAD MODEL (unchanged from v15.1)
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
# FEATURE PREPARATION (v16)
# ============================================================

def prepare_game_features_v16(games_df, team_df, epa_model, baselines, spread_model):
    """
    v16 feature preparation with toss-up detection.
    """
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

        # Core stats
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

        # Interactions (v15.1)
        row.update(create_interaction_features(row))

        # Spread features (v15.1)
        row.update(create_spread_features(home_spread))

        # Round features (v15.1)
        row.update(create_round_features(game["game_type"]))

        # NEW v16: Toss-up features
        tossup_features = create_tossup_features(seed_diff, home_spread)
        # Set tossup_baseline if this is a baseline game and a tossup
        if offset_source == "baseline" and tossup_features["is_tossup"]:
            tossup_features["tossup_baseline"] = 1
        row.update(tossup_features)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# RIDGE MODEL (unchanged from v15.1)
# ============================================================

def train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=1.0):
    train_df = features_df[
        features_df["season"].isin(train_seasons)
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
# NEW v16: CALIBRATED XGBOOST
# ============================================================

def train_xgboost_calibrated(features_df, feature_cols, train_seasons, calib_seasons,
                             n_estimators=100, max_depth=3, learning_rate=0.1):
    """
    v16: Train XGBoost with Platt scaling calibration.
    Fixes the overconfidence issue (98-99% predictions).
    """
    if not HAS_XGBOOST:
        return None
    
    train_df = features_df[
        features_df["season"].isin(train_seasons)
    ].dropna(subset=feature_cols + ["offset_logit"])

    if len(train_df) < 30:
        return None

    X_train = train_df[feature_cols].values
    y_train = train_df["home_wins"].values

    # Train XGBoost
    try:
        xgb = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=42
        )
        xgb.fit(X_train, y_train)
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        return None

    # Calibrate with Platt scaling on calib_seasons
    calib_df = features_df[
        features_df["season"].isin(calib_seasons)
    ].dropna(subset=feature_cols + ["offset_logit"])

    if len(calib_df) < 10:
        # Not enough calibration data, use uncalibrated
        print("  Warning: Not enough calibration data for XGBoost Platt scaling")
        return {
            "model": xgb,
            "feature_cols": feature_cols,
            "train_samples": len(train_df),
            "type": "xgboost",
            "calibrated": False,
            "platt_a": 1.0,
            "platt_b": 0.0,
        }

    X_calib = calib_df[feature_cols].values
    y_calib = calib_df["home_wins"].values

    # Get raw probabilities
    raw_probs = xgb.predict_proba(X_calib)[:, 1]
    
    # Platt scaling: fit logistic regression on log-odds
    raw_logits = logit(np.clip(raw_probs, 0.01, 0.99))
    
    try:
        X_platt = sm.add_constant(raw_logits)
        platt_fit = sm.GLM(y_calib, X_platt, family=sm.families.Binomial()).fit()
        platt_b = float(platt_fit.params[0])
        platt_a = float(platt_fit.params[1])
        
        # Sanity check
        if platt_a <= 0 or abs(platt_a) > 5 or abs(platt_b) > 3:
            print(f"  Warning: Platt params unstable (a={platt_a:.2f}, b={platt_b:.2f}), using identity")
            platt_a, platt_b = 1.0, 0.0
            calibrated = False
        else:
            calibrated = True
            print(f"  XGBoost Platt scaling: a={platt_a:.3f}, b={platt_b:.3f}")
            
    except Exception as e:
        print(f"  Warning: Platt scaling failed: {e}")
        platt_a, platt_b = 1.0, 0.0
        calibrated = False

    return {
        "model": xgb,
        "feature_cols": feature_cols,
        "train_samples": len(train_df),
        "calib_samples": len(calib_df),
        "type": "xgboost",
        "calibrated": calibrated,
        "platt_a": platt_a,
        "platt_b": platt_b,
    }


def predict_xgboost_calibrated(model_dict, game_features):
    """Predict using calibrated XGBoost."""
    if model_dict is None or model_dict.get("type") != "xgboost":
        return None
    
    try:
        X = np.array([[game_features[col] for col in model_dict["feature_cols"]]], dtype=float)
        if np.any(np.isnan(X)):
            return None
        
        # Raw probability
        raw_prob = float(model_dict["model"].predict_proba(X)[0, 1])
        
        # Apply Platt scaling
        raw_logit = logit(np.clip(raw_prob, 0.01, 0.99))
        cal_logit = model_dict["platt_b"] + model_dict["platt_a"] * raw_logit
        cal_prob = float(expit(np.clip(cal_logit, -4, 4)))
        
        return {
            "raw_prob": raw_prob,
            "calibrated_prob": cal_prob,
        }
    except Exception:
        return None


# ============================================================
# NEW v16: DYNAMIC ENSEMBLE WEIGHT TUNING
# ============================================================

def tune_ensemble_weight(ridge_model, xgb_model, features_df, val_seasons, lambda_base=1.0):
    """
    v16: Find optimal XGBoost weight for ensemble.
    Only applied to baseline games.
    """
    if xgb_model is None:
        return 0.0
    
    val_df = features_df[
        (features_df["season"].isin(val_seasons)) & 
        (features_df["offset_source"] == "baseline")
    ].dropna(subset=ridge_model["feature_cols"] + ["offset_logit"])

    if len(val_df) < 10:
        print("  Not enough baseline validation games, using default weight=0.3")
        return 0.3

    best_weight, best_ll = 0.0, float("inf")

    print("\n  Tuning XGBoost ensemble weight on baseline games...")
    for weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        lls = []
        
        for _, r in val_df.iterrows():
            # Ridge prediction
            ridge_pred = predict_raw(ridge_model, r)
            if ridge_pred is None:
                continue
            
            # Apply lambda
            blended_logit = float(ridge_pred["offset_logit"] + lambda_base * ridge_pred["adjustment"])
            ridge_prob = float(expit(np.clip(blended_logit, -4, 4)))
            
            # XGBoost prediction (calibrated)
            xgb_pred = predict_xgboost_calibrated(xgb_model, r)
            if xgb_pred is None:
                ensemble_prob = ridge_prob
            else:
                xgb_prob = xgb_pred["calibrated_prob"]
                ensemble_prob = (1 - weight) * ridge_prob + weight * xgb_prob
            
            actual = int(r["home_wins"])
            prob = ensemble_prob if actual == 1 else (1 - ensemble_prob)
            lls.append(-np.log(np.clip(prob, 0.01, 0.99)))
        
        if lls:
            ll = np.mean(lls)
            print(f"    weight={weight:.1f}: ll={ll:.4f}")
            if ll < best_ll:
                best_ll, best_weight = ll, weight

    print(f"  Best XGBoost weight: {best_weight:.1f}")
    return best_weight


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_baseline_only(model_dict, features_df, calib_seasons, shrinkage=0.3):
    platt = {"spread": {"a": 1.0, "b": 0.0}, "baseline": {"a": 1.0, "b": 0.0}}

    calib_df = features_df[
        (features_df["season"].isin(calib_seasons)) & (features_df["offset_source"] == "baseline")
    ].dropna(subset=model_dict["feature_cols"] + ["offset_logit"])

    if len(calib_df) < 5:
        print("  Insufficient baseline games for calibration")
        return platt

    preds = []
    for _, r in calib_df.iterrows():
        pr = predict_raw(model_dict, r)
        if pr is None:
            continue
        preds.append({"prob": pr["raw_prob"], "actual": int(r["home_wins"])})

    if len(preds) < 5:
        print("  Insufficient valid baseline preds for calibration")
        return platt

    df = pd.DataFrame(preds)
    clipped = df["prob"].clip(0.05, 0.95)
    X = sm.add_constant(logit(clipped).values)

    try:
        fit = sm.GLM(df["actual"].values, X, family=sm.families.Binomial()).fit()
        a_raw, b_raw = float(fit.params[1]), float(fit.params[0])

        if a_raw < 0 or abs(a_raw) > 10 or abs(b_raw) > 5:
            print(f"  Baseline Platt unstable (a={a_raw:.2f}, b={b_raw:.2f}), using identity")
            return platt

        a = 1.0 + shrinkage * (a_raw - 1.0)
        b = 0.0 + shrinkage * b_raw

        a = float(np.clip(a, 0.8, 1.2))
        b = float(np.clip(b, -0.2, 0.2))

        platt["baseline"] = {"a": a, "b": b}
        print(f"  Baseline Platt (n={len(preds)}): a={a:.3f}, b={b:.3f}")

    except Exception as e:
        print(f"  Baseline Platt failed: {e}")

    return platt


# ============================================================
# LAMBDA TUNING (v15.1 - force spread to 0)
# ============================================================

def tune_source_lambdas(model_dict, features_df, tune_seasons, platt_params,
                        improve_gate=0.025, force_spread_zero=True, verbose=True):
    tune_df = features_df[features_df["season"].isin(tune_seasons)].dropna(
        subset=model_dict["feature_cols"] + ["offset_logit"]
    )

    best = {"spread": 0.0, "base": 1.0}

    for src, name in [("spread", "spread"), ("baseline", "base")]:
        if name == "spread" and force_spread_zero:
            best["spread"] = 0.0
            if verbose:
                print(f"  λ_spread forced to 0.0 (trust market)")
            continue
            
        df = tune_df[tune_df["offset_source"] == src]
        if len(df) < 10:
            continue

        best_lam, best_imp = 0.0, -float("inf")

        for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            model_lls, offset_lls = [], []

            for _, r in df.iterrows():
                pr = predict_raw(model_dict, r)
                if pr is None:
                    continue

                blended_logit = float(pr["offset_logit"] + lam * pr["adjustment"])
                p = float(expit(np.clip(blended_logit, -4, 4)))

                actual = int(r["home_wins"])
                model_prob = p if actual == 1 else (1 - p)

                off_p = float(expit(float(r["offset_logit"])))
                offset_prob = off_p if actual == 1 else (1 - off_p)

                model_lls.append(-np.log(np.clip(model_prob, 0.01, 0.99)))
                offset_lls.append(-np.log(np.clip(offset_prob, 0.01, 0.99)))

            if model_lls:
                imp = float(np.mean(offset_lls) - np.mean(model_lls))
                if imp > best_imp:
                    best_imp, best_lam = imp, lam

        if name == "spread" and best_imp < improve_gate:
            best["spread"] = 0.0
            if verbose:
                print(f"  λ_spread forced to 0.0 (Δll={best_imp:+.4f} < {improve_gate})")
        else:
            best[name] = float(best_lam)
            if verbose:
                print(f"  λ_{name:<5} = {best_lam:.1f} (Δll={best_imp:+.4f})")

    return best


# ============================================================
# PREDICTION WITH CALIBRATED ENSEMBLE
# ============================================================

def predict_with_source_lambda(model_dict, game_features, platt_params,
                               lambda_spread=0.0, lambda_base=1.0):
    pred = predict_raw(model_dict, game_features)
    if pred is None:
        return None

    src = game_features["offset_source"]
    lam = float(lambda_spread if src == "spread" else lambda_base)

    blended_logit = float(pred["offset_logit"] + lam * pred["adjustment"])
    blended_logit = float(np.clip(blended_logit, -4, 4))
    blended_prob = float(expit(blended_logit))

    pl = platt_params.get(src, {"a": 1.0, "b": 0.0})
    cal_logit = float(pl["b"] + pl["a"] * logit(np.clip(blended_prob, 0.05, 0.95)))
    cal_logit = float(np.clip(cal_logit, -4, 4))
    final_prob = float(expit(cal_logit))

    return {
        "home_prob": final_prob,
        "raw_prob": float(pred["raw_prob"]),
        "blended_prob": blended_prob,
        "adjustment": float(pred["adjustment"]),
        "offset_logit": float(pred["offset_logit"]),
        "lambda_used": lam,
        "predicted_winner": "home" if final_prob > 0.5 else "away",
    }


def predict_ensemble_v16(model_dict, game_features, platt_params, lambda_params, 
                         xgb_model=None, xgb_weight=0.3):
    """
    v16: Ensemble prediction with calibrated XGBoost.
    - Spread games: Use market directly (λ_spread=0)
    - Baseline games: Ridge + calibrated XGBoost ensemble
    """
    ridge_pred = predict_with_source_lambda(
        model_dict, game_features, platt_params,
        lambda_spread=lambda_params["spread"],
        lambda_base=lambda_params["base"]
    )
    if ridge_pred is None:
        return None
    
    src = game_features["offset_source"]
    
    # For spread games, just use ridge (which uses offset directly)
    if src == "spread" or xgb_model is None:
        return ridge_pred
    
    # For baseline games, ensemble with calibrated XGBoost
    xgb_pred = predict_xgboost_calibrated(xgb_model, game_features)
    if xgb_pred is None:
        return ridge_pred
    
    ridge_prob = ridge_pred["home_prob"]
    xgb_prob = xgb_pred["calibrated_prob"]
    
    ensemble_prob = float((1 - xgb_weight) * ridge_prob + xgb_weight * xgb_prob)
    ensemble_prob = float(np.clip(ensemble_prob, 0.01, 0.99))
    
    return {
        "home_prob": ensemble_prob,
        "raw_prob": ridge_pred["raw_prob"],
        "blended_prob": ridge_pred["blended_prob"],
        "adjustment": ridge_pred["adjustment"],
        "offset_logit": ridge_pred["offset_logit"],
        "lambda_used": ridge_pred["lambda_used"],
        "predicted_winner": "home" if ensemble_prob > 0.5 else "away",
        "ridge_prob": ridge_prob,
        "xgb_raw_prob": xgb_pred["raw_prob"],
        "xgb_calibrated_prob": xgb_prob,
        "is_ensemble": True,
    }


# ============================================================
# UPSET TAGGING
# ============================================================

def upset_tag_by_seed(row, pred_home_prob,
                      alert_p=0.35,
                      watch_p=0.30,
                      longshot_seed_gap=4,
                      longshot_p=0.25,
                      min_seed_gap=2,
                      coinflip_seed_gap=1,
                      coinflip_p=0.47):
    home_seed = int(row["home_seed"])
    away_seed = int(row["away_seed"])
    seed_gap = abs(away_seed - home_seed)

    p_home = float(pred_home_prob)

    if home_seed > away_seed:
        underdog_prob = p_home
    elif away_seed > home_seed:
        underdog_prob = 1.0 - p_home
    else:
        underdog_prob = min(p_home, 1.0 - p_home)

    tag = ""

    if seed_gap == coinflip_seed_gap and underdog_prob >= coinflip_p:
        tag = "COINFLIP UPSET"

    if seed_gap >= longshot_seed_gap and underdog_prob >= longshot_p:
        tag = "UPSET LONGSHOT"

    if seed_gap >= min_seed_gap:
        if underdog_prob >= alert_p:
            tag = "UPSET ALERT"
        elif underdog_prob >= watch_p and tag == "":
            tag = "UPSET WATCH"

    return tag, float(underdog_prob), int(seed_gap)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(results_df):
    out = {}
    out["n_games"] = len(results_df)
    out["accuracy"] = float(results_df["correct"].mean()) if len(results_df) else 0.0

    probs = results_df.apply(
        lambda r: r["home_prob"] if int(r["actual_home_wins"]) == 1 else (1 - r["home_prob"]),
        axis=1,
    )
    out["log_loss"] = float(-np.mean(np.log(probs.clip(0.01, 0.99)))) if len(results_df) else np.nan

    off = results_df.apply(
        lambda r: expit(r["offset_logit"]) if int(r["actual_home_wins"]) == 1 else (1 - expit(r["offset_logit"])),
        axis=1,
    )
    out["offset_log_loss"] = float(-np.mean(np.log(off.clip(0.01, 0.99)))) if len(results_df) else np.nan

    base = results_df.apply(
        lambda r: r["baseline_prob"] if int(r["actual_home_wins"]) == 1 else (1 - r["baseline_prob"]),
        axis=1,
    )
    out["baseline_log_loss"] = float(-np.mean(np.log(base.clip(0.01, 0.99)))) if len(results_df) else np.nan

    return out


def is_seed_upset(row):
    home_seed = int(row["home_seed"])
    away_seed = int(row["away_seed"])
    winner = row["actual"]

    if winner == row["home_team"]:
        win_seed = home_seed
    else:
        win_seed = away_seed
        
    lose_seed = home_seed if winner != row["home_team"] else away_seed

    return win_seed > lose_seed


def predicted_seed_upset(row):
    home_seed = int(row["home_seed"])
    away_seed = int(row["away_seed"])
    pred_team = row["predicted"]

    if pred_team == row["home_team"]:
        pred_seed = home_seed
        other_seed = away_seed
    else:
        pred_seed = away_seed
        other_seed = home_seed

    return pred_seed > other_seed


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def analyze_feature_importance(model_dict, verbose=True):
    if model_dict is None:
        return None
    
    coefs = model_dict["model"].params[1:]
    feature_cols = model_dict["feature_cols"]
    
    importance = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coef": np.abs(coefs)
    }).sort_values("abs_coef", ascending=False)
    
    if verbose:
        print("\nFeature Importance (by |coefficient|):")
        print("-" * 50)
        for _, row in importance.head(15).iterrows():
            print(f"  {row['feature']:<30} {row['coefficient']:+.4f}")
    
    return importance


# ============================================================
# 2024 VALIDATION
# ============================================================

def validate_on_2024(model_dict, features_df, platt_params, lambda_params,
                     xgb_model=None, xgb_model_full=None, xgb_weight=0.3):
    print("\n" + "=" * 140)
    ensemble_str = f" + XGB ensemble (weight={xgb_weight:.1f})" if xgb_model else ""
    print(f"2024 PLAYOFF VALIDATION - MODEL v16 (λ_spread={lambda_params['spread']:.1f}, λ_base={lambda_params['base']:.1f}{ensemble_str})")
    print("=" * 140)

    test_df = features_df[features_df["season"] == 2024].copy()

    round_names = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference", "SB": "Super Bowl"}

    rows = []
    for _, gf in test_df.iterrows():
        # v16: Use calibrated ensemble
        pred = predict_ensemble_v16(
            model_dict, gf, platt_params, lambda_params, 
            xgb_model=xgb_model, xgb_weight=xgb_weight
        )
        if pred is None:
            continue

        pred_team = gf["home_team"] if pred["predicted_winner"] == "home" else gf["away_team"]
        actual = gf["winner"]
        correct = (pred_team == actual)

        home_prob = float(pred["home_prob"])
        off_prob = float(expit(float(gf["offset_logit"])))
        delta = home_prob - off_prob

        tag, ud_prob, seed_gap = upset_tag_by_seed(gf, home_prob)

        # XGBoost comparison (full model) if available
        xgb_full_prob = None
        if xgb_model_full is not None:
            xgb_pred = predict_xgboost_calibrated(xgb_model_full, gf)
            if xgb_pred is not None:
                xgb_full_prob = xgb_pred["calibrated_prob"]

        row_data = {
            "round": round_names.get(gf["game_type"], gf["game_type"]),
            "matchup": f"{gf['orig_away_team']} @ {gf['orig_home_team']}",
            "away_team": gf["away_team"],
            "home_team": gf["home_team"],
            "away_seed": int(gf["away_seed"]),
            "home_seed": int(gf["home_seed"]),
            "seed_diff": int(gf["seed_diff"]),
            "baseline_prob": float(gf["baseline_prob"]),
            "offset_source": gf["offset_source"],
            "offset_logit": float(gf["offset_logit"]),
            "home_prob": home_prob,
            "offset_prob": off_prob,
            "delta_vs_offset": float(delta),
            "lambda_used": float(pred["lambda_used"]),
            "predicted": pred_team,
            "actual": actual,
            "correct": bool(correct),
            "actual_home_wins": 1 if actual == gf["home_team"] else 0,
            "upset_tag": tag,
            "underdog_prob": float(ud_prob),
            "seed_gap": int(seed_gap),
            "score": f"{gf['away_score']}-{gf['home_score']}",
            "is_tossup": int(gf.get("is_tossup", 0)),
            "spread_magnitude": float(gf.get("spread_magnitude", 0)),
        }
        
        if xgb_full_prob is not None:
            row_data["xgb_prob"] = xgb_full_prob
        
        # Store ridge vs xgb for ensemble games
        if pred.get("is_ensemble"):
            row_data["ridge_prob"] = pred.get("ridge_prob", home_prob)
            row_data["xgb_cal_prob"] = pred.get("xgb_calibrated_prob", home_prob)
        
        rows.append(row_data)

    results_df = pd.DataFrame(rows)

    # Print by round
    for rname in ["Wild Card", "Divisional", "Conference", "Super Bowl"]:
        rg = results_df[results_df["round"] == rname]
        if len(rg) == 0:
            continue

        print(f"\n{rname.upper()}")
        header = f"{'Matchup':<16} {'Seeds':<7} {'Src':<5} {'λ':<4} {'Final':>6} {'Off':>6} {'Δ':>6} {'UD%':>5} {'Sprd':>5} {'T':>2} {'ALERT':<14} {'Pred':<5} {'Act':<5} ✓"
        if "xgb_prob" in results_df.columns:
            header += " {'XGB':>5}"
        print(header)
        print("-" * 140)

        for _, row in rg.iterrows():
            seeds = f"{int(row['away_seed'])}v{int(row['home_seed'])}"
            src = "sprd" if row["offset_source"] == "spread" else "base"
            mark = "✓" if row["correct"] else "✗"
            alert = row["upset_tag"] if row["upset_tag"] else ""
            sprd = row.get("spread_magnitude", 0)
            tossup = "T" if row.get("is_tossup", 0) else ""
            
            line = (
                f"{row['matchup']:<16} {seeds:<7} {src:<5} {row['lambda_used']:<4.1f} "
                f"{row['home_prob']:>5.0%} {row['offset_prob']:>5.0%} {row['delta_vs_offset']:>+5.0%} "
                f"{row['underdog_prob']:>5.0%} {sprd:>5.1f} {tossup:>2} {alert:<14} {row['predicted']:<5} {row['actual']:<5} {mark}"
            )
            
            if "xgb_prob" in row.index and pd.notna(row.get("xgb_prob")):
                line += f" {row['xgb_prob']:>5.0%}"
            
            print(line)

    # Metrics
    print("\n" + "=" * 140)
    print("METRICS - MODEL v16")
    print("=" * 140)

    m = evaluate_predictions(results_df)
    print(f"\nAccuracy: {results_df['correct'].sum()}/{len(results_df)} ({m['accuracy']:.1%})")

    print("\nLog loss (lower is better):")
    print(f"  Model v16:    {m['log_loss']:.4f}")
    print(f"  Offset-only:  {m['offset_log_loss']:.4f}")
    print(f"  Baseline:     {m['baseline_log_loss']:.4f}")
    print(f"  vs Baseline:  {m['baseline_log_loss'] - m['log_loss']:+.4f} {'✓' if (m['baseline_log_loss'] - m['log_loss']) > 0 else '✗'}")
    print(f"  vs Offset:    {m['offset_log_loss'] - m['log_loss']:+.4f} {'✓' if (m['offset_log_loss'] - m['log_loss']) > 0 else '✗'}")

    # XGBoost (calibrated) comparison
    if "xgb_prob" in results_df.columns:
        xgb_probs = results_df.apply(
            lambda r: r["xgb_prob"] if int(r["actual_home_wins"]) == 1 else (1 - r["xgb_prob"])
            if pd.notna(r.get("xgb_prob")) else np.nan,
            axis=1,
        ).dropna()
        if len(xgb_probs) > 0:
            xgb_ll = float(-np.mean(np.log(xgb_probs.clip(0.01, 0.99))))
            print(f"  XGBoost(cal): {xgb_ll:.4f}")

    # Upsets
    if len(results_df):
        results_df["is_upset"] = results_df.apply(is_seed_upset, axis=1)
        results_df["predicted_upset"] = results_df.apply(predicted_seed_upset, axis=1)
        actual_upsets = int(results_df["is_upset"].sum())
        predicted_upsets = int(results_df["predicted_upset"].sum())
        pred_upset_recall = float((results_df["is_upset"] & results_df["predicted_upset"]).sum() / actual_upsets) if actual_upsets else 0.0

        print("\nUpsets (worse seed wins):")
        print(f"  Actual:    {actual_upsets}")
        print(f"  Predicted: {predicted_upsets}")
        print(f"  Pred upset recall: {pred_upset_recall:.1%}")

        results_df["alert_fired"] = results_df["upset_tag"].astype(str).str.len() > 0
        results_df["sig_upset"] = results_df["is_upset"] & (results_df["seed_gap"] >= 2)

        sig_actual = int(results_df["sig_upset"].sum())
        sig_alert_hits = int((results_df["sig_upset"] & results_df["alert_fired"]).sum())
        sig_alert_recall = (sig_alert_hits / sig_actual) if sig_actual else 0.0

        print("\nSignificant Upsets (seed gap >= 2):")
        print(f"  Actual: {sig_actual}")
        print(f"  Alert recall: {sig_alert_recall:.1%}")

        alert_fired = int(results_df["alert_fired"].sum())
        alert_recall = float((results_df["is_upset"] & results_df["alert_fired"]).sum() / actual_upsets) if actual_upsets else 0.0
        alert_precision = float((results_df["is_upset"] & results_df["alert_fired"]).sum() / alert_fired) if alert_fired else 0.0

        print("\nUpset Alerts:")
        print(f"  Alerts fired: {alert_fired}")
        print(f"  Alert recall: {alert_recall:.1%}")
        print(f"  Alert precision: {alert_precision:.1%}")

    # NEW v16: Toss-up analysis
    tossup_games = results_df[results_df["is_tossup"] == 1]
    if len(tossup_games) > 0:
        print(f"\nToss-up Games (seed_diff ≤ 1, close spread):")
        print(f"  Total: {len(tossup_games)}")
        print(f"  Correct: {tossup_games['correct'].sum()}/{len(tossup_games)} ({tossup_games['correct'].mean():.1%})")

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("NFL PLAYOFF MODEL v16 - CALIBRATED XGBOOST ENSEMBLE")
    print("=" * 100)
    
    games_df, team_df, spread_df = load_all_data()
    games_df = merge_spread_data_safe(games_df, spread_df)
    spread_sanity_check(games_df)

    # Splits
    train_seasons = list(range(2000, 2023))
    calib_seasons = [2021, 2022, 2023]
    tune_seasons = list(range(2015, 2024))
    spread_val_seasons = [2020, 2021, 2022, 2023]
    xgb_calib_seasons = [2021, 2022, 2023]  # For XGBoost Platt scaling

    # Baselines
    baselines = compute_historical_baselines(games_df, max_season=2023)
    print(f"\nHistorical Baselines: Home win rate = {baselines['home_win_rate']:.1%}")

    epa_model = fit_expected_win_pct_model(team_df, max_season=2023)
    print(f"EPA -> Win%: {epa_model['intercept']:.3f} + {epa_model['slope']:.3f} * net_epa")

    # Build temp features for spread tuning
    temp_features = prepare_game_features_v16(games_df, team_df, epa_model, baselines, spread_model=None)

    # Spread model
    spread_model = tune_spread_alpha_v14(
        temp_features,
        train_seasons=train_seasons,
        val_seasons=spread_val_seasons,
        baselines=baselines,
        alphas=(0.1, 0.25, 0.5, 1.0, 2.0)
    )
    print_spread_model(spread_model)

    # Prepare v16 features
    print("\nPreparing v16 features (with toss-up detection)...")
    features_df = prepare_game_features_v16(games_df, team_df, epa_model, baselines, spread_model)
    print(f"Total games: {len(features_df)}")

    # Feature columns (v15.1 simplified + v16 tossup)
    feature_cols = [
        # Base
        "delta_net_epa", "delta_pd_pg", "seed_diff",
        "delta_vulnerability", "delta_underseeded", "delta_momentum",
        "delta_pass_edge", "delta_rush_edge",
        "delta_ol_exposure", "delta_pass_d_exposure",
        "home_ol_exposure", "away_ol_exposure",
        "total_ol_exposure", "total_pass_d_exposure",
        # Interactions (v15.1)
        "pass_x_ol", "seed_x_epa",
        # Non-linear (v15.1)
        "seed_diff_sq", "seed_diff_abs",
        # Round (v15.1)
        "is_superbowl",
        # Spread (v15.1)
        "spread_magnitude", "spread_confidence", "is_close_game",
        # NEW v16: Toss-up
        "is_tossup", "is_big_favorite", "is_medium_gap", "is_close_seeds",
    ]
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    # Tune ridge alpha
    print("\nTuning ridge alpha on 2023...")
    best_alpha, best_ll = None, float("inf")

    for a in [0.5, 1.0, 2.0, 5.0]:
        md = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=a)
        if md is None:
            continue

        test_2023 = features_df[features_df["season"] == 2023].dropna(subset=feature_cols + ["offset_logit"])
        preds = []
        for _, r in test_2023.iterrows():
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

    # Train final model
    model_dict = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=best_alpha)
    print(f"\nBest α={best_alpha}, trained on {model_dict['train_samples']} games")

    # Feature importance
    analyze_feature_importance(model_dict)

    # NEW v16: Train CALIBRATED XGBoost for baseline games
    xgb_model = None
    xgb_model_full = None
    if HAS_XGBOOST:
        print("\nTraining calibrated XGBoost for baseline games...")
        baseline_df = features_df[features_df["offset_source"] == "baseline"]
        xgb_model = train_xgboost_calibrated(
            baseline_df, feature_cols, train_seasons, xgb_calib_seasons
        )
        if xgb_model:
            print(f"  XGBoost trained on {xgb_model['train_samples']} baseline games")
            print(f"  Calibrated: {xgb_model['calibrated']}")
        
        # Full XGBoost for comparison
        print("\nTraining calibrated XGBoost (full) for comparison...")
        xgb_model_full = train_xgboost_calibrated(
            features_df, feature_cols, train_seasons, xgb_calib_seasons
        )
        if xgb_model_full:
            print(f"  XGBoost (full) trained on {xgb_model_full['train_samples']} games")

    # Calibration
    print("\nCalibrating ridge (baseline only)...")
    platt_params = calibrate_baseline_only(model_dict, features_df, calib_seasons, shrinkage=0.3)

    # Tune lambdas
    print("\nTuning λ by source (v16 - trust market)...")
    lambda_params = tune_source_lambdas(
        model_dict, features_df, tune_seasons, platt_params, 
        improve_gate=0.025, force_spread_zero=True
    )

    # NEW v16: Tune XGBoost ensemble weight
    xgb_weight = 0.3  # Default
    if xgb_model is not None:
        xgb_weight = tune_ensemble_weight(
            model_dict, xgb_model, features_df, 
            val_seasons=[2022, 2023], 
            lambda_base=lambda_params["base"]
        )

    # Validate 2024
    results_2024 = validate_on_2024(
        model_dict, features_df, platt_params, lambda_params,
        xgb_model=xgb_model,
        xgb_model_full=xgb_model_full,
        xgb_weight=xgb_weight
    )

    # Historical rolling validation
    print("\n" + "="*140)
    print("HISTORICAL ROLLING VALIDATION (2015-2024) - v16")
    print("="*140)
    
    all_results = []
    
    for test_year in range(2015, 2025):
        train_yrs = list(range(2000, test_year - 2))
        calib_yrs = [test_year - 2, test_year - 1]
        tune_yrs = list(range(2010, test_year))
        spread_val_yrs = list(range(test_year - 4, test_year))
        xgb_calib_yrs = calib_yrs
        
        w_baselines = compute_historical_baselines(games_df, max_season=test_year - 1)
        w_epa = fit_expected_win_pct_model(team_df, max_season=test_year - 1)
        
        temp = prepare_game_features_v16(games_df, team_df, w_epa, w_baselines, spread_model=None)
        w_spread = tune_spread_alpha_v14(
            temp,
            train_seasons=train_yrs,
            val_seasons=spread_val_yrs,
            baselines=w_baselines,
            verbose=False
        )
        
        w_features = prepare_game_features_v16(games_df, team_df, w_epa, w_baselines, w_spread)
        
        avail_cols = [c for c in feature_cols if c in w_features.columns]
        
        model = train_ridge_offset_model(w_features, avail_cols, train_yrs, alpha=best_alpha)
        if model is None:
            continue
        
        # Train calibrated XGBoost for this window
        w_baseline_df = w_features[w_features["offset_source"] == "baseline"]
        w_xgb = train_xgboost_calibrated(w_baseline_df, avail_cols, train_yrs, xgb_calib_yrs) if HAS_XGBOOST else None
        
        platt = calibrate_baseline_only(model, w_features, calib_yrs)
        lam_params = tune_source_lambdas(
            model, w_features, tune_yrs, platt, 
            improve_gate=0.025, force_spread_zero=True, verbose=False
        )
        
        # Tune ensemble weight
        w_xgb_weight = 0.3
        if w_xgb is not None:
            # Simple: use fixed weight for historical (tuning per-window is expensive)
            w_xgb_weight = xgb_weight
        
        test_df = w_features[w_features['season'] == test_year].dropna(subset=avail_cols + ['offset_logit'])
        
        results = []
        for _, gf in test_df.iterrows():
            pred = predict_ensemble_v16(
                model, gf, platt, lam_params, 
                xgb_model=w_xgb, xgb_weight=w_xgb_weight
            )
            if pred is None:
                continue
            
            actual = 1 if gf['winner'] == gf['home_team'] else 0
            results.append({
                'correct': (pred['predicted_winner'] == 'home') == (actual == 1),
                'home_prob': pred['home_prob'],
                'offset_logit': gf['offset_logit'],
                'baseline_prob': gf['baseline_prob'],
                'actual': actual
            })
        
        if results:
            rdf = pd.DataFrame(results)
            model_ll = float(-np.mean([np.log(np.clip(r['home_prob'] if r['actual'] else 1-r['home_prob'], 0.01, 0.99)) for _, r in rdf.iterrows()]))
            offset_ll = float(-np.mean([np.log(np.clip(expit(r['offset_logit']) if r['actual'] else 1-expit(r['offset_logit']), 0.01, 0.99)) for _, r in rdf.iterrows()]))
            base_ll = float(-np.mean([np.log(np.clip(r['baseline_prob'] if r['actual'] else 1-r['baseline_prob'], 0.01, 0.99)) for _, r in rdf.iterrows()]))
            
            all_results.append({
                'season': test_year, 'games': len(rdf), 'correct': int(rdf['correct'].sum()),
                'acc': float(rdf['correct'].mean()), 'model_ll': model_ll, 'offset_ll': offset_ll, 'base_ll': base_ll,
                'lam_spread': float(lam_params['spread']), 'lam_base': float(lam_params['base'])
            })
    
    hist_df = pd.DataFrame(all_results)
    
    print(f"\n{'Season':<8} {'N':<5} {'Acc':<7} {'Model':<8} {'Offset':<8} {'Base':<8} {'vs Base':<9} {'vs Off':<9} {'λs':<4} {'λb':<4}")
    print("-" * 100)
    for _, r in hist_df.iterrows():
        vs_base = float(r['base_ll'] - r['model_ll'])
        vs_off = float(r['offset_ll'] - r['model_ll'])
        print(f"{int(r['season']):<8} {int(r['games']):<5} {r['acc']:.1%}   {r['model_ll']:.4f}   {r['offset_ll']:.4f}   "
              f"{r['base_ll']:.4f}   {vs_base:+.4f} {'✓' if vs_base > 0 else ''}   {vs_off:+.4f} {'✓' if vs_off > 0 else ''}   "
              f"{r['lam_spread']:.1f}  {r['lam_base']:.1f}")
    
    print("-" * 100)
    avg_model = float(hist_df['model_ll'].mean())
    avg_offset = float(hist_df['offset_ll'].mean())
    avg_base = float(hist_df['base_ll'].mean())
    total_games = int(hist_df['games'].sum())
    total_correct = int(hist_df['correct'].sum())
    overall_acc = float(total_correct / total_games)
    
    print(f"{'AVG':<8} {total_games:<5} {overall_acc:.1%}   "
          f"{avg_model:.4f}   {avg_offset:.4f}   {avg_base:.4f}   "
          f"{avg_base-avg_model:+.4f}      {avg_offset-avg_model:+.4f}")
    
    print(f"\nModel beats baseline: {int((hist_df['base_ll'] > hist_df['model_ll']).sum())}/{len(hist_df)}")
    print(f"Model beats offset:   {int((hist_df['offset_ll'] > hist_df['model_ll']).sum())}/{len(hist_df)}")

    # Save
    out_path = "../validation_results_2024_v16.csv"
    results_2024.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n" + "=" * 100)
    print("v16 SUMMARY")
    print("=" * 100)
    print(f"""
    NEW IN v16:
    ✓ Calibrated XGBoost (Platt scaling fixes overconfidence)
    ✓ Dynamic ensemble weight tuning (best weight: {xgb_weight:.1f})
    ✓ Toss-up detector feature (seed_diff ≤ 1 + close spread)
    ✓ Seed tier indicators (big_favorite, medium_gap, close_seeds)
    
    PRESERVED FROM v15.1:
    ✓ λ_spread=0.0 (trust market for spread games)
    ✓ Simplified features (22 base + 4 new = 26 total)
    ✓ XGBoost ensemble for baseline-only games
    ✓ Conservative ridge calibration
    
    MODEL STRATEGY:
    ✓ Spread games: Use market probability directly
    ✓ Baseline games: Ridge + calibrated XGBoost ({1-xgb_weight:.0%}/{xgb_weight:.0%} blend)
    """)

    return results_2024


if __name__ == "__main__":
    main()
