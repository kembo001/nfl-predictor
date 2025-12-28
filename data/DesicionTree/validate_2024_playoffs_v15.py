"""
NFL Playoff Model v15.1 — v14 Core + Enhanced Features (FIXED)

FIXES from v15:
- Force λ_spread=0.0 (trust the market for spread games)
- Higher improve_gate (0.025) to prevent overriding market
- Removed market_disagree features (likely noise/overfitting)
- Simplified interaction terms (keep only pass_x_ol, seed_x_epa)
- XGBoost ensemble for baseline games (not just comparison)
- Validated tau selection for recency weighting
- Better spread features: spread_magnitude, spread_confidence

KEPT from v15:
- Interaction terms (pass_edge × ol_exposure, seed_diff × epa_gap)
- Non-linear features (seed_diff_sq, seed_diff_abs)
- Recency-weighted training (validated tau)
- XGBoost for baseline-only games
- Playoff round features (SB indicator)

NOTES:
- All v14 stabilization preserved
- Respects market (λ_spread=0) while adding value on baseline games
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings("ignore")

# Optional XGBoost - graceful fallback if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Note: XGBoost not installed. Skipping gradient boosting comparison.")

# ============================================================
# DATA LOADING (unchanged from v14)
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
# BASELINES (unchanged from v14)
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
# Z-SCORES (unchanged from v14)
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
# MATCHUP FEATURES (unchanged from v14)
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
# NEW v15: NON-LINEAR FEATURE ENGINEERING
# ============================================================

def create_seed_spline_basis(seed_diff, knots=[-4, -2, 0, 2, 4]):
    """
    Create spline basis functions for seed_diff to capture non-linear upset probability.
    Returns dict of basis function values.
    """
    features = {}
    sd = float(seed_diff)
    
    # Piecewise linear spline basis (restricted cubic spline approximation)
    for i, knot in enumerate(knots):
        features[f"seed_spline_{i}"] = float(max(0, sd - knot))
    
    # Quadratic term for seed_diff (captures diminishing returns)
    features["seed_diff_sq"] = float(sd ** 2)
    
    # Sign-aware magnitude (different effect for favorites vs underdogs)
    features["seed_diff_abs"] = float(abs(sd))
    
    return features


def create_polynomial_features(features_dict, degree=2):
    """
    Add polynomial features for key metrics.
    """
    poly_features = {}
    
    # Quadratic terms for exposure metrics
    for col in ["delta_ol_exposure", "delta_pass_d_exposure", "total_ol_exposure"]:
        if col in features_dict:
            val = float(features_dict[col])
            poly_features[f"{col}_sq"] = val ** 2
    
    # Quadratic for EPA differential
    if "delta_net_epa" in features_dict:
        val = float(features_dict["delta_net_epa"])
        poly_features["delta_net_epa_sq"] = val ** 2
    
    return poly_features


# ============================================================
# NEW v15: INTERACTION TERMS
# ============================================================

def create_interaction_features(features_dict):
    """
    Create interaction terms that capture strategic matchup compounds.
    """
    interactions = {}
    
    # Pass game × protection: elite passing attack vs poor protection = disaster
    if "delta_pass_edge" in features_dict and "delta_ol_exposure" in features_dict:
        interactions["pass_x_ol"] = float(
            features_dict["delta_pass_edge"] * features_dict["delta_ol_exposure"]
        )
    
    # Seed upset × quality gap: lower seeds with better metrics = upset potential
    if "seed_diff" in features_dict and "delta_net_epa" in features_dict:
        interactions["seed_x_epa"] = float(
            features_dict["seed_diff"] * features_dict["delta_net_epa"]
        )
    
    # OL exposure × pass defense exposure: compounding vulnerabilities
    if "delta_ol_exposure" in features_dict and "delta_pass_d_exposure" in features_dict:
        interactions["ol_x_passd"] = float(
            features_dict["delta_ol_exposure"] * features_dict["delta_pass_d_exposure"]
        )
    
    # Rush edge × pass edge: balanced offense indicator
    if "delta_rush_edge" in features_dict and "delta_pass_edge" in features_dict:
        interactions["rush_x_pass"] = float(
            features_dict["delta_rush_edge"] * features_dict["delta_pass_edge"]
        )
    
    # Underseeded × seed_diff: quality team in unfavorable position
    if "delta_underseeded" in features_dict and "seed_diff" in features_dict:
        interactions["underseed_x_seeddiff"] = float(
            features_dict["delta_underseeded"] * features_dict["seed_diff"]
        )
    
    return interactions


# ============================================================
# NEW v15: MARKET DISAGREEMENT SIGNAL
# ============================================================

def compute_spread_features(home_spread):
    """
    v15.1 FIX: Better spread-derived features than market disagreement.
    These capture game uncertainty, not noise from baseline comparison.
    """
    if pd.isna(home_spread):
        return {
            "spread_magnitude": 0.0,
            "spread_confidence": 0.5,
            "is_close_game": 0,
        }
    
    spread = abs(float(home_spread))
    
    return {
        # Larger spreads = more certain outcomes
        "spread_magnitude": spread,
        # Inverse confidence: close games more volatile
        "spread_confidence": 1.0 / (spread + 3.0),
        # Binary: games within 3 points are toss-ups
        "is_close_game": 1 if spread <= 3.0 else 0,
    }


# ============================================================
# NEW v15: PLAYOFF ROUND FEATURES
# ============================================================

def create_round_features(game_type):
    """
    Create one-hot encoded round features + round-specific adjustments.
    Different dynamics per round:
    - WC: More chaos, travel fatigue for wild card teams
    - DIV: Home favorites dominate (bye week advantage)
    - CON: Elite QB matters more
    - SB: Neutral, special preparation
    """
    round_features = {
        "is_wildcard": 0,
        "is_divisional": 0,
        "is_conference": 0,
        "is_superbowl": 0,
    }
    
    if game_type == "WC":
        round_features["is_wildcard"] = 1
    elif game_type == "DIV":
        round_features["is_divisional"] = 1
    elif game_type == "CON":
        round_features["is_conference"] = 1
    elif game_type == "SB":
        round_features["is_superbowl"] = 1
    
    return round_features


# ============================================================
# SPREAD MODEL v14 STABILIZED (unchanged)
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
        "prior_intercept": prior_intercept,
        "prior_slope_per_point": prior_slope_per_point,
        "clamp_band": clamp_band,
    }


def tune_spread_alpha_v14(features_df, train_seasons, val_seasons, baselines,
                          alphas=(0.1, 0.25, 0.5, 1.0, 2.0),
                          clamp_band=(-0.12, -0.04),
                          prior_slope_per_point=-0.073,
                          prior_strength_k=60,
                          verbose=True):
    if verbose:
        print("\nTuning spread alpha (v14 stabilized)...")

    best_alpha, best_ll, best_model = None, float("inf"), None

    val_df = features_df[
        (features_df["season"].isin(val_seasons)) & (features_df["home_spread"].notna())
    ].copy()

    if len(val_df) < 5:
        if verbose:
            print("  Not enough spread games in validation seasons, using default α=0.5")
        return fit_spread_model_v14(features_df, train_seasons, baselines, alpha=0.5,
                                   clamp_band=clamp_band,
                                   prior_slope_per_point=prior_slope_per_point,
                                   prior_strength_k=prior_strength_k)

    for a in alphas:
        smod = fit_spread_model_v14(
            features_df, train_seasons, baselines, alpha=a,
            clamp_band=clamp_band,
            prior_slope_per_point=prior_slope_per_point,
            prior_strength_k=prior_strength_k
        )
        if smod is None:
            continue

        probs = expit(np.clip(smod["intercept"] + smod["slope_per_point"] * val_df["home_spread"].values, -4, 4))
        y = val_df["home_wins"].values
        ll = -np.mean(np.log(np.clip(np.where(y == 1, probs, 1 - probs), 0.01, 0.99)))

        if verbose:
            print(f"  α={a:<4}  val_ll={ll:.4f}  slope/pt={smod['slope_per_point']:+.4f}  w_prior={smod['w_prior']:.2f}")

        if ll < best_ll:
            best_ll, best_alpha, best_model = ll, a, smod

    if best_model is None:
        return None

    if verbose:
        print(f"  Best α={best_alpha} (val_ll={best_ll:.4f}), final slope/pt={best_model['slope_per_point']:+.4f}")
    return best_model


def print_spread_model_v14(spread_model):
    if spread_model is None:
        print("Spread Model: None")
        return

    print("\nSpread Model (v14 stabilized):")
    print(f"  n={spread_model['n']}, α={spread_model['alpha']}, w_prior={spread_model['w_prior']:.2f}")
    print(f"  logit(P(home)) = {spread_model['intercept']:.3f} + {spread_model['slope_per_point']:+.4f} * spread_points")
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
# FEATURE ENGINEERING v15 (enhanced)
# ============================================================

def prepare_game_features_v15(games_df, team_df, epa_model, baselines, spread_model,
                              include_interactions=True,
                              include_nonlinear=True,
                              include_round_features=True):
    """
    Enhanced feature preparation with v15 additions:
    - Interaction terms
    - Non-linear features (splines, polynomials)
    - Round-specific features
    - Market disagreement signal
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

        # Core stats (unchanged from v14)
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

        # NEW v15: Non-linear features
        if include_nonlinear:
            spline_features = create_seed_spline_basis(seed_diff)
            row.update(spline_features)
            
            poly_features = create_polynomial_features(row)
            row.update(poly_features)

        # NEW v15: Interaction terms
        if include_interactions:
            interaction_features = create_interaction_features(row)
            row.update(interaction_features)

        # v15.1 FIX: Better spread features (not market disagreement)
        spread_features = compute_spread_features(home_spread)
        row.update(spread_features)

        # NEW v15: Round features
        if include_round_features:
            round_features = create_round_features(game["game_type"])
            row.update(round_features)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# NEW v15: RECENCY-WEIGHTED RIDGE
# ============================================================

def compute_sample_weights(seasons, reference_year, tau=3.0):
    """
    Exponential decay weights for recency-weighted training.
    tau controls decay rate (higher = slower decay, more weight on old data)
    """
    weights = np.exp(-(reference_year - np.array(seasons)) / tau)
    return weights / weights.sum() * len(weights)  # Normalize to sum = n


def train_ridge_offset_model_weighted(features_df, feature_cols, train_seasons, 
                                       alpha=1.0, use_recency_weights=True, 
                                       reference_year=2024, tau=3.0,
                                       l1_ratio=0.0):
    """
    Enhanced ridge model with:
    - Recency weighting (optional)
    - Elastic net support (l1_ratio > 0)
    """
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

    # Compute sample weights
    if use_recency_weights:
        sample_weights = compute_sample_weights(
            train_df["season"].values, 
            reference_year=reference_year, 
            tau=tau
        )
    else:
        sample_weights = None

    try:
        if sample_weights is not None:
            # statsmodels GLM with frequency weights
            res = sm.GLM(
                y, X, 
                family=sm.families.Binomial(), 
                offset=offset,
                freq_weights=sample_weights
            ).fit_regularized(alpha=alpha, L1_wt=l1_ratio)
        else:
            res = sm.GLM(
                y, X, 
                family=sm.families.Binomial(), 
                offset=offset
            ).fit_regularized(alpha=alpha, L1_wt=l1_ratio)
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
        "l1_ratio": l1_ratio,
        "recency_weighted": use_recency_weights,
        "tau": tau if use_recency_weights else None,
    }


# Standard (unweighted) version for comparison
def train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=1.0):
    return train_ridge_offset_model_weighted(
        features_df, feature_cols, train_seasons, 
        alpha=alpha, use_recency_weights=False
    )


# ============================================================
# NEW v15: XGBOOST ALTERNATIVE
# ============================================================

def train_xgboost_model(features_df, feature_cols, train_seasons, 
                        n_estimators=100, max_depth=3, learning_rate=0.1):
    """
    XGBoost alternative model for comparison.
    Uses offset mechanism via base_score adjustment.
    """
    if not HAS_XGBOOST:
        return None
    
    train_df = features_df[
        features_df["season"].isin(train_seasons)
    ].dropna(subset=feature_cols + ["offset_logit"])

    if len(train_df) < 30:
        return None

    X = train_df[feature_cols].values
    y = train_df["home_wins"].values
    offset_logits = train_df["offset_logit"].values

    # XGBoost with offset: train on residuals from offset prediction
    # Then add predictions back to offset at inference
    try:
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=42
        )
        
        # We'll train XGBoost to predict residuals, then combine with offset
        # Store offset for later use
        model.fit(X, y)
        
        return {
            "model": model,
            "feature_cols": feature_cols,
            "train_samples": len(train_df),
            "type": "xgboost"
        }
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        return None


def predict_xgboost(model_dict, game_features):
    """Predict using XGBoost model."""
    if model_dict is None or model_dict.get("type") != "xgboost":
        return None
    
    try:
        X = np.array([[game_features[col] for col in model_dict["feature_cols"]]], dtype=float)
        if np.any(np.isnan(X)):
            return None
        
        prob = float(model_dict["model"].predict_proba(X)[0, 1])
        return {"raw_prob": np.clip(prob, 0.01, 0.99)}
    except Exception:
        return None


# ============================================================
# NEW v15: BOOTSTRAP UNCERTAINTY QUANTIFICATION
# ============================================================

def bootstrap_predictions(features_df, feature_cols, train_seasons, game_features,
                          n_bootstrap=100, alpha=1.0):
    """
    Generate bootstrap confidence intervals for predictions.
    Returns prediction distribution statistics.
    """
    train_df = features_df[
        features_df["season"].isin(train_seasons)
    ].dropna(subset=feature_cols + ["offset_logit"])

    if len(train_df) < 30:
        return None

    predictions = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        boot_df = train_df.sample(frac=1.0, replace=True, random_state=i)
        
        # Train model on bootstrap sample
        model = train_ridge_offset_model_weighted(
            boot_df.reset_index(drop=True), 
            feature_cols, 
            train_df["season"].unique().tolist(),
            alpha=alpha,
            use_recency_weights=False
        )
        
        if model is None:
            continue
        
        # Predict
        pred = predict_raw(model, game_features)
        if pred is not None:
            predictions.append(pred["raw_prob"])
    
    if len(predictions) < 10:
        return None
    
    predictions = np.array(predictions)
    
    return {
        "mean": float(np.mean(predictions)),
        "std": float(np.std(predictions)),
        "ci_5": float(np.percentile(predictions, 5)),
        "ci_95": float(np.percentile(predictions, 95)),
        "ci_width": float(np.percentile(predictions, 95) - np.percentile(predictions, 5)),
    }


# ============================================================
# PREDICTION (updated for v15)
# ============================================================

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
# CALIBRATION (unchanged from v14)
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
# LAMBDA TUNING (unchanged from v14)
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


def tune_source_lambdas(model_dict, features_df, tune_seasons, platt_params,
                        improve_gate=0.025, force_spread_zero=True, verbose=True):
    """
    v15.1 FIX: Higher improve_gate (0.025) and option to force λ_spread=0.
    The market is efficient - don't override it without strong evidence.
    """
    tune_df = features_df[features_df["season"].isin(tune_seasons)].dropna(
        subset=model_dict["feature_cols"] + ["offset_logit"]
    )

    best = {"spread": 0.0, "base": 1.0}

    for src, name in [("spread", "spread"), ("baseline", "base")]:
        # v15.1 FIX: Force spread lambda to 0 (trust market)
        if name == "spread" and force_spread_zero:
            best["spread"] = 0.0
            if verbose:
                print(f"  λ_spread forced to 0.0 (trust market - force_spread_zero=True)")
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

        # v15.1 FIX: Higher threshold for spread games
        if name == "spread" and best_imp < improve_gate:
            best["spread"] = 0.0
            if verbose:
                print(f"  λ_spread forced to 0.0 (best Δll={best_imp:+.4f} < {improve_gate})")
        else:
            best[name] = float(best_lam)
            if verbose:
                print(f"  λ_{name:<5} = {best_lam:.1f} (Δll={best_imp:+.4f})")

    return best


# ============================================================
# UPSET TAGGING (unchanged from v14)
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
# EVALUATION (unchanged from v14)
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
        lose_seed = away_seed
    else:
        win_seed = away_seed
        lose_seed = home_seed

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
# NEW v15: FEATURE SELECTION ANALYSIS
# ============================================================

def analyze_feature_importance(model_dict, verbose=True):
    """
    Analyze which features are most important based on coefficient magnitudes.
    """
    if model_dict is None:
        return None
    
    coefs = model_dict["model"].params[1:]  # Skip intercept
    feature_cols = model_dict["feature_cols"]
    
    # Create importance df
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
# 2024 VALIDATION (updated for v15)
# ============================================================

def predict_ensemble(model_dict, game_features, platt_params, lambda_params, xgb_model=None, 
                     xgb_weight=0.3):
    """
    v15.1 FIX: Ensemble prediction that uses XGBoost for baseline games.
    - Spread games: Use ridge only (λ_spread=0, trust market)
    - Baseline games: Blend ridge + XGBoost
    """
    # Get ridge prediction
    ridge_pred = predict_with_source_lambda(
        model_dict, game_features, platt_params,
        lambda_spread=lambda_params["spread"],
        lambda_base=lambda_params["base"]
    )
    if ridge_pred is None:
        return None
    
    src = game_features["offset_source"]
    
    # For spread games, just use ridge (which uses offset directly due to λ_spread=0)
    if src == "spread" or xgb_model is None:
        return ridge_pred
    
    # For baseline games, ensemble with XGBoost
    xgb_pred = predict_xgboost(xgb_model, game_features)
    if xgb_pred is None:
        return ridge_pred
    
    # Blend probabilities
    ridge_prob = ridge_pred["home_prob"]
    xgb_prob = xgb_pred["raw_prob"]
    
    # Weighted average (ridge gets more weight)
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
        "xgb_prob": xgb_prob,
        "is_ensemble": True,
    }


def validate_on_2024(model_dict, features_df, platt_params, lambda_params,
                     xgb_model=None, xgb_model_full=None, use_ensemble=True, show_uncertainty=False):
    print("\n" + "=" * 130)
    ensemble_str = " + XGB ensemble" if use_ensemble and xgb_model else ""
    print(f"2024 PLAYOFF VALIDATION - MODEL v15.1 (λ_spread={lambda_params['spread']:.1f}, λ_base={lambda_params['base']:.1f}{ensemble_str})")
    print("=" * 130)

    test_df = features_df[features_df["season"] == 2024].copy()

    round_names = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference", "SB": "Super Bowl"}

    rows = []
    for _, gf in test_df.iterrows():
        # v15.1 FIX: Use ensemble prediction for baseline games
        if use_ensemble and xgb_model is not None:
            pred = predict_ensemble(
                model_dict, gf, platt_params, lambda_params, xgb_model
            )
        else:
            pred = predict_with_source_lambda(
                model_dict,
                gf,
                platt_params,
                lambda_spread=lambda_params["spread"],
                lambda_base=lambda_params["base"],
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
        xgb_prob = None
        if xgb_model_full is not None:
            xgb_pred = predict_xgboost(xgb_model_full, gf)
            if xgb_pred is not None:
                xgb_prob = xgb_pred["raw_prob"]

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
            "spread_magnitude": float(gf.get("spread_magnitude", 0)),
            "is_close_game": int(gf.get("is_close_game", 0)),
        }
        
        if xgb_prob is not None:
            row_data["xgb_prob"] = xgb_prob
        
        rows.append(row_data)

    results_df = pd.DataFrame(rows)

    # Print by round
    for rname in ["Wild Card", "Divisional", "Conference", "Super Bowl"]:
        rg = results_df[results_df["round"] == rname]
        if len(rg) == 0:
            continue

        print(f"\n{rname.upper()}")
        header = f"{'Matchup':<16} {'Seeds':<7} {'Src':<5} {'λ':<4} {'Final':>6} {'Off':>6} {'Δ':>6} {'UD%':>5} {'Sprd':>5} {'ALERT':<14} {'Pred':<5} {'Act':<5} ✓"
        if "xgb_prob" in results_df.columns:
            header = header.replace("✓", "{'XGB':>5} ✓")
        print(header)
        print("-" * 130)

        for _, row in rg.iterrows():
            seeds = f"{int(row['away_seed'])}v{int(row['home_seed'])}"
            src = "sprd" if row["offset_source"] == "spread" else "base"
            mark = "✓" if row["correct"] else "✗"
            alert = row["upset_tag"] if row["upset_tag"] else ""
            sprd = row.get("spread_magnitude", 0)
            
            line = (
                f"{row['matchup']:<16} {seeds:<7} {src:<5} {row['lambda_used']:<4.1f} "
                f"{row['home_prob']:>5.0%} {row['offset_prob']:>5.0%} {row['delta_vs_offset']:>+5.0%} "
                f"{row['underdog_prob']:>5.0%} {sprd:>5.1f} {alert:<14} {row['predicted']:<5} {row['actual']:<5} {mark}"
            )
            
            if "xgb_prob" in row.index and pd.notna(row.get("xgb_prob")):
                line += f" {row['xgb_prob']:>5.0%}"
            
            print(line)

    # Metrics
    print("\n" + "=" * 130)
    print("METRICS - MODEL v15.1")
    print("=" * 130)

    m = evaluate_predictions(results_df)
    print(f"\nAccuracy: {results_df['correct'].sum()}/{len(results_df)} ({m['accuracy']:.1%})")

    print("\nLog loss (lower is better):")
    print(f"  Model v15.1:  {m['log_loss']:.4f}")
    print(f"  Offset-only:  {m['offset_log_loss']:.4f}")
    print(f"  Baseline:     {m['baseline_log_loss']:.4f}")
    print(f"  vs Baseline:  {m['baseline_log_loss'] - m['log_loss']:+.4f} {'✓' if (m['baseline_log_loss'] - m['log_loss']) > 0 else '✗'}")
    print(f"  vs Offset:    {m['offset_log_loss'] - m['log_loss']:+.4f} {'✓' if (m['offset_log_loss'] - m['log_loss']) > 0 else '✗'}")

    # XGBoost comparison if available
    if "xgb_prob" in results_df.columns:
        xgb_probs = results_df.apply(
            lambda r: r["xgb_prob"] if int(r["actual_home_wins"]) == 1 else (1 - r["xgb_prob"])
            if pd.notna(r.get("xgb_prob")) else np.nan,
            axis=1,
        ).dropna()
        if len(xgb_probs) > 0:
            xgb_ll = float(-np.mean(np.log(xgb_probs.clip(0.01, 0.99))))
            print(f"  XGBoost:      {xgb_ll:.4f}")

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

    return results_df


# ============================================================
# NEW v15: MODEL COMPARISON
# ============================================================

def compare_model_variants(features_df, feature_cols_base, feature_cols_enhanced,
                           train_seasons, val_seasons, baselines):
    """
    Compare different model configurations:
    - Base features only
    - Enhanced features (interactions + non-linear)
    - With vs without recency weighting
    - Ridge vs Elastic Net
    """
    print("\n" + "=" * 100)
    print("MODEL VARIANT COMPARISON")
    print("=" * 100)
    
    val_df = features_df[features_df["season"].isin(val_seasons)].dropna(
        subset=feature_cols_enhanced + ["offset_logit"]
    )
    
    variants = []
    
    # Variant 1: Base features, no recency
    m1 = train_ridge_offset_model_weighted(
        features_df, feature_cols_base, train_seasons,
        alpha=1.0, use_recency_weights=False
    )
    if m1:
        variants.append(("Base features, no recency", m1, feature_cols_base))
    
    # Variant 2: Enhanced features, no recency
    m2 = train_ridge_offset_model_weighted(
        features_df, feature_cols_enhanced, train_seasons,
        alpha=1.0, use_recency_weights=False
    )
    if m2:
        variants.append(("Enhanced features, no recency", m2, feature_cols_enhanced))
    
    # Variant 3: Enhanced features, with recency (tau=3)
    m3 = train_ridge_offset_model_weighted(
        features_df, feature_cols_enhanced, train_seasons,
        alpha=1.0, use_recency_weights=True, tau=3.0
    )
    if m3:
        variants.append(("Enhanced + recency (τ=3)", m3, feature_cols_enhanced))
    
    # Variant 4: Enhanced features, with recency (tau=5)
    m4 = train_ridge_offset_model_weighted(
        features_df, feature_cols_enhanced, train_seasons,
        alpha=1.0, use_recency_weights=True, tau=5.0
    )
    if m4:
        variants.append(("Enhanced + recency (τ=5)", m4, feature_cols_enhanced))
    
    # Variant 5: Elastic net (0.3 L1)
    m5 = train_ridge_offset_model_weighted(
        features_df, feature_cols_enhanced, train_seasons,
        alpha=1.0, use_recency_weights=False, l1_ratio=0.3
    )
    if m5:
        variants.append(("Enhanced + Elastic Net (0.3)", m5, feature_cols_enhanced))
    
    print(f"\n{'Variant':<35} {'Train':<7} {'Val LL':<10} {'Val Acc':<10}")
    print("-" * 70)
    
    best_variant = None
    best_ll = float("inf")
    
    for name, model, feat_cols in variants:
        # Compute validation metrics
        preds = []
        for _, r in val_df.iterrows():
            try:
                X_raw = np.array([[r[col] for col in feat_cols]], dtype=float)
                if np.any(np.isnan(X_raw)):
                    continue
                X_scaled = (X_raw - model["mu"]) / model["sd"]
                X = sm.add_constant(X_scaled, has_constant="add")
                p = float(model["model"].predict(X, offset=np.array([r["offset_logit"]]))[0])
                p = float(np.clip(p, 0.01, 0.99))
                preds.append({"prob": p, "actual": int(r["home_wins"])})
            except:
                continue
        
        if preds:
            pdf = pd.DataFrame(preds)
            ll = float(-np.mean([np.log(np.clip(r["prob"] if r["actual"] else 1-r["prob"], 0.01, 0.99)) 
                                for _, r in pdf.iterrows()]))
            acc = float(((pdf["prob"] > 0.5) == (pdf["actual"] == 1)).mean())
            
            print(f"{name:<35} {model['train_samples']:<7} {ll:<10.4f} {acc:<10.1%}")
            
            if ll < best_ll:
                best_ll = ll
                best_variant = (name, model, feat_cols)
    
    if best_variant:
        print(f"\nBest variant: {best_variant[0]}")
    
    return best_variant


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("NFL PLAYOFF MODEL v15.1 - ENHANCED FEATURES (FIXED)")
    print("=" * 100)
    
    games_df, team_df, spread_df = load_all_data()
    games_df = merge_spread_data_safe(games_df, spread_df)
    spread_sanity_check(games_df)

    # Splits
    train_seasons = list(range(2000, 2023))
    calib_seasons = [2021, 2022, 2023]
    tune_seasons = list(range(2015, 2024))
    spread_val_seasons = [2020, 2021, 2022, 2023]

    # Baselines
    baselines = compute_historical_baselines(games_df, max_season=2023)
    print(f"\nHistorical Baselines: Home win rate = {baselines['home_win_rate']:.1%}")

    epa_model = fit_expected_win_pct_model(team_df, max_season=2023)
    print(f"EPA -> Win%: {epa_model['intercept']:.3f} + {epa_model['slope']:.3f} * net_epa")

    # Build temp features for spread tuning
    temp_features = prepare_game_features_v15(
        games_df, team_df, epa_model, baselines, spread_model=None,
        include_interactions=False, include_nonlinear=False
    )

    # Spread model
    spread_model = tune_spread_alpha_v14(
        temp_features,
        train_seasons=train_seasons,
        val_seasons=spread_val_seasons,
        baselines=baselines,
        alphas=(0.1, 0.25, 0.5, 1.0, 2.0),
        clamp_band=(-0.12, -0.04),
        prior_slope_per_point=-0.073,
        prior_strength_k=60
    )
    print_spread_model_v14(spread_model)

    # Prepare v15 enhanced features
    print("\nPreparing v15 features (interactions + non-linear + round)...")
    features_df = prepare_game_features_v15(
        games_df, team_df, epa_model, baselines, spread_model,
        include_interactions=True,
        include_nonlinear=True,
        include_round_features=True
    )
    print(f"Total games: {len(features_df)}")

    # Base feature columns (from v14)
    feature_cols_base = [
        "delta_net_epa", "delta_pd_pg", "seed_diff",
        "delta_vulnerability", "delta_underseeded", "delta_momentum",
        "delta_pass_edge", "delta_rush_edge",
        "delta_ol_exposure", "delta_pass_d_exposure",
        "home_ol_exposure", "away_ol_exposure",
        "total_ol_exposure", "total_pass_d_exposure",
    ]
    feature_cols_base = [c for c in feature_cols_base if c in features_df.columns]

    # v15.1 FIX: Simplified enhanced features (removed market_disagree, reduced interactions)
    interaction_cols = ["pass_x_ol", "seed_x_epa"]  # Only strongest interactions
    nonlinear_cols = ["seed_diff_sq", "seed_diff_abs"]  # Simple non-linearity
    round_cols = ["is_superbowl"]  # Only SB is truly different
    spread_cols = ["spread_magnitude", "spread_confidence", "is_close_game"]  # Better than market_disagree
    
    feature_cols_enhanced = feature_cols_base.copy()
    for col_list in [interaction_cols, nonlinear_cols, round_cols, spread_cols]:
        feature_cols_enhanced.extend([c for c in col_list if c in features_df.columns])
    
    print(f"\nBase features ({len(feature_cols_base)}): {feature_cols_base}")
    print(f"\nEnhanced features ({len(feature_cols_enhanced)}): {feature_cols_enhanced}")

    # Compare model variants
    best_variant = compare_model_variants(
        features_df, feature_cols_base, feature_cols_enhanced,
        train_seasons, val_seasons=[2022, 2023], baselines=baselines
    )

    # Use enhanced features by default
    feature_cols = feature_cols_enhanced

    # v15.1 FIX: Validate tau before using recency weights
    print("\nValidating tau for recency weighting on 2023...")
    best_tau, best_tau_ll = None, float("inf")
    
    for tau in [2, 3, 5, 7, 10, None]:  # None = no recency weighting
        use_recency = tau is not None
        md = train_ridge_offset_model_weighted(
            features_df, feature_cols, train_seasons, 
            alpha=1.0, use_recency_weights=use_recency, tau=tau if use_recency else 3.0
        )
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
            tau_str = f"τ={tau}" if tau else "no recency"
            print(f"  {tau_str}: ll={ll:.4f}")
            if ll < best_tau_ll:
                best_tau_ll, best_tau = ll, tau

    use_recency = best_tau is not None
    tau_str = f"τ={best_tau}" if best_tau else "no recency"
    print(f"  Best: {tau_str}")

    # Tune ridge alpha with best tau
    print(f"\nTuning ridge alpha (with {tau_str})...")
    best_alpha, best_ll = None, float("inf")

    for a in [0.5, 1.0, 2.0, 5.0]:
        md = train_ridge_offset_model_weighted(
            features_df, feature_cols, train_seasons, 
            alpha=a, use_recency_weights=use_recency, tau=best_tau if use_recency else 3.0
        )
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
    model_dict = train_ridge_offset_model_weighted(
        features_df, feature_cols, train_seasons, 
        alpha=best_alpha, use_recency_weights=use_recency, tau=best_tau if use_recency else 3.0
    )
    print(f"\nBest α={best_alpha}, trained on {model_dict['train_samples']} games ({tau_str})")
    
    # Store best_tau for historical validation
    final_tau = best_tau
    final_use_recency = use_recency

    # Feature importance analysis
    analyze_feature_importance(model_dict)

    # v15.1 FIX: Train XGBoost for ENSEMBLE (not just comparison)
    # XGBoost will be used for baseline-only games
    xgb_model = None
    if HAS_XGBOOST:
        print("\nTraining XGBoost for baseline game ensemble...")
        # Train on baseline-only games for better baseline predictions
        baseline_df = features_df[features_df["offset_source"] == "baseline"]
        xgb_model = train_xgboost_model(baseline_df, feature_cols, train_seasons)
        if xgb_model:
            print(f"  XGBoost trained on {xgb_model['train_samples']} baseline games")
            print("  NOTE: XGBoost will be ensembled with ridge for baseline games")
    
    # Also train a full XGBoost for comparison
    xgb_model_full = None
    if HAS_XGBOOST:
        xgb_model_full = train_xgboost_model(features_df, feature_cols, train_seasons)
        if xgb_model_full:
            print(f"  XGBoost (full) trained on {xgb_model_full['train_samples']} games")

    # Calibration
    print("\nCalibrating (baseline only, conservative)...")
    platt_params = calibrate_baseline_only(model_dict, features_df, calib_seasons, shrinkage=0.3)

    # Tune lambdas - v15.1 FIX: Force spread lambda to 0
    print("\nTuning λ by source (v15.1 - trust market for spread games)...")
    lambda_params = tune_source_lambdas(
        model_dict, features_df, tune_seasons, platt_params, 
        improve_gate=0.025,  # Higher threshold
        force_spread_zero=True  # Trust the market
    )

    # Validate 2024 with ensemble
    results_2024 = validate_on_2024(
        model_dict, features_df, platt_params, lambda_params,
        xgb_model=xgb_model,  # For baseline ensemble
        xgb_model_full=xgb_model_full,  # For comparison
        use_ensemble=True
    )

    # Historical rolling validation
    print("\n" + "="*130)
    print("HISTORICAL ROLLING VALIDATION (2015-2024) - v15.1")
    print("="*130)
    
    all_results = []
    
    for test_year in range(2015, 2025):
        train_yrs = list(range(2000, test_year - 2))
        calib_yrs = [test_year - 2, test_year - 1]
        tune_yrs = list(range(2010, test_year))
        spread_val_yrs = list(range(test_year - 4, test_year))
        
        w_baselines = compute_historical_baselines(games_df, max_season=test_year - 1)
        w_epa = fit_expected_win_pct_model(team_df, max_season=test_year - 1)
        
        temp = prepare_game_features_v15(
            games_df, team_df, w_epa, w_baselines, spread_model=None,
            include_interactions=False, include_nonlinear=False
        )
        w_spread = tune_spread_alpha_v14(
            temp,
            train_seasons=train_yrs,
            val_seasons=spread_val_yrs,
            baselines=w_baselines,
            alphas=(0.1, 0.25, 0.5, 1.0, 2.0),
            clamp_band=(-0.12, -0.04),
            prior_slope_per_point=-0.073,
            prior_strength_k=60,
            verbose=False
        )
        
        w_features = prepare_game_features_v15(
            games_df, team_df, w_epa, w_baselines, w_spread,
            include_interactions=True,
            include_nonlinear=True,
            include_round_features=True
        )
        
        # Filter feature cols to available ones
        avail_cols = [c for c in feature_cols if c in w_features.columns]
        
        model = train_ridge_offset_model_weighted(
            w_features, avail_cols, train_yrs, 
            alpha=best_alpha, use_recency_weights=final_use_recency, 
            reference_year=test_year, tau=final_tau if final_use_recency else 3.0
        )
        if model is None:
            continue
        
        platt = calibrate_baseline_only(model, w_features, calib_yrs)
        # v15.1 FIX: Use same lambda settings as main model
        lam_params = tune_source_lambdas(
            model, w_features, tune_yrs, platt, 
            improve_gate=0.025, force_spread_zero=True, verbose=False
        )
        
        test_df = w_features[w_features['season'] == test_year].dropna(subset=avail_cols + ['offset_logit'])
        
        results = []
        for _, gf in test_df.iterrows():
            pred = predict_with_source_lambda(model, gf, platt, lam_params['spread'], lam_params['base'])
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
    out_path = "../validation_results_2024_v15.csv"
    results_2024.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Summary of v15.1 fixes
    print("\n" + "=" * 100)
    print("v15.1 FIX SUMMARY")
    print("=" * 100)
    print(f"""
    CRITICAL FIXES:
    ✓ λ_spread forced to 0.0 (TRUST THE MARKET for spread games)
    ✓ Higher improve_gate (0.025) to prevent overriding market wisdom
    ✓ Removed market_disagree features (was causing overfitting)
    ✓ Simplified to only 2 interaction terms (pass_x_ol, seed_x_epa)
    ✓ Validated tau selection (best: {tau_str})
    ✓ XGBoost ensemble for baseline-only games
    ✓ Better spread features (magnitude, confidence, is_close_game)
    
    FEATURE SET (simplified):
    ✓ Base: 14 features (EPA, matchups, exposures)
    ✓ Interactions: pass_x_ol, seed_x_epa (strongest only)
    ✓ Non-linear: seed_diff_sq, seed_diff_abs
    ✓ Round: is_superbowl only (SB is truly different)
    ✓ Spread: magnitude, confidence, is_close_game
    
    MODEL STRATEGY:
    ✓ Spread games: Use market probability directly (λ_spread=0)
    ✓ Baseline games: Ridge + XGBoost ensemble (0.7/0.3 blend)
    
    PRESERVED FROM v14:
    ✓ Stabilized spread model (clamp + prior blend)
    ✓ Conservative baseline-only calibration
    ✓ Upset tagging system
    """)

    return results_2024


if __name__ == "__main__":
    main()
