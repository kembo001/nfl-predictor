"""
NFL Playoff Model v14.1 — v11 Core + v13/v14 Stabilization + Upset Alerts

WHAT'S NEW vs your v14 run:
- Adds tiered upset tags (WATCH / ALERT / LONGSHOT) that actually fire in big seed-gap games
- Prints underdog probability + tag inline in the 2024 table
- Tracks alert recall (and predicted-upset recall) on 2024 + rolling validation

NOTES:
- Spread games: default λ_spread=0.0 when improvements are tiny, trust market
- Baseline games: λ_base tuned (often 1.0)
- Baseline-only calibration remains conservative (and capped)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
import warnings
warnings.filterwarnings("ignore")

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
    """
    Builds games_df['home_spread'] where:
      - negative => home favored
      - positive => home underdog
    """
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

    # EPA z-scores
    home_pass_off = get_z(home_data, "passing_epa")
    home_rush_off = get_z(home_data, "rushing_epa")
    away_pass_off = get_z(away_data, "passing_epa")
    away_rush_off = get_z(away_data, "rushing_epa")

    home_pass_def = get_z(home_data, "defensive_pass_epa")
    home_rush_def = get_z(home_data, "defensive_rush_epa")
    away_pass_def = get_z(away_data, "defensive_pass_epa")
    away_rush_def = get_z(away_data, "defensive_rush_epa")

    # Edges
    home_pass_edge = home_pass_off - away_pass_def
    home_rush_edge = home_rush_off - away_rush_def
    away_pass_edge = away_pass_off - home_pass_def
    away_rush_edge = away_rush_off - home_rush_def

    features["delta_pass_edge"] = home_pass_edge - away_pass_edge
    features["delta_rush_edge"] = home_rush_edge - away_rush_edge

    # OL/DL
    home_ol = get_z(home_data, "protection_rate")
    away_ol = get_z(away_data, "protection_rate")
    home_dl = get_z(home_data, "pressure_rate")
    away_dl = get_z(away_data, "pressure_rate")

    # Exposures (>=0)
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

    # Raw for reporting/debug
    features["home_ol_z"] = home_ol
    features["away_ol_z"] = away_ol
    features["home_dl_z"] = home_dl
    features["away_dl_z"] = away_dl

    return features


# ============================================================
# SPREAD MODEL v14 STABILIZED (tune alpha + clamp + prior blend)
# ============================================================

def fit_spread_model_v14(features_df, train_seasons, baselines, alpha=0.5,
                         clamp_band=(-0.12, -0.04),
                         prior_slope_per_point=-0.073,
                         prior_strength_k=60):
    """
    Fits ridge logistic on spread games, then stabilizes:
      - clamp slope per point into clamp_band
      - prior blend slope + intercept with simple Bayesian weight w_prior = n/(n+k)
        prior intercept = logit(home_win_rate)
        prior slope = prior_slope_per_point
    """
    train_df = features_df[
        (features_df["season"].isin(train_seasons)) & (features_df["home_spread"].notna())
    ].copy()

    n = len(train_df)
    if n < 20:
        return None

    # Use touchdown scaling in X for numerical stability, but report per-point
    spread_td = train_df["home_spread"].values / 7.0
    X = sm.add_constant(spread_td)
    y = train_df["home_wins"].values

    glm = sm.GLM(y, X, family=sm.families.Binomial())
    model = glm.fit_regularized(alpha=alpha, L1_wt=0.0)

    intercept_hat = float(model.params[0])
    slope_td_hat = float(model.params[1])
    slope_per_point_hat = slope_td_hat / 7.0

    # Clamp slope per point
    slope_per_point_clamped = float(np.clip(slope_per_point_hat, clamp_band[0], clamp_band[1]))

    # Prior blend
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
                          prior_strength_k=60):
    print("\nTuning spread alpha (v14.1: clamp+prior applied in validation)...")

    best_alpha, best_ll, best_model = None, float("inf"), None

    val_df = features_df[
        (features_df["season"].isin(val_seasons)) & (features_df["home_spread"].notna())
    ].copy()

    if len(val_df) < 5:
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

        print(f"  α={a:<4}  val_ll={ll:.4f}  slope/pt={smod['slope_per_point']:+.4f}  w_prior={smod['w_prior']:.2f}")

        if ll < best_ll:
            best_ll, best_alpha, best_model = ll, a, smod

    if best_model is None:
        return None

    print(f"  Best α={best_alpha} (val_ll={best_ll:.4f}), final slope/pt={best_model['slope_per_point']:+.4f}")
    return best_model


def print_spread_model_v14(spread_model):
    if spread_model is None:
        print("Spread Model: None")
        return

    print("\nSpread Model (v14.1 stabilized):")
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
# FEATURE ENGINEERING v14
# ============================================================

def prepare_game_features_v14(games_df, team_df, epa_model, baselines, spread_model):
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

        # Skip if game EPA missing
        if pd.isna(game.get("away_offensive_epa")) or pd.isna(game.get("home_offensive_epa")):
            continue

        away_seed = int(away_data["playoff_seed"])
        home_seed = int(home_data["playoff_seed"])

        # Neutral handling: treat better seed as "home" for consistency
        home_spread = game.get("home_spread", np.nan)
        away, home = orig_away, orig_home
        if is_neutral and away_seed < home_seed:
            # swap
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

        # Matchup
        matchup = compute_matchup_features(away_data, home_data)

        # Vulnerability (expected win% vs actual win%)
        away_exp = float(epa_model["intercept"] + epa_model["slope"] * away_net_epa)
        home_exp = float(epa_model["intercept"] + epa_model["slope"] * home_net_epa)
        away_win_pct = float(away_data["win_pct"])
        home_win_pct = float(home_data["win_pct"])

        # Underseeded (quality rank by PD/G)
        season_teams = team_df[team_df["season"] == season].copy()
        season_teams["pd_pg"] = season_teams["point_differential"] / (
            (season_teams["wins"] + season_teams["losses"]).clip(lower=1)
        )
        season_teams["qrank"] = season_teams["pd_pg"].rank(ascending=False)

        away_q = season_teams[season_teams["team"] == away]["qrank"].values
        home_q = season_teams[season_teams["team"] == home]["qrank"].values
        away_quality_rank = int(away_q[0]) if len(away_q) else len(season_teams) // 2
        home_quality_rank = int(home_q[0]) if len(home_q) else len(season_teams) // 2

        # Momentum residual (safe)
        away_mom = float(away_data.get("momentum_residual", 0) or 0)
        home_mom = float(home_data.get("momentum_residual", 0) or 0)

        # Offsets: spread if available else baseline
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
            "seed_diff": away_seed - home_seed,

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
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# RIDGE OFFSET GLM
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

    # If any features missing, fall back to offset only
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
# CALIBRATION (baseline only, conservative + capped)
# ============================================================

def calibrate_baseline_only(model_dict, features_df, calib_seasons, shrinkage=0.3):
    """
    Calibrate ONLY baseline-offset games.
    Spread games = identity (market).
    """
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

        # reject unstable
        if a_raw < 0 or abs(a_raw) > 10 or abs(b_raw) > 5:
            print(f"  Baseline Platt unstable (a={a_raw:.2f}, b={b_raw:.2f}), using identity")
            return platt

        a = 1.0 + shrinkage * (a_raw - 1.0)
        b = 0.0 + shrinkage * b_raw

        # hard caps
        a = float(np.clip(a, 0.8, 1.2))
        b = float(np.clip(b, -0.2, 0.2))

        platt["baseline"] = {"a": a, "b": b}
        print(f"  Baseline Platt (n={len(preds)}): a={a:.3f}, b={b:.3f}")

    except Exception as e:
        print(f"  Baseline Platt failed: {e}")

    return platt


# ============================================================
# SOURCE-AWARE λ PREDICTION
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

    # calibrate: spread identity, baseline conservative
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
                        improve_gate=0.005):
    """
    Tunes λ separately for spread and baseline games.
    v14 behavior: if spread improvement is tiny, force λ_spread=0.
    """
    tune_df = features_df[features_df["season"].isin(tune_seasons)].dropna(
        subset=model_dict["feature_cols"] + ["offset_logit"]
    )

    best = {"spread": 0.0, "base": 1.0}

    for src, name in [("spread", "spread"), ("baseline", "base")]:
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
            print(f"  λ_spread forced to 0.0 (best Δll={best_imp:+.4f} < {improve_gate})")
        else:
            best[name] = float(best_lam)
            print(f"  λ_{name:<5} = {best_lam:.1f} (Δll={best_imp:+.4f})")

    return best


# ============================================================
# UPSET TAGGING (THIS IS THE FIX THAT MAKES IT SHOW UP)
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

    # NEW: coinflip upset label (seed gap == 1)
    if seed_gap == coinflip_seed_gap and underdog_prob >= coinflip_p:
        tag = "COINFLIP UPSET"

    # Longshot (big seed gap)
    if seed_gap >= longshot_seed_gap and underdog_prob >= longshot_p:
        tag = "UPSET LONGSHOT"

    # Stronger tiers (only for real seed-gap games)
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

    # log loss model
    probs = results_df.apply(
        lambda r: r["home_prob"] if int(r["actual_home_wins"]) == 1 else (1 - r["home_prob"]),
        axis=1,
    )
    out["log_loss"] = float(-np.mean(np.log(probs.clip(0.01, 0.99)))) if len(results_df) else np.nan

    # offset-only log loss
    off = results_df.apply(
        lambda r: expit(r["offset_logit"]) if int(r["actual_home_wins"]) == 1 else (1 - expit(r["offset_logit"])),
        axis=1,
    )
    out["offset_log_loss"] = float(-np.mean(np.log(off.clip(0.01, 0.99)))) if len(results_df) else np.nan

    # baseline log loss
    base = results_df.apply(
        lambda r: r["baseline_prob"] if int(r["actual_home_wins"]) == 1 else (1 - r["baseline_prob"]),
        axis=1,
    )
    out["baseline_log_loss"] = float(-np.mean(np.log(base.clip(0.01, 0.99)))) if len(results_df) else np.nan

    return out


def is_seed_upset(row):
    """
    "Upset" definition: worse seed wins.
    """
    home_seed = int(row["home_seed"])
    away_seed = int(row["away_seed"])
    winner = row["actual"]

    # identify winner seed
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
# 2024 VALIDATION + PRINT
# ============================================================

def validate_on_2024(model_dict, features_df, platt_params, lambda_params):
    print("\n" + "=" * 120)
    print(f"2024 PLAYOFF VALIDATION - MODEL v14.1 (λ_spread={lambda_params['spread']:.1f}, λ_base={lambda_params['base']:.1f})")
    print("=" * 120)

    test_df = features_df[features_df["season"] == 2024].copy()

    round_names = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference", "SB": "Super Bowl"}

    rows = []
    for _, gf in test_df.iterrows():
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

        rows.append({
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
        })

    results_df = pd.DataFrame(rows)

    # Print by round
    for rname in ["Wild Card", "Divisional", "Conference", "Super Bowl"]:
        rg = results_df[results_df["round"] == rname]
        if len(rg) == 0:
            continue

        print(f"\n{rname.upper()}")
        print(f"{'Matchup':<16} {'Seeds':<7} {'Src':<5} {'λ':<4} {'Final':>6} {'Off':>6} {'Δ':>6} {'UD%':>5} {'ALERT':<14} {'Pred':<5} {'Act':<5} ✓")
        print("-" * 120)

        for _, row in rg.iterrows():
            seeds = f"{int(row['away_seed'])}v{int(row['home_seed'])}"
            src = "sprd" if row["offset_source"] == "spread" else "base"
            mark = "✓" if row["correct"] else "✗"
            alert = row["upset_tag"] if row["upset_tag"] else ""
            print(
                f"{row['matchup']:<16} {seeds:<7} {src:<5} {row['lambda_used']:<4.1f} "
                f"{row['home_prob']:>5.0%} {row['offset_prob']:>5.0%} {row['delta_vs_offset']:>+5.0%} "
                f"{row['underdog_prob']:>5.0%} {alert:<14} {row['predicted']:<5} {row['actual']:<5} {mark}"
            )

    # Metrics
    print("\n" + "=" * 120)
    print("METRICS - MODEL v14.1")
    print("=" * 120)

    m = evaluate_predictions(results_df)
    print(f"\nAccuracy: {results_df['correct'].sum()}/{len(results_df)} ({m['accuracy']:.1%})")

    print("\nLog loss (lower is better):")
    print(f"  Model v14.1:  {m['log_loss']:.4f}")
    print(f"  Offset-only:  {m['offset_log_loss']:.4f}")
    print(f"  Baseline:     {m['baseline_log_loss']:.4f}")
    print(f"  vs Baseline:  {m['baseline_log_loss'] - m['log_loss']:+.4f} {'✓' if (m['baseline_log_loss'] - m['log_loss']) > 0 else '✗'}")
    print(f"  vs Offset:    {m['offset_log_loss'] - m['log_loss']:+.4f} {'✓' if (m['offset_log_loss'] - m['log_loss']) > 0 else '✗'}")

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

        # Alerts
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
# MAIN
# ============================================================

def main():
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
    temp_features = prepare_game_features_v14(games_df, team_df, epa_model, baselines, spread_model=None)

    # Spread model: tune alpha, clamp, prior blend
    spread_model = tune_spread_alpha_v14(
        temp_features,
        train_seasons=train_seasons,
        val_seasons=spread_val_seasons,
        baselines=baselines,
        alphas=(0.1, 0.25, 0.5, 1.0, 2.0),
        # If you want spread closer to v11, tighten this band:
        clamp_band=(-0.12, -0.04),
        prior_slope_per_point=-0.073,
        prior_strength_k=60
    )
    print_spread_model_v14(spread_model)

    # Final features
    print("\nPreparing v14.1 features...")
    features_df = prepare_game_features_v14(games_df, team_df, epa_model, baselines, spread_model)
    print(f"Total games: {len(features_df)}")

    feature_cols = [
        "delta_net_epa", "delta_pd_pg", "seed_diff",
        "delta_vulnerability", "delta_underseeded", "delta_momentum",
        "delta_pass_edge", "delta_rush_edge",
        "delta_ol_exposure", "delta_pass_d_exposure",
        "home_ol_exposure", "away_ol_exposure",
        "total_ol_exposure", "total_pass_d_exposure",
    ]
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    # Tune ridge alpha on 2023
    print("\nTraining ridge offset model (select alpha on 2023)...")
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

    model_dict = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=best_alpha)
    print(f"\nBest α={best_alpha}, trained on {model_dict['train_samples']} games")

    print("\nCoefficients (standardized):")
    for name, coef in zip(["const"] + feature_cols, model_dict["model"].params):
        print(f"  {name:<22} {float(coef):>+8.4f}")

    # Calibration baseline only
    print("\nCalibrating (baseline only, conservative)...")
    platt_params = calibrate_baseline_only(model_dict, features_df, calib_seasons, shrinkage=0.3)

    # Tune lambdas
    print("\nTuning λ by source (v14.1)...")
    lambda_params = tune_source_lambdas(model_dict, features_df, tune_seasons, platt_params, improve_gate=0.005)

    # Validate 2024 + alerts
    results_2024 = validate_on_2024(model_dict, features_df, platt_params, lambda_params)

    # Save
    out_path = "../validation_results_2024_v14.csv"
    results_2024.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return results_2024


if __name__ == "__main__":
    main()
