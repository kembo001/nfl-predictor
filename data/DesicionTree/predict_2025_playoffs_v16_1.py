"""
NFL Playoff Predictor 2025 - Based on Model v16.1
Trained on 2000-2024 playoff data, predicting 2025 playoffs

v16.1 Key Fixes from v16:
1. Fixed toss-up logic for baseline games (no more abs_spread=7.0 default)
2. Ridge trained baseline-only (no train/inference mismatch)
3. Added has_spread + tossup_baseline features
4. Spread games return spread-offset probability directly (no ridge, no ensemble)
5. More conservative XGBoost defaults

Wild Card Matchups:
AFC: #7 LAC @ #2 NE, #6 BUF @ #3 JAX, #5 HOU @ #4 PIT
NFC: #7 GB @ #2 CHI, #6 SF @ #3 PHI, #5 LA @ #4 CAR
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
    print("Note: XGBoost not installed. Using ridge-only predictions.")

# ============================================================
# DATA LOADING
# ============================================================

def load_training_data():
    games_df = pd.read_csv("../nfl_playoff_results_2000_2024_with_epa.csv")
    team_df = pd.read_csv("../df_with_upsets_merged.csv")
    spread_df = pd.read_csv("../csv/nfl_biggest_playoff_upsets_2000_2024.csv")
    print(f"Loaded {len(games_df)} historical playoff games (2000-2024)")
    print(f"Loaded {len(team_df)} team-season records")
    return games_df, team_df, spread_df

def load_2025_teams():
    df = pd.read_csv("../csv/2025/team_stats_2025_playoffs.csv")
    df["season"] = 2025
    df["net_epa"] = df["total_offensive_epa"] - df["defensive_epa"]
    df["momentum_residual"] = 0.0
    print(f"Loaded {len(df)} playoff teams for 2025")
    return df

def merge_spread_data_safe(games_df, spread_df):
    games_df = games_df.copy()
    spread_lookup = {}
    for _, row in spread_df.iterrows():
        season, game_type = row["season"], row["game_type"]
        underdog, magnitude = row["underdog"], abs(row["spread_magnitude"])
        teams = {row["winner"], row["loser"]}
        matching = games_df[(games_df["season"] == season) & (games_df["game_type"] == game_type) & 
                           (games_df["home_team"].isin(teams)) & (games_df["away_team"].isin(teams))]
        if len(matching) > 0:
            game = matching.iloc[0]
            home = game["home_team"]
            home_spread = magnitude if home == underdog else -magnitude
            spread_lookup[(season, game_type, game["away_team"], home)] = home_spread
    games_df["home_spread"] = games_df.apply(
        lambda r: spread_lookup.get((r["season"], r["game_type"], r["away_team"], r["home_team"]), np.nan), axis=1)
    print(f"Matched spread data for {games_df['home_spread'].notna().sum()}/{len(games_df)} games")
    return games_df

# ============================================================
# BASELINES & Z-SCORES
# ============================================================

def compute_historical_baselines(games_df, max_season):
    train_games = games_df[games_df["season"] <= max_season]
    home_games = train_games[train_games["location"] == "Home"]
    if len(home_games) == 0: return {"home_win_rate": 0.55}
    home_wins = (home_games["winner"] == home_games["home_team"]).sum()
    return {"home_win_rate": home_wins / len(home_games)}

def baseline_probability(home_seed, away_seed, is_neutral, baselines):
    seed_diff = away_seed - home_seed
    if is_neutral: base_prob = 0.50 + (seed_diff * 0.03)
    else: base_prob = baselines["home_win_rate"] + (seed_diff * 0.02)
    return np.clip(base_prob, 0.20, 0.85)

def fit_expected_win_pct_model(team_df, max_season):
    train_data = team_df[team_df["season"] <= max_season].dropna(subset=["win_pct", "net_epa"])
    if len(train_data) < 20: return {"intercept": 0.5, "slope": 2.0}
    X = sm.add_constant(train_data["net_epa"].values.reshape(-1, 1))
    model = sm.OLS(train_data["win_pct"].values, X).fit()
    return {"intercept": model.params[0], "slope": model.params[1]}

def compute_season_zscores(team_df):
    team_df = team_df.copy()
    higher = ["passing_epa", "rushing_epa", "total_offensive_epa", "net_epa", "point_differential", 
              "win_pct", "pressure_rate", "sack_rate", "protection_rate"]
    lower = ["defensive_epa", "defensive_pass_epa", "defensive_rush_epa", "sacks_allowed_rate"]
    for stat in higher:
        if stat in team_df.columns:
            team_df[f"z_{stat}"] = team_df.groupby("season")[stat].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    for stat in lower:
        if stat in team_df.columns:
            team_df[f"z_{stat}"] = team_df.groupby("season")[stat].transform(lambda x: -(x - x.mean()) / (x.std() + 1e-9))
    return team_df

# ============================================================
# MATCHUP FEATURES
# ============================================================

def compute_matchup_features(away_data, home_data):
    features = {}
    def get_z(data, stat, default=0.0):
        z_col = f"z_{stat}"
        return float(data[z_col]) if (z_col in data.index and pd.notna(data[z_col])) else default
    
    home_pass_off, home_rush_off = get_z(home_data, "passing_epa"), get_z(home_data, "rushing_epa")
    away_pass_off, away_rush_off = get_z(away_data, "passing_epa"), get_z(away_data, "rushing_epa")
    home_pass_def, home_rush_def = get_z(home_data, "defensive_pass_epa"), get_z(home_data, "defensive_rush_epa")
    away_pass_def, away_rush_def = get_z(away_data, "defensive_pass_epa"), get_z(away_data, "defensive_rush_epa")
    
    features["delta_pass_edge"] = (home_pass_off - away_pass_def) - (away_pass_off - home_pass_def)
    features["delta_rush_edge"] = (home_rush_off - away_rush_def) - (away_rush_off - home_rush_def)
    
    home_ol, away_ol = get_z(home_data, "protection_rate"), get_z(away_data, "protection_rate")
    home_dl, away_dl = get_z(home_data, "pressure_rate"), get_z(away_data, "pressure_rate")
    
    home_ol_exp, away_ol_exp = float(np.maximum(0, away_dl - home_ol)), float(np.maximum(0, home_dl - away_ol))
    features.update({"home_ol_exposure": home_ol_exp, "away_ol_exposure": away_ol_exp, 
                    "delta_ol_exposure": away_ol_exp - home_ol_exp, "total_ol_exposure": home_ol_exp + away_ol_exp})
    
    home_pass_d_exp = float(np.maximum(0, away_pass_off - home_pass_def))
    away_pass_d_exp = float(np.maximum(0, home_pass_off - away_pass_def))
    features.update({"home_pass_d_exposure": home_pass_d_exp, "away_pass_d_exposure": away_pass_d_exp,
                    "delta_pass_d_exposure": away_pass_d_exp - home_pass_d_exp,
                    "total_pass_d_exposure": home_pass_d_exp + away_pass_d_exp})
    features["delta_rush_d_exposure"] = float(np.maximum(0, home_rush_off - away_rush_def) - 
                                              np.maximum(0, away_rush_off - home_rush_def))
    features.update({"home_ol_z": home_ol, "away_ol_z": away_ol, "home_dl_z": home_dl, "away_dl_z": away_dl})
    return features

# ============================================================
# SPREAD MODEL
# ============================================================

def fit_spread_model(features_df, train_seasons, baselines, alpha=0.5):
    train_df = features_df[(features_df["season"].isin(train_seasons)) & (features_df["home_spread"].notna())].copy()
    if len(train_df) < 20: return None
    spread_td, y = train_df["home_spread"].values / 7.0, train_df["home_wins"].values
    X = sm.add_constant(spread_td)
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit_regularized(alpha=alpha, L1_wt=0.0)
    prior_slope = -0.073
    w = len(train_df) / (len(train_df) + 60)
    slope = float(np.clip(model.params[1] / 7.0, -0.12, -0.04))
    slope_final = w * slope + (1 - w) * prior_slope
    intercept_final = w * float(model.params[0]) + (1 - w) * logit(np.clip(baselines["home_win_rate"], 0.05, 0.95))
    return {"intercept": intercept_final, "slope_per_point": slope_final, "n": len(train_df)}

def get_spread_offset_logit(home_spread, spread_model):
    if spread_model is None or pd.isna(home_spread): return np.nan
    return float(np.clip(spread_model["intercept"] + spread_model["slope_per_point"] * float(home_spread), -4, 4))

# ============================================================
# FEATURE ENGINEERING (v16.1)
# ============================================================

def create_spread_features(home_spread):
    """v16.1: Added has_spread feature, confidence=0 when missing"""
    if pd.isna(home_spread):
        return {"has_spread": 0, "spread_magnitude": 0.0, "spread_confidence": 0.0, "is_close_game": 0}
    spread = abs(float(home_spread))
    return {"has_spread": 1, "spread_magnitude": float(spread), 
            "spread_confidence": float(1.0 / (spread + 3.0)), "is_close_game": 1 if spread <= 3.0 else 0}

def create_tossup_features(seed_diff, home_spread):
    """
    v16.1 FIX: Baseline games (no spread) can be tossups on seeds alone.
    Missing spread => treat as uncertain/close rather than defaulting to abs_spread=7.0
    """
    abs_seed_diff = abs(int(seed_diff))
    has_spread = pd.notna(home_spread)
    
    if has_spread:
        close_spread = abs(float(home_spread)) <= 3.5
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

def prepare_game_features(games_df, team_df, epa_model, baselines, spread_model):
    team_df = compute_season_zscores(team_df)
    rows = []
    for _, game in games_df.iterrows():
        season, is_neutral = int(game["season"]), (game["location"] == "Neutral")
        away_row = team_df[(team_df["team"] == game["away_team"]) & (team_df["season"] == season)]
        home_row = team_df[(team_df["team"] == game["home_team"]) & (team_df["season"] == season)]
        if len(away_row) == 0 or len(home_row) == 0: continue
        if pd.isna(game.get("away_offensive_epa")) or pd.isna(game.get("home_offensive_epa")): continue
        
        away_data, home_data = away_row.iloc[0], home_row.iloc[0]
        away_seed, home_seed = int(away_data["playoff_seed"]), int(home_data["playoff_seed"])
        home_spread = game.get("home_spread", np.nan)
        
        if is_neutral and away_seed < home_seed:
            away_data, home_data = home_data, away_data
            away_seed, home_seed = home_seed, away_seed
            if pd.notna(home_spread): home_spread = -float(home_spread)
        
        away_games = float(away_data["wins"] + away_data["losses"])
        home_games = float(home_data["wins"] + home_data["losses"])
        away_net_epa = float(away_data.get("net_epa", 0) or 0)
        home_net_epa = float(home_data.get("net_epa", 0) or 0)
        
        matchup = compute_matchup_features(away_data, home_data)
        seed_diff = away_seed - home_seed
        baseline_prob = float(baseline_probability(home_seed, away_seed, is_neutral, baselines))
        baseline_logit = float(logit(np.clip(baseline_prob, 0.01, 0.99)))
        spread_offset = get_spread_offset_logit(home_spread, spread_model)
        
        offset_logit = float(spread_offset) if pd.notna(spread_offset) else baseline_logit
        offset_source = "spread" if pd.notna(spread_offset) else "baseline"
        
        # Quality rank for underseeded
        season_teams = team_df[team_df["season"] == season].copy()
        season_teams["pd_pg"] = season_teams["point_differential"] / ((season_teams["wins"] + season_teams["losses"]).clip(lower=1))
        season_teams["qrank"] = season_teams["pd_pg"].rank(ascending=False)
        away_q = season_teams[season_teams["team"] == game["away_team"]]["qrank"].values
        home_q = season_teams[season_teams["team"] == game["home_team"]]["qrank"].values
        away_quality_rank = int(away_q[0]) if len(away_q) else len(season_teams) // 2
        home_quality_rank = int(home_q[0]) if len(home_q) else len(season_teams) // 2
        
        row = {"season": season, "game_type": game["game_type"], "away_team": game["away_team"], 
               "home_team": game["home_team"], "winner": game["winner"], 
               "home_wins": 1 if game["winner"] == game["home_team"] else 0,
               "is_neutral": 1 if is_neutral else 0, "away_seed": away_seed, "home_seed": home_seed,
               "seed_diff": seed_diff, "seed_diff_sq": seed_diff**2, "seed_diff_abs": abs(seed_diff),
               "delta_net_epa": home_net_epa - away_net_epa,
               "delta_pd_pg": float(home_data["point_differential"]/max(home_games,1) - away_data["point_differential"]/max(away_games,1)),
               "delta_vulnerability": (float(away_data["win_pct"]) - float(epa_model["intercept"] + epa_model["slope"]*away_net_epa)) -
                                     (float(home_data["win_pct"]) - float(epa_model["intercept"] + epa_model["slope"]*home_net_epa)),
               "delta_underseeded": (away_seed - away_quality_rank) - (home_seed - home_quality_rank),
               "delta_momentum": float(home_data.get("momentum_residual", 0) or 0) - float(away_data.get("momentum_residual", 0) or 0),
               "baseline_prob": baseline_prob, "baseline_logit": baseline_logit, "home_spread": home_spread,
               "spread_offset": spread_offset, "offset_logit": offset_logit, "offset_source": offset_source}
        row.update(matchup)
        # Interactions
        row["pass_x_ol"] = float(row["delta_pass_edge"] * row["delta_ol_exposure"])
        row["seed_x_epa"] = float(seed_diff * row["delta_net_epa"])
        # Round
        row["is_superbowl"] = 1 if game["game_type"] == "SB" else 0
        # v16.1 spread features
        row.update(create_spread_features(home_spread))
        # v16.1 tossup features (FIXED)
        row.update(create_tossup_features(seed_diff, home_spread))
        rows.append(row)
    return pd.DataFrame(rows)

# ============================================================
# RIDGE MODEL (v16.1: baseline-only training)
# ============================================================

def train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=1.0):
    """v16.1: Train ridge ONLY on baseline games (no spread train/inference mismatch)"""
    train_df = features_df[
        (features_df["season"].isin(train_seasons)) & 
        (features_df["offset_source"] == "baseline")
    ].dropna(subset=feature_cols + ["offset_logit"])
    if len(train_df) < 30: return None
    X_raw = train_df[feature_cols].values
    mu, sd = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-9
    X = sm.add_constant((X_raw - mu) / sd)
    try: res = sm.GLM(train_df["home_wins"].values, X, family=sm.families.Binomial(), 
                      offset=train_df["offset_logit"].values).fit_regularized(alpha=alpha, L1_wt=0.0)
    except: return None
    return {"model": res, "feature_cols": feature_cols, "mu": mu, "sd": sd, "train_samples": len(train_df)}

def predict_raw(model_dict, game_features):
    if model_dict is None: return None
    try: X_raw = np.array([[game_features[col] for col in model_dict["feature_cols"]]], dtype=float)
    except: return None
    offset_logit = float(game_features["offset_logit"])
    if np.any(np.isnan(X_raw)): return {"raw_prob": float(expit(np.clip(offset_logit, -4, 4))), "adjustment": 0.0, "offset_logit": offset_logit}
    X = sm.add_constant((X_raw - model_dict["mu"]) / model_dict["sd"], has_constant="add")
    p = float(np.clip(model_dict["model"].predict(X, offset=np.array([offset_logit]))[0], 0.01, 0.99))
    return {"raw_prob": p, "adjustment": float(logit(p) - offset_logit), "offset_logit": offset_logit}

# ============================================================
# XGBOOST (v16.1: more conservative defaults)
# ============================================================

def train_xgboost_calibrated(features_df, feature_cols, train_seasons, calib_seasons):
    """v16.1: More conservative XGBoost defaults for small sample sizes"""
    if not HAS_XGBOOST: return None
    train_df = features_df[features_df["season"].isin(train_seasons)].dropna(subset=feature_cols)
    if len(train_df) < 30: return None
    try:
        xgb = XGBClassifier(
            n_estimators=500, max_depth=2, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=2.0, reg_alpha=0.0,
            objective='binary:logistic', eval_metric='logloss',
            verbosity=0, random_state=42
        )
        xgb.fit(train_df[feature_cols].values, train_df["home_wins"].values)
    except: return None
    
    calib_df = features_df[features_df["season"].isin(calib_seasons)].dropna(subset=feature_cols)
    if len(calib_df) < 10:
        return {"model": xgb, "feature_cols": feature_cols, "platt_a": 1.0, "platt_b": 0.0, 
                "calibrated": False, "train_samples": len(train_df)}
    
    raw_probs = xgb.predict_proba(calib_df[feature_cols].values)[:, 1]
    raw_logits = logit(np.clip(raw_probs, 0.01, 0.99))
    try:
        platt_fit = sm.GLM(calib_df["home_wins"].values, sm.add_constant(raw_logits), family=sm.families.Binomial()).fit()
        platt_b, platt_a = float(platt_fit.params[0]), float(platt_fit.params[1])
        if platt_a <= 0 or abs(platt_a) > 5 or abs(platt_b) > 3: platt_a, platt_b = 1.0, 0.0
    except: platt_a, platt_b = 1.0, 0.0
    return {"model": xgb, "feature_cols": feature_cols, "platt_a": platt_a, "platt_b": platt_b, 
            "calibrated": True, "train_samples": len(train_df), "calib_samples": len(calib_df)}

def predict_xgboost_calibrated(model_dict, game_features):
    if model_dict is None: return None
    try:
        X = np.array([[game_features[col] for col in model_dict["feature_cols"]]], dtype=float)
        if np.any(np.isnan(X)): return None
        raw_prob = float(model_dict["model"].predict_proba(X)[0, 1])
        raw_logit = logit(np.clip(raw_prob, 0.01, 0.99))
        cal_logit = model_dict["platt_b"] + model_dict["platt_a"] * raw_logit
        return {"raw_prob": raw_prob, "calibrated_prob": float(expit(np.clip(cal_logit, -4, 4)))}
    except: return None

# ============================================================
# CALIBRATION
# ============================================================

def calibrate_baseline_only(model_dict, features_df, calib_seasons, shrinkage=0.3):
    platt = {"spread": {"a": 1.0, "b": 0.0}, "baseline": {"a": 1.0, "b": 0.0}}
    calib_df = features_df[(features_df["season"].isin(calib_seasons)) & 
                           (features_df["offset_source"] == "baseline")].dropna(subset=model_dict["feature_cols"] + ["offset_logit"])
    if len(calib_df) < 5: return platt
    
    preds = []
    for _, r in calib_df.iterrows():
        pr = predict_raw(model_dict, r)
        if pr: preds.append({"prob": pr["raw_prob"], "actual": int(r["home_wins"])})
    if len(preds) < 5: return platt
    
    df = pd.DataFrame(preds)
    try:
        fit = sm.GLM(df["actual"].values, sm.add_constant(logit(df["prob"].clip(0.05, 0.95)).values), 
                    family=sm.families.Binomial()).fit()
        a_raw, b_raw = float(fit.params[1]), float(fit.params[0])
        if a_raw < 0 or abs(a_raw) > 10 or abs(b_raw) > 5: return platt
        a = float(np.clip(1.0 + shrinkage * (a_raw - 1.0), 0.8, 1.2))
        b = float(np.clip(0.0 + shrinkage * b_raw, -0.2, 0.2))
        platt["baseline"] = {"a": a, "b": b}
    except: pass
    return platt

# ============================================================
# ENSEMBLE PREDICTION (v16.1)
# ============================================================

def predict_ensemble_v16_1(model_dict, game_features, platt_params, 
                           xgb_model=None, xgb_weight=0.3, lambda_base=1.0):
    """
    v16.1 Ensemble:
    - Spread games: return spread-offset probability directly (NO ridge, NO ensemble)
    - Baseline games: ridge + calibration + optional XGB ensemble
    """
    src = game_features["offset_source"]
    offset_logit = float(game_features["offset_logit"])
    
    # v16.1: Spread games bypass everything - trust market directly
    if src == "spread":
        p = float(expit(np.clip(offset_logit, -4, 4)))
        return {"home_prob": p, "predicted_winner": "home" if p > 0.5 else "away", 
                "is_ensemble": False, "lambda_used": 0.0}
    
    # Baseline games: ridge + calibration
    pr = predict_raw(model_dict, game_features)
    if pr is None: return None
    
    blended_logit = float(np.clip(offset_logit + lambda_base * pr["adjustment"], -4, 4))
    blended_prob = float(expit(blended_logit))
    
    # Apply Platt calibration
    pl = platt_params.get("baseline", {"a": 1.0, "b": 0.0})
    cal_logit = float(np.clip(pl["b"] + pl["a"] * logit(np.clip(blended_prob, 0.05, 0.95)), -4, 4))
    ridge_prob = float(expit(cal_logit))
    
    final_prob = ridge_prob
    
    # XGB ensemble (baseline games only)
    if xgb_model is not None:
        xgb_pred = predict_xgboost_calibrated(xgb_model, game_features)
        if xgb_pred is not None:
            xgb_prob = float(xgb_pred["calibrated_prob"])
            final_prob = float(np.clip((1 - xgb_weight) * ridge_prob + xgb_weight * xgb_prob, 0.01, 0.99))
    
    return {"home_prob": final_prob, "predicted_winner": "home" if final_prob > 0.5 else "away",
            "ridge_prob": ridge_prob, "is_ensemble": xgb_model is not None, "lambda_used": lambda_base}

def upset_tag(home_seed, away_seed, home_prob):
    seed_gap = abs(away_seed - home_seed)
    underdog_prob = home_prob if home_seed > away_seed else (1 - home_prob) if away_seed > home_seed else min(home_prob, 1-home_prob)
    if seed_gap >= 4 and underdog_prob >= 0.25: return "UPSET LONGSHOT", underdog_prob
    if seed_gap >= 2 and underdog_prob >= 0.35: return "UPSET ALERT", underdog_prob
    if seed_gap >= 2 and underdog_prob >= 0.30: return "UPSET WATCH", underdog_prob
    if seed_gap == 1 and underdog_prob >= 0.47: return "COINFLIP", underdog_prob
    return "", underdog_prob

# ============================================================
# 2025 PREDICTION
# ============================================================

def create_2025_matchup(away_team, home_team, team_df_2025, game_type, is_neutral=False):
    away_data = team_df_2025[team_df_2025["team"] == away_team].iloc[0]
    home_data = team_df_2025[team_df_2025["team"] == home_team].iloc[0]
    return {"season": 2025, "game_type": game_type, "away_team": away_team, "home_team": home_team,
            "location": "Neutral" if is_neutral else "Home", "away_offensive_epa": away_data["total_offensive_epa"],
            "home_offensive_epa": home_data["total_offensive_epa"], "winner": None}

def prepare_2025_game_features(matchup, team_df_2025, epa_model, baselines):
    team_df_2025 = compute_season_zscores(team_df_2025)
    away_data = team_df_2025[team_df_2025["team"] == matchup["away_team"]].iloc[0]
    home_data = team_df_2025[team_df_2025["team"] == matchup["home_team"]].iloc[0]
    is_neutral = matchup["location"] == "Neutral"
    away_seed, home_seed = int(away_data["playoff_seed"]), int(home_data["playoff_seed"])
    
    if is_neutral and away_seed < home_seed:
        away_data, home_data = home_data, away_data
        away_seed, home_seed = home_seed, away_seed
    
    away_games, home_games = float(away_data["wins"] + away_data["losses"]), float(home_data["wins"] + home_data["losses"])
    away_net_epa, home_net_epa = float(away_data["net_epa"]), float(home_data["net_epa"])
    
    # Quality ranks
    team_df_copy = team_df_2025.copy()
    team_df_copy["pd_pg"] = team_df_copy["point_differential"] / ((team_df_copy["wins"] + team_df_copy["losses"]).clip(lower=1))
    team_df_copy["qrank"] = team_df_copy["pd_pg"].rank(ascending=False)
    away_q = team_df_copy[team_df_copy["team"] == matchup["away_team"]]["qrank"].values
    home_q = team_df_copy[team_df_copy["team"] == matchup["home_team"]]["qrank"].values
    away_quality_rank = int(away_q[0]) if len(away_q) else 7
    home_quality_rank = int(home_q[0]) if len(home_q) else 7
    
    matchup_feats = compute_matchup_features(away_data, home_data)
    seed_diff = away_seed - home_seed
    baseline_prob = float(baseline_probability(home_seed, away_seed, is_neutral, baselines))
    baseline_logit = float(logit(np.clip(baseline_prob, 0.01, 0.99)))
    
    row = {"season": 2025, "game_type": matchup["game_type"], "away_team": matchup["away_team"],
           "home_team": matchup["home_team"], "is_neutral": 1 if is_neutral else 0,
           "away_seed": away_seed, "home_seed": home_seed, "seed_diff": seed_diff,
           "seed_diff_sq": seed_diff**2, "seed_diff_abs": abs(seed_diff),
           "delta_net_epa": home_net_epa - away_net_epa,
           "delta_pd_pg": float(home_data["point_differential"]/max(home_games,1) - away_data["point_differential"]/max(away_games,1)),
           "delta_vulnerability": (float(away_data["win_pct"]) - float(epa_model["intercept"] + epa_model["slope"]*away_net_epa)) -
                                 (float(home_data["win_pct"]) - float(epa_model["intercept"] + epa_model["slope"]*home_net_epa)),
           "delta_underseeded": (away_seed - away_quality_rank) - (home_seed - home_quality_rank),
           "delta_momentum": 0.0, "baseline_prob": baseline_prob, "baseline_logit": baseline_logit,
           "home_spread": np.nan, "spread_offset": np.nan, "offset_logit": baseline_logit, "offset_source": "baseline"}
    row.update(matchup_feats)
    row["pass_x_ol"] = float(row["delta_pass_edge"] * row["delta_ol_exposure"])
    row["seed_x_epa"] = float(seed_diff * row["delta_net_epa"])
    row["is_superbowl"] = 1 if matchup["game_type"] == "SB" else 0
    # v16.1 spread features (no spread for 2025)
    row.update(create_spread_features(np.nan))
    # v16.1 tossup features (FIXED - baseline games can be tossups)
    row.update(create_tossup_features(seed_diff, np.nan))
    return row

def print_prediction(pred, game_feats, round_name):
    away, home = game_feats["away_team"], game_feats["home_team"]
    away_seed, home_seed = int(game_feats["away_seed"]), int(game_feats["home_seed"])
    home_prob = pred["home_prob"]
    winner = home if pred["predicted_winner"] == "home" else away
    tag, ud_prob = upset_tag(home_seed, away_seed, home_prob)
    is_tossup = "T" if game_feats.get("is_tossup", 0) else ""
    tossup_base = "CAR" if game_feats.get("tossup_baseline", 0) else ""
    
    print(f"\n  {away} ({away_seed}) @ {home} ({home_seed}) {is_tossup}{tossup_base}")
    print(f"    Home Win Prob: {home_prob:.1%}  |  Predicted: {winner}")
    print(f"    Seed Diff: {game_feats['seed_diff']}  |  EPA Diff: {game_feats['delta_net_epa']:+.1f}")
    if tag: print(f"    ⚠️  {tag} ({ud_prob:.1%})")
    return winner

def main():
    print("=" * 80)
    print("NFL 2025 PLAYOFF PREDICTIONS - Model v16.1")
    print("=" * 80)
    
    # Load data
    games_df, team_df_hist, spread_df = load_training_data()
    team_df_2025 = load_2025_teams()
    games_df = merge_spread_data_safe(games_df, spread_df)
    
    team_df = pd.concat([team_df_hist, team_df_2025], ignore_index=True)
    
    # Training configuration
    train_seasons = list(range(2000, 2025))
    calib_seasons = [2022, 2023, 2024]
    
    # Compute baselines
    baselines = compute_historical_baselines(games_df, max_season=2024)
    epa_model = fit_expected_win_pct_model(team_df_hist, max_season=2024)
    print(f"\nHistorical home win rate: {baselines['home_win_rate']:.1%}")
    
    # Prepare features
    spread_model = fit_spread_model(prepare_game_features(games_df, team_df_hist, epa_model, baselines, None),
                                    train_seasons, baselines)
    features_df = prepare_game_features(games_df, team_df_hist, epa_model, baselines, spread_model)
    
    # v16.1 Feature columns (includes has_spread and tossup_baseline)
    feature_cols = ["delta_net_epa", "delta_pd_pg", "seed_diff", "delta_vulnerability", 
                   "delta_underseeded", "delta_momentum",
                   "delta_pass_edge", "delta_rush_edge", "delta_ol_exposure", "delta_pass_d_exposure",
                   "home_ol_exposure", "away_ol_exposure", "total_ol_exposure", "total_pass_d_exposure",
                   "pass_x_ol", "seed_x_epa", "seed_diff_sq", "seed_diff_abs", "is_superbowl",
                   "has_spread", "spread_magnitude", "spread_confidence", "is_close_game",
                   "is_tossup", "tossup_baseline", "is_big_favorite", "is_medium_gap", "is_close_seeds"]
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    
    # v16.1: Train ridge on baseline games only
    print(f"\nTraining ridge model on baseline games only...")
    model_dict = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=1.0)
    print(f"  Trained on {model_dict['train_samples']} baseline games")
    
    # v16.1: XGBoost for baseline games (more conservative)
    xgb_model = None
    if HAS_XGBOOST:
        print("Training calibrated XGBoost for baseline games...")
        baseline_df = features_df[features_df["offset_source"] == "baseline"]
        xgb_model = train_xgboost_calibrated(baseline_df, feature_cols, train_seasons, calib_seasons)
        if xgb_model: 
            print(f"  XGBoost trained on {xgb_model['train_samples']} baseline games")
            print(f"  Platt calibrated: {xgb_model['calibrated']}")
    
    # Calibration
    platt_params = calibrate_baseline_only(model_dict, features_df, calib_seasons)
    
    # v16.1 parameters
    lambda_base = 1.0
    xgb_weight = 0.3
    
    print(f"\nv16.1 Parameters:")
    print(f"  λ_spread=0.0 (trust market), λ_base={lambda_base}")
    print(f"  XGB weight={xgb_weight}")
    print(f"  Tossup detection: baseline games can be tossups on seeds alone")
    
    # ============================================================
    # 2025 PLAYOFF PREDICTIONS
    # ============================================================
    
    print("\n" + "=" * 80)
    print("2025 NFL PLAYOFF PREDICTIONS (v16.1)")
    print("=" * 80)
    
    # Wild Card Round
    print("\n" + "-" * 40)
    print("WILD CARD ROUND")
    print("-" * 40)
    
    wc_matchups = [
        ("LAC", "NE", "AFC"),
        ("BUF", "JAX", "AFC"),
        ("HOU", "PIT", "AFC"),
        ("GB", "CHI", "NFC"),
        ("SF", "PHI", "NFC"),
        ("LA", "CAR", "NFC"),
    ]
    
    wc_winners = {}
    for away, home, conf in wc_matchups:
        matchup = create_2025_matchup(away, home, team_df_2025, "WC")
        game_feats = prepare_2025_game_features(matchup, team_df_2025, epa_model, baselines)
        pred = predict_ensemble_v16_1(model_dict, game_feats, platt_params, xgb_model, xgb_weight, lambda_base)
        winner = print_prediction(pred, game_feats, "Wild Card")
        wc_winners[(conf, away, home)] = winner
    
    # Divisional Round
    print("\n" + "-" * 40)
    print("DIVISIONAL ROUND (Projected)")
    print("-" * 40)
    
    afc_wc_winners = [wc_winners[("AFC", "LAC", "NE")], wc_winners[("AFC", "BUF", "JAX")], wc_winners[("AFC", "HOU", "PIT")]]
    afc_seeds = {"DEN": 1, "NE": 2, "JAX": 3, "PIT": 4, "HOU": 5, "BUF": 6, "LAC": 7}
    afc_survivors = sorted(afc_wc_winners, key=lambda x: afc_seeds[x])
    
    nfc_wc_winners = [wc_winners[("NFC", "GB", "CHI")], wc_winners[("NFC", "SF", "PHI")], wc_winners[("NFC", "LA", "CAR")]]
    nfc_seeds = {"SEA": 1, "CHI": 2, "PHI": 3, "CAR": 4, "LA": 5, "SF": 6, "GB": 7}
    nfc_survivors = sorted(nfc_wc_winners, key=lambda x: nfc_seeds[x])
    
    afc_div_matchups = [
        (afc_survivors[2], "DEN"),
        (afc_survivors[1], afc_survivors[0]) if afc_seeds[afc_survivors[0]] < afc_seeds[afc_survivors[1]] else (afc_survivors[0], afc_survivors[1])
    ]
    
    nfc_div_matchups = [
        (nfc_survivors[2], "SEA"),
        (nfc_survivors[1], nfc_survivors[0]) if nfc_seeds[nfc_survivors[0]] < nfc_seeds[nfc_survivors[1]] else (nfc_survivors[0], nfc_survivors[1])
    ]
    
    div_winners = {}
    for away, home in afc_div_matchups:
        matchup = create_2025_matchup(away, home, team_df_2025, "DIV")
        game_feats = prepare_2025_game_features(matchup, team_df_2025, epa_model, baselines)
        pred = predict_ensemble_v16_1(model_dict, game_feats, platt_params, xgb_model, xgb_weight, lambda_base)
        winner = print_prediction(pred, game_feats, "Divisional")
        div_winners[("AFC", away, home)] = winner
    
    for away, home in nfc_div_matchups:
        matchup = create_2025_matchup(away, home, team_df_2025, "DIV")
        game_feats = prepare_2025_game_features(matchup, team_df_2025, epa_model, baselines)
        pred = predict_ensemble_v16_1(model_dict, game_feats, platt_params, xgb_model, xgb_weight, lambda_base)
        winner = print_prediction(pred, game_feats, "Divisional")
        div_winners[("NFC", away, home)] = winner
    
    # Conference Championships
    print("\n" + "-" * 40)
    print("CONFERENCE CHAMPIONSHIPS (Projected)")
    print("-" * 40)
    
    afc_div_winners = [div_winners[("AFC", afc_div_matchups[0][0], afc_div_matchups[0][1])],
                       div_winners[("AFC", afc_div_matchups[1][0], afc_div_matchups[1][1])]]
    nfc_div_winners = [div_winners[("NFC", nfc_div_matchups[0][0], nfc_div_matchups[0][1])],
                       div_winners[("NFC", nfc_div_matchups[1][0], nfc_div_matchups[1][1])]]
    
    afc_home = min(afc_div_winners, key=lambda x: afc_seeds[x])
    afc_away = max(afc_div_winners, key=lambda x: afc_seeds[x])
    nfc_home = min(nfc_div_winners, key=lambda x: nfc_seeds[x])
    nfc_away = max(nfc_div_winners, key=lambda x: nfc_seeds[x])
    
    conf_winners = {}
    for conf, away, home in [("AFC", afc_away, afc_home), ("NFC", nfc_away, nfc_home)]:
        matchup = create_2025_matchup(away, home, team_df_2025, "CON")
        game_feats = prepare_2025_game_features(matchup, team_df_2025, epa_model, baselines)
        pred = predict_ensemble_v16_1(model_dict, game_feats, platt_params, xgb_model, xgb_weight, lambda_base)
        winner = print_prediction(pred, game_feats, "Conference")
        conf_winners[conf] = winner
    
    # Super Bowl
    print("\n" + "-" * 40)
    print("SUPER BOWL (Projected)")
    print("-" * 40)
    
    sb_matchup = create_2025_matchup(conf_winners["AFC"], conf_winners["NFC"], team_df_2025, "SB", is_neutral=True)
    game_feats = prepare_2025_game_features(sb_matchup, team_df_2025, epa_model, baselines)
    pred = predict_ensemble_v16_1(model_dict, game_feats, platt_params, xgb_model, xgb_weight, lambda_base)
    sb_winner = print_prediction(pred, game_feats, "Super Bowl")
    
    print("\n" + "=" * 80)
    print(f"🏆 PREDICTED SUPER BOWL CHAMPION: {sb_winner}")
    print("=" * 80)
    
    return

if __name__ == "__main__":
    main()
