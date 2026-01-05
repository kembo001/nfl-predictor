"""
NFL 2025 Playoff Predictions - Full Bracket
Uses v16.2 model to predict:
- Wild Card round (with Vegas spreads)
- Divisional, Conference, Super Bowl (without spreads)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
import warnings
import re

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Note: XGBoost not installed. Using Ridge model only.")


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data():
    """Load historical data for model training"""
    games_df = pd.read_csv("../nfl_playoff_results_2000_2024_with_epa.csv")
    team_df = pd.read_csv("../df_with_upsets_merged.csv")
    spread_df = pd.read_csv("../csv/nfl_biggest_playoff_upsets_2000_2024.csv")
    
    print(f"Training data: {len(games_df)} playoff games, {len(team_df)} team-seasons")
    return games_df, team_df, spread_df


def load_2025_data():
    """Load 2025 playoff team stats and wild card schedule"""
    team_stats = pd.read_csv("../csv/2025/team_stats_2025_playoffs.csv")
    wild_card = pd.read_csv("../csv/2025/wild_card.csv")
    
    print(f"\n2025 Playoff Teams: {len(team_stats)}")
    print(f"Wild Card Games Scheduled: {len(wild_card)}")
    
    return team_stats, wild_card


def parse_spread(spread_str):
    """
    Parse spread string in various formats:
    - 'CHI (-1)' or 'HOU (3.5)' - parentheses format
    - 'LAR -10' or 'DEN +3' - space format
    
    Returns: (team_name, spread_value)
    
    Convention:
    - Negative spread = that team is favored (must win by that margin)
    - Positive spread = that team is underdog (can lose by that margin)
    """
    if pd.isna(spread_str) or not isinstance(spread_str, str):
        return None, None
    
    spread_str = spread_str.strip()
    
    # Try parentheses format first: "CHI (-1)" or "HOU (3.5)"
    match = re.match(r'(\w+)\s*\(([-+]?[\d.]+)\)', spread_str)
    if match:
        team = match.group(1)
        spread = float(match.group(2))
        return team, spread
    
    # Try space format: "LAR -10" or "DEN +3"
    match = re.match(r'(\w+)\s+([-+]?[\d.]+)', spread_str)
    if match:
        team = match.group(1)
        spread = float(match.group(2))
        return team, spread
    
    return None, None


# Team abbreviation mapping (common variations)
TEAM_ABBREV_MAP = {
    'LAR': 'LA',   # Los Angeles Rams
    'LV': 'LV',    # Las Vegas Raiders  
    'LAC': 'LAC',  # Los Angeles Chargers
    'WSH': 'WAS',  # Washington
    'JAC': 'JAX',  # Jacksonville
}

def normalize_team_name(team):
    """Normalize team abbreviation to match team_stats format"""
    return TEAM_ABBREV_MAP.get(team, team)


def parse_wild_card_games(wild_card_df, team_stats):
    """Convert wild card schedule to game format with proper spreads"""
    games = []
    valid_teams = set(team_stats['team'].values)
    
    for _, row in wild_card_df.iterrows():
        matchup = row['matchup']
        if pd.isna(matchup):
            continue
            
        # Parse "AWAY @ HOME"
        parts = matchup.split(' @ ')
        if len(parts) != 2:
            continue
            
        away_team = normalize_team_name(parts[0].strip())
        home_team = normalize_team_name(parts[1].strip())
        
        # Validate teams exist in our data
        if away_team not in valid_teams:
            print(f"  Warning: Team '{away_team}' not found in team stats")
            continue
        if home_team not in valid_teams:
            print(f"  Warning: Team '{home_team}' not found in team stats")
            continue
        
        # Get spread info
        # Format: "TEAM (spread)" where negative = favored, positive = underdog
        spread_team_raw, spread_val = parse_spread(row['spread'])
        spread_team = normalize_team_name(spread_team_raw) if spread_team_raw else None
        
        # Convert to home_spread (negative = home favored)
        # Example: "CHI (-1)" means CHI favored by 1
        # Example: "HOU (3.5)" means HOU is underdog by 3.5 (opponent favored by 3.5)
        if spread_team is not None:
            if spread_val < 0:  # spread_team is the favorite
                if spread_team == home_team:
                    home_spread = spread_val  # home favored
                else:
                    home_spread = -spread_val  # away favored, flip for home perspective
            else:  # spread_val >= 0, spread_team is the underdog
                if spread_team == home_team:
                    home_spread = spread_val  # home is underdog (positive spread)
                else:
                    home_spread = -spread_val  # away is underdog, home is favored
        else:
            home_spread = np.nan
        
        # Get seeds from team stats
        away_data = team_stats[team_stats['team'] == away_team]
        home_data = team_stats[team_stats['team'] == home_team]
        
        if len(away_data) == 0 or len(home_data) == 0:
            print(f"  Warning: Missing team data for {matchup}")
            continue
        
        games.append({
            'away_team': away_team,
            'home_team': home_team,
            'away_seed': int(away_data.iloc[0]['playoff_seed']),
            'home_seed': int(home_data.iloc[0]['playoff_seed']),
            'home_spread': home_spread,
            'game_type': 'WC',
            'conference': away_data.iloc[0]['conference'],
            'gameday': row.get('gameday', ''),
        })
    
    return games


# ============================================================
# MODEL COMPONENTS (from v16.2)
# ============================================================

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
    return games_df


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


def compute_season_zscores(team_df):
    team_df = team_df.copy()
    
    higher_is_better = [
        "passing_epa", "rushing_epa", "total_offensive_epa", "net_epa",
        "point_differential", "win_pct", "pressure_rate", "sack_rate", "protection_rate",
    ]
    lower_is_better = [
        "defensive_epa", "defensive_pass_epa", "defensive_rush_epa", "sacks_allowed_rate"
    ]

    for stat in higher_is_better:
        if stat in team_df.columns:
            mean_val = team_df[stat].mean()
            std_val = team_df[stat].std() + 1e-9
            team_df[f"z_{stat}"] = (team_df[stat] - mean_val) / std_val

    for stat in lower_is_better:
        if stat in team_df.columns:
            mean_val = team_df[stat].mean()
            std_val = team_df[stat].std() + 1e-9
            team_df[f"z_{stat}"] = -(team_df[stat] - mean_val) / std_val

    return team_df


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

    features["delta_pass_edge"] = (home_pass_off - away_pass_def) - (away_pass_off - home_pass_def)
    features["delta_rush_edge"] = (home_rush_off - away_rush_def) - (away_rush_off - home_rush_def)

    home_ol = get_z(home_data, "protection_rate")
    away_ol = get_z(away_data, "protection_rate")
    home_dl = get_z(home_data, "pressure_rate")
    away_dl = get_z(away_data, "pressure_rate")

    features["home_ol_exposure"] = float(np.maximum(0, away_dl - home_ol))
    features["away_ol_exposure"] = float(np.maximum(0, home_dl - away_ol))
    features["delta_ol_exposure"] = features["away_ol_exposure"] - features["home_ol_exposure"]
    features["total_ol_exposure"] = features["home_ol_exposure"] + features["away_ol_exposure"]

    home_pass_d_exp = float(np.maximum(0, away_pass_off - home_pass_def))
    away_pass_d_exp = float(np.maximum(0, home_pass_off - away_pass_def))
    features["home_pass_d_exposure"] = home_pass_d_exp
    features["away_pass_d_exposure"] = away_pass_d_exp
    features["delta_pass_d_exposure"] = away_pass_d_exp - home_pass_d_exp
    features["total_pass_d_exposure"] = home_pass_d_exp + away_pass_d_exp
    features["delta_rush_d_exposure"] = float(
        np.maximum(0, home_rush_off - away_rush_def) - np.maximum(0, away_rush_off - home_rush_def)
    )

    features["home_ol_z"] = home_ol
    features["away_ol_z"] = away_ol
    features["home_dl_z"] = home_dl
    features["away_dl_z"] = away_dl

    return features


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


def create_tossup_features(seed_diff, home_spread):
    abs_seed_diff = abs(int(seed_diff))
    has_spread = pd.notna(home_spread)
    close_spread = (abs(float(home_spread)) <= 3.5) if has_spread else True
    is_tossup = 1 if (abs_seed_diff <= 1 and close_spread) else 0

    return {
        "is_tossup": int(is_tossup),
        "is_big_favorite": 1 if abs_seed_diff >= 4 else 0,
        "is_medium_gap": 1 if abs_seed_diff in (2, 3) else 0,
        "is_close_seeds": 1 if abs_seed_diff <= 1 else 0,
        "tossup_baseline": 1 if ((not has_spread) and (abs_seed_diff <= 1)) else 0,
    }


def fit_spread_model(features_df, train_seasons, baselines, alpha=0.5):
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
    slope_per_point = np.clip(slope_td_hat / 7.0, -0.12, -0.04)

    prior_intercept = float(logit(np.clip(baselines["home_win_rate"], 0.05, 0.95)))
    w = float(n / (n + 60))

    return {
        "intercept": w * intercept_hat + (1 - w) * prior_intercept,
        "slope_per_point": w * slope_per_point + (1 - w) * (-0.073),
    }


def get_spread_offset_logit(home_spread, spread_model):
    if spread_model is None or pd.isna(home_spread):
        return np.nan
    offset = spread_model["intercept"] + spread_model["slope_per_point"] * float(home_spread)
    return float(np.clip(offset, -4, 4))


# ============================================================
# FEATURE PREPARATION FOR 2025 GAMES
# ============================================================

def prepare_2025_game_features(game, team_stats_z, epa_model, baselines, spread_model):
    """Prepare features for a single 2025 playoff game"""
    
    away_team = game['away_team']
    home_team = game['home_team']
    is_neutral = game.get('is_neutral', False)
    home_spread = game.get('home_spread', np.nan)
    
    away_data = team_stats_z[team_stats_z['team'] == away_team].iloc[0]
    home_data = team_stats_z[team_stats_z['team'] == home_team].iloc[0]
    
    away_seed = int(away_data['playoff_seed'])
    home_seed = int(home_data['playoff_seed'])
    
    # For neutral site (Super Bowl), reorder so lower seed is "home"
    if is_neutral and away_seed < home_seed:
        away_team, home_team = home_team, away_team
        away_data, home_data = home_data, away_data
        away_seed, home_seed = home_seed, away_seed
        if pd.notna(home_spread):
            home_spread = -float(home_spread)
    
    # Basic stats
    away_net_epa = float(away_data.get("net_epa", 0))
    home_net_epa = float(home_data.get("net_epa", 0))
    away_games = float(away_data["wins"] + away_data["losses"])
    home_games = float(home_data["wins"] + home_data["losses"])
    away_pd_pg = float(away_data["point_differential"] / max(away_games, 1))
    home_pd_pg = float(home_data["point_differential"] / max(home_games, 1))
    
    # Expected win pct from EPA
    away_exp = float(epa_model["intercept"] + epa_model["slope"] * away_net_epa)
    home_exp = float(epa_model["intercept"] + epa_model["slope"] * home_net_epa)
    away_win_pct = float(away_data["win_pct"])
    home_win_pct = float(home_data["win_pct"])
    
    # Quality rank
    team_stats_z_copy = team_stats_z.copy()
    team_stats_z_copy["pd_pg"] = team_stats_z_copy["point_differential"] / (
        (team_stats_z_copy["wins"] + team_stats_z_copy["losses"]).clip(lower=1)
    )
    team_stats_z_copy["qrank"] = team_stats_z_copy["pd_pg"].rank(ascending=False)
    
    away_q = team_stats_z_copy[team_stats_z_copy["team"] == away_team]["qrank"].values
    home_q = team_stats_z_copy[team_stats_z_copy["team"] == home_team]["qrank"].values
    away_quality_rank = int(away_q[0]) if len(away_q) else 7
    home_quality_rank = int(home_q[0]) if len(home_q) else 7
    
    # Matchup features
    matchup = compute_matchup_features(away_data, home_data)
    
    # Baseline and spread
    baseline_prob = float(baseline_probability(home_seed, away_seed, is_neutral, baselines))
    baseline_logit = float(logit(np.clip(baseline_prob, 0.01, 0.99)))
    spread_offset = get_spread_offset_logit(home_spread, spread_model)
    
    if pd.notna(spread_offset):
        offset_logit = float(spread_offset)
        offset_source = "spread"
    else:
        offset_logit = float(baseline_logit)
        offset_source = "baseline"
    
    seed_diff = away_seed - home_seed
    
    features = {
        "away_team": away_team,
        "home_team": home_team,
        "away_seed": away_seed,
        "home_seed": home_seed,
        "is_neutral": 1 if is_neutral else 0,
        "seed_diff": seed_diff,
        "seed_diff_sq": seed_diff ** 2,
        "seed_diff_abs": abs(seed_diff),
        
        "delta_net_epa": home_net_epa - away_net_epa,
        "delta_pd_pg": home_pd_pg - away_pd_pg,
        "delta_vulnerability": (away_win_pct - away_exp) - (home_win_pct - home_exp),
        "delta_underseeded": (away_seed - away_quality_rank) - (home_seed - home_quality_rank),
        "delta_momentum": 0.0,  # Not available for 2025
        
        "baseline_prob": baseline_prob,
        "baseline_logit": baseline_logit,
        "home_spread": home_spread,
        "spread_offset": spread_offset,
        "offset_logit": offset_logit,
        "offset_source": offset_source,
        
        "is_superbowl": 1 if game.get('game_type') == 'SB' else 0,
    }
    
    features.update(matchup)
    features.update(create_spread_features(home_spread))
    features.update(create_tossup_features(seed_diff, home_spread))
    
    # Interaction features
    features["pass_x_ol"] = float(features["delta_pass_edge"] * features["delta_ol_exposure"])
    features["seed_x_epa"] = float(features["seed_diff"] * features["delta_net_epa"])
    
    return features


# ============================================================
# PREDICTION
# ============================================================

def predict_game(game_features, baselines, adj_cap=0.80, tossup_shrink=0.35, 
                 epa_weight=0.35, spread_weight=0.65):
    """
    Predict a single game using blended model.
    
    For games WITH spreads:
    - Blend spread probability with EPA-based adjustments
    - spread_weight controls how much to trust Vegas (default 65%)
    - epa_weight controls EPA influence (default 35%)
    
    For games WITHOUT spreads:
    - Use baseline + EPA adjustments
    """
    
    src = game_features["offset_source"]
    offset_logit = float(game_features["offset_logit"])
    home_spread = game_features.get("home_spread", np.nan)
    baseline_logit = float(game_features["baseline_logit"])
    
    # EPA-based adjustments (always computed)
    delta_epa = float(game_features["delta_net_epa"])
    delta_pd = float(game_features["delta_pd_pg"])
    
    # Scale EPA adjustment - stronger signal
    # Typical EPA range is -100 to +200, so 0.005 per EPA point
    epa_adj = delta_epa * 0.005
    pd_adj = delta_pd * 0.012
    
    # Cap the adjustments
    total_epa_adj = np.clip(epa_adj + pd_adj, -adj_cap, adj_cap)
    
    # EPA-only probability (from baseline)
    epa_logit = baseline_logit + total_epa_adj
    epa_prob = float(expit(np.clip(epa_logit, -4, 4)))
    
    if src == "spread":
        # BLENDED APPROACH: Combine spread with EPA
        spread_prob = float(expit(np.clip(offset_logit, -4, 4)))
        
        # Check for EPA disagreement with spread
        epa_disagrees = (spread_prob > 0.5 and epa_prob < 0.5) or (spread_prob < 0.5 and epa_prob > 0.5)
        
        # If EPA strongly disagrees, increase EPA weight
        if epa_disagrees and abs(delta_epa) > 100:
            # Strong EPA disagreement - boost EPA influence
            effective_epa_weight = min(0.50, epa_weight + 0.15)
            effective_spread_weight = 1.0 - effective_epa_weight
        else:
            effective_epa_weight = epa_weight
            effective_spread_weight = spread_weight
        
        # Blend the probabilities
        home_prob = (effective_spread_weight * spread_prob) + (effective_epa_weight * epa_prob)
        
        # Store for output
        blend_info = {
            "spread_prob": spread_prob,
            "epa_prob": epa_prob,
            "spread_weight": effective_spread_weight,
            "epa_weight": effective_epa_weight,
            "epa_disagrees": epa_disagrees,
        }
    else:
        # BASELINE-ONLY: Use EPA adjustments on baseline
        home_prob = epa_prob
        
        # Tossup shrink for close seed matchups
        if int(game_features.get("is_tossup", 0)) == 1:
            base_prob = float(expit(baseline_logit))
            home_prob = (1 - tossup_shrink) * home_prob + tossup_shrink * base_prob
        
        blend_info = {
            "spread_prob": None,
            "epa_prob": epa_prob,
            "spread_weight": 0.0,
            "epa_weight": 1.0,
            "epa_disagrees": False,
        }
    
    home_prob = float(np.clip(home_prob, 0.01, 0.99))
    
    predicted_winner = game_features["home_team"] if home_prob > 0.5 else game_features["away_team"]
    predicted_loser = game_features["away_team"] if home_prob > 0.5 else game_features["home_team"]
    win_prob = home_prob if home_prob > 0.5 else (1 - home_prob)
    
    # Determine if this is an upset prediction
    home_seed = int(game_features["home_seed"])
    away_seed = int(game_features["away_seed"])
    
    # An upset is when higher seed (worse team) beats lower seed (better team)
    # Or when underdog (positive spread) is predicted to win
    is_upset = False
    upset_type = None
    
    if predicted_winner == game_features["home_team"]:
        winner_seed = home_seed
        loser_seed = away_seed
    else:
        winner_seed = away_seed
        loser_seed = home_seed
    
    # Seed-based upset (higher seed number wins)
    if winner_seed > loser_seed:
        is_upset = True
        upset_type = "seed"
    
    # Spread-based upset (underdog wins)
    if pd.notna(home_spread):
        if home_spread > 0 and predicted_winner == game_features["home_team"]:
            # Home was underdog but predicted to win
            is_upset = True
            upset_type = "spread"
        elif home_spread < 0 and predicted_winner == game_features["away_team"]:
            # Away was underdog but predicted to win
            is_upset = True
            upset_type = "spread"
    
    return {
        "home_prob": home_prob,
        "predicted_winner": predicted_winner,
        "predicted_loser": predicted_loser,
        "win_probability": win_prob,
        "offset_source": src,
        "is_upset": is_upset,
        "upset_type": upset_type,
        "winner_seed": winner_seed,
        "loser_seed": loser_seed,
        "delta_epa": delta_epa,
        "blend_info": blend_info,
    }


# ============================================================
# BRACKET SIMULATION
# ============================================================

def get_divisional_matchups(wc_winners, team_stats):
    """
    Generate Divisional round matchups based on Wild Card winners.
    NFL rules: #1 seed plays lowest remaining seed, remaining two seeds play each other.
    Lower seed always hosts.
    """
    matchups = []
    
    for conf in ['AFC', 'NFC']:
        # Get #1 seed (bye team)
        bye_team_row = team_stats[(team_stats['conference'] == conf) & (team_stats['playoff_seed'] == 1)]
        if len(bye_team_row) == 0:
            continue
        bye_team = bye_team_row['team'].values[0]
        
        # Get WC winners from this conference
        conf_winners = []
        for w in wc_winners:
            team_row = team_stats[team_stats['team'] == w]
            if len(team_row) > 0 and team_row['conference'].values[0] == conf:
                conf_winners.append(w)
        
        if len(conf_winners) < 3:
            print(f"  Warning: Only {len(conf_winners)} winners found for {conf}")
            continue
        
        # Get seeds for winners
        winner_seeds = []
        for w in conf_winners:
            seed = int(team_stats[team_stats['team'] == w]['playoff_seed'].values[0])
            winner_seeds.append((w, seed))
        
        # Sort by seed (lowest/best first)
        winner_seeds.sort(key=lambda x: x[1])
        
        # #1 plays lowest remaining seed (highest seed number = worst remaining team)
        worst_remaining = winner_seeds[-1]  # Highest seed number
        
        # Other two play each other
        other_two = winner_seeds[:-1]  # The two better seeds
        
        # Game 1: #1 hosts worst remaining
        matchups.append({
            'away_team': worst_remaining[0],
            'home_team': bye_team,
            'game_type': 'DIV',
            'conference': conf,
            'is_neutral': False,
        })
        
        # Game 2: Better seed hosts between remaining two
        if len(other_two) >= 2:
            if other_two[0][1] < other_two[1][1]:
                home, away = other_two[0][0], other_two[1][0]
            else:
                home, away = other_two[1][0], other_two[0][0]
            
            matchups.append({
                'away_team': away,
                'home_team': home,
                'game_type': 'DIV',
                'conference': conf,
                'is_neutral': False,
            })
    
    return matchups


def get_conference_matchups(div_winners, team_stats):
    """Generate Conference Championship matchups"""
    matchups = []
    
    for conf in ['AFC', 'NFC']:
        conf_winners = [w for w in div_winners if team_stats[team_stats['team'] == w]['conference'].values[0] == conf]
        
        if len(conf_winners) != 2:
            continue
        
        # Get seeds
        seeds = []
        for w in conf_winners:
            seed = int(team_stats[team_stats['team'] == w]['playoff_seed'].values[0])
            seeds.append((w, seed))
        
        seeds.sort(key=lambda x: x[1])
        
        # Lower seed hosts
        matchups.append({
            'away_team': seeds[1][0],
            'home_team': seeds[0][0],
            'game_type': 'CON',
            'conference': conf,
            'is_neutral': False,
        })
    
    return matchups


def get_superbowl_matchup(conf_winners, team_stats):
    """Generate Super Bowl matchup"""
    afc_champ = None
    nfc_champ = None
    
    for w in conf_winners:
        conf = team_stats[team_stats['team'] == w]['conference'].values[0]
        if conf == 'AFC':
            afc_champ = w
        else:
            nfc_champ = w
    
    if afc_champ and nfc_champ:
        return [{
            'away_team': afc_champ,
            'home_team': nfc_champ,
            'game_type': 'SB',
            'conference': 'SB',
            'is_neutral': True,
        }]
    return []


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("NFL 2025 PLAYOFF PREDICTIONS")
    print("Model v16.2 - Full Bracket Simulation")
    print("=" * 80)
    
    # Load training data
    games_df, team_df, spread_df = load_training_data()
    games_df = merge_spread_data_safe(games_df, spread_df)
    
    # Load 2025 data
    team_stats, wild_card_df = load_2025_data()
    
    # Compute z-scores for 2025 teams
    team_stats_z = compute_season_zscores(team_stats)
    
    # Train model components
    train_seasons = list(range(2000, 2024))
    baselines = compute_historical_baselines(games_df, max_season=2023)
    epa_model = fit_expected_win_pct_model(team_df, max_season=2023)
    
    # Prepare historical features for spread model
    temp_features = []
    for _, game in games_df.iterrows():
        season = int(game["season"])
        if season > 2023:
            continue
        home_wins = 1 if game["winner"] == game["home_team"] else 0
        temp_features.append({
            "season": season,
            "home_spread": game.get("home_spread", np.nan),
            "home_wins": home_wins,
        })
    temp_df = pd.DataFrame(temp_features)
    spread_model = fit_spread_model(temp_df, train_seasons, baselines, alpha=0.5)
    
    print(f"\nHistorical home win rate: {baselines['home_win_rate']:.1%}")
    if spread_model:
        print(f"Spread model: logit(P) = {spread_model['intercept']:.3f} + {spread_model['slope_per_point']:.4f} * spread")
    
    print("\n" + "-" * 60)
    print("MODEL METHODOLOGY")
    print("-" * 60)
    print("• Wild Card (with spreads): 65% Vegas + 35% EPA blend")
    print("• If EPA strongly disagrees (>±100): EPA weight boosted to 50%")
    print("• Later rounds (no spreads): Baseline + EPA adjustments")
    print("• EPA adjustment: +0.5% win prob per +1 EPA differential")
    print("-" * 60)
    
    # =========================================
    # WILD CARD ROUND (with spreads)
    # =========================================
    print("\n" + "=" * 80)
    print("WILD CARD ROUND - January 10-12, 2026")
    print("(Using Vegas spreads)")
    print("=" * 80)
    
    wc_games = parse_wild_card_games(wild_card_df, team_stats)
    
    print(f"\nGames loaded: {len(wc_games)}")
    
    # Check for expected number of WC games (should be 6)
    if len(wc_games) != 6:
        print(f"\n⚠️  WARNING: Expected 6 Wild Card games, but only found {len(wc_games)}")
        print("   Please verify wild_card.csv contains all matchups")
    
    wc_winners = []
    wc_upsets = []
    
    print(f"\n{'Matchup':<14} {'Spread':<12} {'EPA Δ':>8} {'Sprd%':>6} {'EPA%':>6} {'Final':>6} {'Pred':<5} {'Alert':<10}")
    print("-" * 85)
    
    for game in wc_games:
        features = prepare_2025_game_features(game, team_stats_z, epa_model, baselines, spread_model)
        pred = predict_game(features, baselines)
        
        wc_winners.append(pred['predicted_winner'])
        
        # Format spread string
        if pd.notna(game['home_spread']):
            if game['home_spread'] < 0:
                spread_str = f"{game['home_team']} {game['home_spread']:.1f}"
            else:
                spread_str = f"{game['away_team']} {-game['home_spread']:.1f}"
        else:
            spread_str = "No line"
        
        matchup = f"{game['away_team']}@{game['home_team']}"
        
        # EPA differential (positive = home advantage)
        epa_diff = pred['delta_epa']
        epa_str = f"{epa_diff:+.0f}"
        
        # Blend info
        blend = pred['blend_info']
        if blend['spread_prob'] is not None:
            spread_pct = f"{blend['spread_prob']*100:.0f}%"
            epa_pct = f"{blend['epa_prob']*100:.0f}%"
        else:
            spread_pct = "--"
            epa_pct = f"{blend['epa_prob']*100:.0f}%"
        
        final_pct = f"{pred['home_prob']*100:.0f}%"
        
        # Upset alert
        alert = ""
        if pred['is_upset']:
            alert = "⚠️ UPSET!"
            wc_upsets.append({
                'matchup': matchup,
                'winner': pred['predicted_winner'],
                'loser': pred['predicted_loser'],
                'prob': pred['win_probability'],
                'type': pred['upset_type'],
                'winner_seed': pred['winner_seed'],
                'loser_seed': pred['loser_seed'],
                'epa_disagrees': blend.get('epa_disagrees', False),
            })
        elif blend.get('epa_disagrees'):
            alert = "🟡 EPA conflict"
        
        print(f"{matchup:<14} {spread_str:<12} {epa_str:>8} {spread_pct:>6} {epa_pct:>6} {final_pct:>6} {pred['predicted_winner']:<5} {alert}")
    
    # Legend
    print("\n  Legend: Spread% = Vegas implied | EPA% = EPA model | Final = Blended (65% spread + 35% EPA)")
    
    # Print upset alerts summary
    if wc_upsets:
        print("\n" + "!" * 70)
        print("🚨 WILD CARD UPSET ALERTS 🚨")
        print("!" * 70)
        for upset in wc_upsets:
            print(f"\n  ⚠️  {upset['winner']} (#{upset['winner_seed']}) over {upset['loser']} (#{upset['loser_seed']})")
            print(f"      Win probability: {upset['prob']:.1%}")
            print(f"      Upset type: {upset['type']}")
            if upset.get('epa_disagrees'):
                print(f"      🟡 Note: EPA model disagrees with Vegas spread")
    
    print(f"\nWild Card Winners: {', '.join(wc_winners)}")
    
    all_upsets = wc_upsets.copy()
    
    # =========================================
    # DIVISIONAL ROUND (no spreads)
    # =========================================
    print("\n" + "=" * 80)
    print("DIVISIONAL ROUND")
    print("(No Vegas spreads - using baseline + EPA model)")
    print("=" * 80)
    
    div_matchups = get_divisional_matchups(wc_winners, team_stats)
    div_winners = []
    div_upsets = []
    
    print(f"\n{'Matchup':<14} {'Seeds':<10} {'EPA Δ':>8} {'Home%':>7} {'Pred':<5} {'Prob':>6} {'Alert':<10}")
    print("-" * 70)
    
    for game in div_matchups:
        features = prepare_2025_game_features(game, team_stats_z, epa_model, baselines, spread_model)
        pred = predict_game(features, baselines)
        
        div_winners.append(pred['predicted_winner'])
        
        matchup = f"{game['away_team']}@{game['home_team']}"
        seeds = f"#{features['away_seed']}@#{features['home_seed']}"
        epa_str = f"{pred['delta_epa']:+.0f}"
        home_pct = f"{pred['home_prob']*100:.0f}%"
        
        # Upset alert
        alert = ""
        if pred['is_upset']:
            alert = "⚠️ UPSET!"
            div_upsets.append({
                'matchup': matchup,
                'winner': pred['predicted_winner'],
                'loser': pred['predicted_loser'],
                'prob': pred['win_probability'],
                'winner_seed': pred['winner_seed'],
                'loser_seed': pred['loser_seed'],
                'delta_epa': pred['delta_epa'],
            })
        
        print(f"{matchup:<14} {seeds:<10} {epa_str:>8} {home_pct:>7} {pred['predicted_winner']:<5} {pred['win_probability']:>5.1%} {alert}")
    
    # Print upset alerts summary
    if div_upsets:
        print("\n" + "!" * 55)
        print("🚨 DIVISIONAL UPSET ALERTS 🚨")
        print("!" * 55)
        for upset in div_upsets:
            print(f"\n  ⚠️  {upset['winner']} (#{upset['winner_seed']}) over {upset['loser']} (#{upset['loser_seed']})")
            print(f"      Win probability: {upset['prob']:.1%}")
            print(f"      EPA differential: {upset.get('delta_epa', 0):+.0f}")
        all_upsets.extend(div_upsets)
    
    print(f"\nDivisional Winners: {', '.join(div_winners)}")
    
    # =========================================
    # CONFERENCE CHAMPIONSHIPS (no spreads)
    # =========================================
    print("\n" + "=" * 80)
    print("CONFERENCE CHAMPIONSHIPS")
    print("(No Vegas spreads - using baseline + EPA model)")
    print("=" * 80)
    
    conf_matchups = get_conference_matchups(div_winners, team_stats)
    conf_winners = []
    conf_upsets = []
    
    for game in conf_matchups:
        features = prepare_2025_game_features(game, team_stats_z, epa_model, baselines, spread_model)
        pred = predict_game(features, baselines)
        
        conf_winners.append(pred['predicted_winner'])
        
        conf_name = "AFC Championship" if game['conference'] == 'AFC' else "NFC Championship"
        matchup = f"{game['away_team']} @ {game['home_team']}"
        seeds = f"#{features['away_seed']} vs #{features['home_seed']}"
        
        # Upset alert
        alert = ""
        if pred['is_upset']:
            alert = " ⚠️ UPSET!"
            conf_upsets.append({
                'game': conf_name,
                'winner': pred['predicted_winner'],
                'loser': pred['predicted_loser'],
                'prob': pred['win_probability'],
                'winner_seed': pred['winner_seed'],
                'loser_seed': pred['loser_seed'],
                'delta_epa': pred['delta_epa'],
            })
        
        print(f"\n{conf_name}")
        print(f"  {matchup} ({seeds})")
        print(f"  EPA Δ: {pred['delta_epa']:+.0f} (home advantage)")
        print(f"  Winner: {pred['predicted_winner']} ({pred['win_probability']:.1%}){alert}")
    
    # Print upset alerts summary
    if conf_upsets:
        print("\n" + "!" * 55)
        print("🚨 CONFERENCE CHAMPIONSHIP UPSET ALERTS 🚨")
        print("!" * 55)
        for upset in conf_upsets:
            print(f"\n  ⚠️  {upset['game']}: {upset['winner']} (#{upset['winner_seed']}) over {upset['loser']} (#{upset['loser_seed']})")
            print(f"      Win probability: {upset['prob']:.1%}")
        all_upsets.extend(conf_upsets)
    
    print(f"\nConference Champions: {', '.join(conf_winners)}")
    
    # =========================================
    # SUPER BOWL (no spreads, neutral site)
    # =========================================
    print("\n" + "=" * 80)
    print("SUPER BOWL LX")
    print("(Neutral site - No Vegas spread)")
    print("=" * 80)
    
    sb_matchup = get_superbowl_matchup(conf_winners, team_stats)
    sb_pred = None
    
    if sb_matchup:
        game = sb_matchup[0]
        features = prepare_2025_game_features(game, team_stats_z, epa_model, baselines, spread_model)
        sb_pred = predict_game(features, baselines)
        
        afc_team = game['away_team']
        nfc_team = game['home_team']
        
        # Determine which team is AFC/NFC
        afc_conf = team_stats[team_stats['team'] == afc_team]['conference'].values[0]
        if afc_conf != 'AFC':
            afc_team, nfc_team = nfc_team, afc_team
        
        # Check for upset
        alert = ""
        if sb_pred['is_upset']:
            alert = " ⚠️ UPSET!"
            all_upsets.append({
                'game': 'Super Bowl',
                'winner': sb_pred['predicted_winner'],
                'loser': sb_pred['predicted_loser'],
                'prob': sb_pred['win_probability'],
                'winner_seed': sb_pred['winner_seed'],
                'loser_seed': sb_pred['loser_seed'],
            })
        
        print(f"\n{afc_team} (AFC) vs {nfc_team} (NFC)")
        print(f"Seeds: #{features['away_seed']} vs #{features['home_seed']}")
        print(f"EPA Δ: {sb_pred['delta_epa']:+.0f}")
        print(f"\n🏆 PREDICTED SUPER BOWL CHAMPION: {sb_pred['predicted_winner']}{alert}")
        print(f"   Win Probability: {sb_pred['win_probability']:.1%}")
    
    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 80)
    print("FULL BRACKET SUMMARY")
    print("=" * 80)
    print(f"\nWild Card Winners:     {', '.join(wc_winners)}")
    print(f"Divisional Winners:    {', '.join(div_winners)}")
    print(f"Conference Champions:  {', '.join(conf_winners)}")
    if sb_matchup:
        print(f"\n🏆 SUPER BOWL CHAMPION: {sb_pred['predicted_winner']}")
    
    # Overall upset summary
    if all_upsets:
        print("\n" + "=" * 80)
        print(f"🚨 TOTAL UPSET PREDICTIONS: {len(all_upsets)} 🚨")
        print("=" * 80)
        for i, upset in enumerate(all_upsets, 1):
            winner = upset.get('winner', upset.get('predicted_winner'))
            loser = upset.get('loser', upset.get('predicted_loser'))
            print(f"  {i}. {winner} (#{upset['winner_seed']}) over {loser} (#{upset['loser_seed']}) - {upset['prob']:.1%}")
    else:
        print("\nNo upsets predicted - chalk bracket!")
    
    return {
        'wild_card_winners': wc_winners,
        'divisional_winners': div_winners,
        'conference_winners': conf_winners,
        'super_bowl_winner': sb_pred['predicted_winner'] if sb_matchup else None,
        'upsets': all_upsets,
    }


if __name__ == "__main__":
    results = main()
