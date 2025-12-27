"""
NFL Playoff Model v9 - Stabilized + Matchup Features

KEY FIXES FROM v8:
1. Standardize features + Ridge regularization (fix GLM instability)
2. Actually use calibration season (Platt scaling on 2023)
3. Remove is_neutral from residual model (handle in baseline only)
4. Fix spread merge (use team pairs, not winner/loser)
5. Learn spread→prob from data instead of hardcoding
6. Use spread as offset when available (market prior > seed prior)
7. Include 2023 for baselines (no leakage for 2024)
8. Use delta_underseeded instead of just away_underseeded

NEW MATCHUP FEATURES:
1. Pass/Rush edges (unit-vs-unit matchup advantages)
2. Pressure mismatch (OL weakness vs DL strength)
3. Style skew (who gets to play their preferred game)
4. Weakness exposure flags (quartile interactions)
5. All features z-scored by season for stability

MODELING APPROACH:
- Offset GLM with ridge regularization
- Offset = spread_logit when available, else baseline_logit
- Platt scaling calibration on holdout season
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit, logit
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DATA LOADING
# ============================================================

def load_all_data():
    """Load all required datasets"""
    games_df = pd.read_csv('../nfl_playoff_results_2000_2024_with_epa.csv')
    team_df = pd.read_csv('../df_with_upsets_merged.csv')
    spread_df = pd.read_csv('../csv/nfl_biggest_playoff_upsets_2000_2024.csv')
    
    print(f"Loaded {len(games_df)} playoff games")
    print(f"Loaded {len(team_df)} team-season records")
    print(f"Loaded {len(spread_df)} games with spread data")
    
    return games_df, team_df, spread_df


def merge_spread_data_fixed(games_df, spread_df):
    """
    Merge spread data using team pairs (not winner/loser) to avoid sign issues.
    Spread is typically from home team perspective (negative = home favored).
    """
    games_df = games_df.copy()
    spread_df = spread_df.copy()
    
    # Build lookup using both team orderings
    # The spread file has: underdog, winner, loser, spread_line
    # We need to figure out who was home and assign spread correctly
    
    spread_lookup = {}
    for _, row in spread_df.iterrows():
        season = row['season']
        game_type = row['game_type']
        spread_mag = row['spread_magnitude']
        
        # The underdog is the team getting points
        # If spread_line is positive, home was underdog
        # If spread_line is negative, away was underdog
        spread_line = row['spread_line']
        
        # Find the matching game in games_df to get home/away
        matching = games_df[
            (games_df['season'] == season) & 
            (games_df['game_type'] == game_type) &
            (((games_df['home_team'] == row['winner']) & (games_df['away_team'] == row['loser'])) |
             ((games_df['home_team'] == row['loser']) & (games_df['away_team'] == row['winner'])))
        ]
        
        if len(matching) > 0:
            game = matching.iloc[0]
            # Create key based on actual home/away from games_df
            key = (season, game_type, game['away_team'], game['home_team'])
            
            # Determine spread from home perspective
            # spread_line from the file is from favorite perspective
            # We need to convert: negative spread means favorite is giving points
            if game['home_team'] == row['winner']:
                # Home team won
                if spread_line < 0:
                    # Home was favored (spread_line is negative for favorites)
                    home_spread = spread_line  # Keep as is
                else:
                    # Away was favored, home was underdog
                    home_spread = spread_line  # Positive = home getting points
            else:
                # Away team won
                if spread_line < 0:
                    # Winner (away) was favored
                    home_spread = -spread_line  # Flip: home was underdog
                else:
                    # Loser (home) was favored
                    home_spread = -spread_line  # Home was favored but with negative convention
            
            spread_lookup[key] = spread_mag  # Use magnitude for now, sign determined by context
    
    # Actually, let's simplify: use the spread_line directly and be careful about sign
    # Rebuild with cleaner logic
    spread_lookup = {}
    for _, row in spread_df.iterrows():
        season = row['season']
        game_type = row['game_type']
        
        # Find matching game
        matching = games_df[
            (games_df['season'] == season) & 
            (games_df['game_type'] == game_type) &
            (((games_df['home_team'] == row['winner']) & (games_df['away_team'] == row['loser'])) |
             ((games_df['home_team'] == row['loser']) & (games_df['away_team'] == row['winner'])))
        ]
        
        if len(matching) > 0:
            game = matching.iloc[0]
            key = (season, game_type, game['away_team'], game['home_team'])
            # spread_line in file: negative means that team was favored
            # We'll store the raw value and interpret based on who's listed
            spread_lookup[key] = row['spread_line']
    
    def get_spread(row):
        key = (row['season'], row['game_type'], row['away_team'], row['home_team'])
        return spread_lookup.get(key, np.nan)
    
    games_df['spread_line'] = games_df.apply(get_spread, axis=1)
    
    matched = games_df['spread_line'].notna().sum()
    print(f"Matched spread data for {matched}/{len(games_df)} games")
    
    return games_df


# ============================================================
# LEARNED SPREAD -> PROBABILITY MODEL
# ============================================================

def fit_spread_probability_model(features_df, train_seasons):
    """
    Learn spread -> win probability from historical data.
    This avoids hardcoded formulas and ensures correct orientation.
    """
    train_df = features_df[
        (features_df['season'].isin(train_seasons)) & 
        (features_df['spread_line'].notna())
    ].copy()
    
    if len(train_df) < 20:
        print(f"Warning: Only {len(train_df)} games with spread for training")
        return None
    
    X = sm.add_constant(train_df['spread_line'].values)
    y = train_df['home_wins'].values
    
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        print(f"\nSpread -> Prob Model (trained on {len(train_df)} games):")
        print(f"  P(home wins) = logit^-1({model.params[0]:.3f} + {model.params[1]:.3f} * spread_line)")
        return model
    except Exception as e:
        print(f"Spread model fit failed: {e}")
        return None


def spread_to_probability_learned(spread_line, spread_model):
    """Convert spread to probability using learned model"""
    if spread_model is None or pd.isna(spread_line):
        return np.nan
    # Create 1D array, then add_constant makes it [1, spread_line]
    X = sm.add_constant(np.array([spread_line]), has_constant='add')
    return spread_model.predict(X)[0]


# ============================================================
# BASELINE + EPA MODELS
# ============================================================

def compute_historical_baselines(games_df, max_season):
    """Compute historical baseline probabilities EXCLUDING test data."""
    train_games = games_df[games_df['season'] <= max_season]
    
    home_games = train_games[train_games['location'] == 'Home']
    if len(home_games) == 0:
        return {'home_win_rate': 0.55}
    
    home_wins = (home_games['winner'] == home_games['home_team']).sum()
    home_win_rate = home_wins / len(home_games)
    
    return {'home_win_rate': home_win_rate}


def baseline_probability(home_seed, away_seed, is_neutral, baselines):
    """Compute baseline win probability for home team based on seed + location."""
    seed_diff = away_seed - home_seed
    
    if is_neutral:
        base_prob = 0.50 + (seed_diff * 0.03)
    else:
        base_prob = baselines['home_win_rate'] + (seed_diff * 0.02)
    
    return np.clip(base_prob, 0.20, 0.85)


def fit_expected_win_pct_model(team_df, max_season):
    """Fit win_pct ~ net_epa for vulnerability calculation"""
    train_data = team_df[team_df['season'] <= max_season].copy()
    train_data = train_data.dropna(subset=['win_pct', 'net_epa'])
    
    if len(train_data) < 20:
        return {'intercept': 0.5, 'slope': 2.0}
    
    X = sm.add_constant(train_data['net_epa'].values.reshape(-1, 1))
    y = train_data['win_pct'].values
    model = sm.OLS(y, X).fit()
    
    return {'intercept': model.params[0], 'slope': model.params[1]}


# ============================================================
# SEASON Z-SCORE NORMALIZATION
# ============================================================

def compute_season_zscores(team_df):
    """
    Compute z-scores for key stats within each season.
    This handles era differences in EPA magnitudes.
    """
    team_df = team_df.copy()
    
    # Stats to z-score
    stats_to_zscore = [
        'passing_epa', 'rushing_epa', 'total_offensive_epa',
        'defensive_epa', 'defensive_pass_epa', 'defensive_rush_epa',
        'net_epa', 'point_differential', 'win_pct',
        'pass_rush_rating', 'pressure_rate', 'sack_rate',
        'pass_block_rating', 'protection_rate', 'sacks_allowed_rate'
    ]
    
    for stat in stats_to_zscore:
        if stat not in team_df.columns:
            continue
        
        z_col = f'z_{stat}'
        team_df[z_col] = team_df.groupby('season')[stat].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
    
    return team_df


# ============================================================
# MATCHUP FEATURE ENGINEERING
# ============================================================

def compute_matchup_features(away_data, home_data, away_game_epa, home_game_epa):
    """
    Compute unit-vs-unit matchup features.
    
    Pass/Rush edges: how does each team's offense match up against opponent's defense?
    Pressure mismatch: OL weakness vs DL strength
    Style skew: who gets to play their preferred game?
    """
    features = {}
    
    # ============================================================
    # PASS/RUSH EDGES (using season-level EPA)
    # ============================================================
    
    # Get z-scored values (fall back to raw if z not available)
    def get_z(data, stat):
        z_col = f'z_{stat}'
        if z_col in data.index:
            return data[z_col]
        elif stat in data.index:
            return data[stat]
        return 0
    
    # Offensive EPA (positive = good)
    home_pass_off = get_z(home_data, 'passing_epa')
    home_rush_off = get_z(home_data, 'rushing_epa')
    away_pass_off = get_z(away_data, 'passing_epa')
    away_rush_off = get_z(away_data, 'rushing_epa')
    
    # Defensive EPA (negative = good, so we negate for "strength")
    home_pass_def_strength = -get_z(home_data, 'defensive_pass_epa')
    home_rush_def_strength = -get_z(home_data, 'defensive_rush_epa')
    away_pass_def_strength = -get_z(away_data, 'defensive_pass_epa')
    away_rush_def_strength = -get_z(away_data, 'defensive_rush_epa')
    
    # Pass edge: how offense performs against opponent's pass defense
    # home_pass_edge = home_pass_off - away_pass_def (away allowing is bad for away)
    home_pass_edge = home_pass_off - (-away_pass_def_strength)  # = home_pass_off + away_pass_def_strength
    home_rush_edge = home_rush_off - (-away_rush_def_strength)
    away_pass_edge = away_pass_off - (-home_pass_def_strength)
    away_rush_edge = away_rush_off - (-home_rush_def_strength)
    
    # Simplified: edge = offense_strength + opponent_defense_weakness
    # Since def_strength = -def_epa, and we want offense vs defense:
    # edge = off_epa - def_epa (where lower def_epa = better defense = harder matchup)
    home_pass_edge = home_pass_off + away_pass_def_strength  # Good offense + weak opponent pass D
    home_rush_edge = home_rush_off + away_rush_def_strength
    away_pass_edge = away_pass_off + home_pass_def_strength
    away_rush_edge = away_rush_off + home_rush_def_strength
    
    features['delta_pass_edge'] = home_pass_edge - away_pass_edge
    features['delta_rush_edge'] = home_rush_edge - away_rush_edge
    
    # Style skew: who gets to play their preferred game?
    home_style_lean = home_pass_edge - home_rush_edge  # Positive = pass-heavy preference
    away_style_lean = away_pass_edge - away_rush_edge
    features['delta_style_skew'] = home_style_lean - away_style_lean
    
    # ============================================================
    # PRESSURE MISMATCH (OL weakness vs DL strength)
    # ============================================================
    
    # Pass protection: higher protection_rate = better OL
    home_pass_prot = get_z(home_data, 'protection_rate')
    away_pass_prot = get_z(away_data, 'protection_rate')
    
    # Pass rush: higher pressure_rate = better DL
    home_pass_rush = get_z(home_data, 'pressure_rate')
    away_pass_rush = get_z(away_data, 'pressure_rate')
    
    # Pressure mismatch: positive = that team's OL is outmatched
    # mismatch = opponent_pass_rush - own_protection
    home_pressure_mismatch = away_pass_rush - home_pass_prot  # Positive = home OL in trouble
    away_pressure_mismatch = home_pass_rush - away_pass_prot
    
    features['delta_pressure_mismatch'] = away_pressure_mismatch - home_pressure_mismatch  # Positive = good for home
    
    # Also include raw pressure differential
    features['delta_pass_rush'] = home_pass_rush - away_pass_rush  # Positive = home has better rush
    features['delta_pass_prot'] = home_pass_prot - away_pass_prot  # Positive = home has better protection
    
    # ============================================================
    # WEAKNESS EXPOSURE FLAGS (quartile interactions)
    # ============================================================
    
    # Flag when a team's weakness meets opponent's strength
    # These are binary indicators that fire when matchup is extreme
    
    # Home OL exposed: home protection is bad (z < -0.5) AND away pass rush is good (z > 0.5)
    features['home_ol_exposed'] = 1 if (home_pass_prot < -0.5 and away_pass_rush > 0.5) else 0
    features['away_ol_exposed'] = 1 if (away_pass_prot < -0.5 and home_pass_rush > 0.5) else 0
    features['delta_ol_exposed'] = features['away_ol_exposed'] - features['home_ol_exposed']
    
    # Pass defense exposed: pass D is bad AND opponent passing is elite
    home_pass_d_exposed = 1 if (home_pass_def_strength < -0.5 and away_pass_off > 0.5) else 0
    away_pass_d_exposed = 1 if (away_pass_def_strength < -0.5 and home_pass_off > 0.5) else 0
    features['delta_pass_d_exposed'] = away_pass_d_exposed - home_pass_d_exposed
    
    # Rush defense exposed
    home_rush_d_exposed = 1 if (home_rush_def_strength < -0.5 and away_rush_off > 0.5) else 0
    away_rush_d_exposed = 1 if (away_rush_def_strength < -0.5 and home_rush_off > 0.5) else 0
    features['delta_rush_d_exposed'] = away_rush_d_exposed - home_rush_d_exposed
    
    return features


# ============================================================
# FEATURE ENGINEERING v9
# ============================================================

def prepare_game_features_v9(games_df, team_df, epa_model, baselines, spread_model):
    """
    Prepare features with v9 improvements:
    - Matchup-based features
    - Z-scored stats
    - Learned spread probability
    - Clean neutral game handling
    """
    
    # Z-score team stats by season
    team_df = compute_season_zscores(team_df)
    
    features = []
    
    for _, game in games_df.iterrows():
        season = game['season']
        is_neutral = game['location'] == 'Neutral'
        
        # Original teams
        orig_away = game['away_team']
        orig_home = game['home_team']
        
        # Get team season data
        orig_away_data = team_df[(team_df['team'] == orig_away) & (team_df['season'] == season)]
        orig_home_data = team_df[(team_df['team'] == orig_home) & (team_df['season'] == season)]
        
        if len(orig_away_data) == 0 or len(orig_home_data) == 0:
            continue
        
        orig_away_data = orig_away_data.iloc[0]
        orig_home_data = orig_home_data.iloc[0]
        
        # Skip if missing EPA
        if pd.isna(game.get('away_offensive_epa')) or pd.isna(game.get('home_offensive_epa')):
            continue
        
        orig_away_seed = int(orig_away_data['playoff_seed'])
        orig_home_seed = int(orig_home_data['playoff_seed'])
        
        # ============================================================
        # NEUTRAL GAME: Redefine "home" as better seed
        # ============================================================
        if is_neutral:
            if orig_away_seed < orig_home_seed:
                away, home = orig_home, orig_away
                away_data, home_data = orig_home_data, orig_away_data
                away_seed, home_seed = orig_home_seed, orig_away_seed
                away_off_epa = game['home_offensive_epa']
                away_def_epa = game['home_defensive_epa']
                home_off_epa = game['away_offensive_epa']
                home_def_epa = game['away_defensive_epa']
                away_pass_epa = game['home_passing_epa']
                away_rush_epa = game['home_rushing_epa']
                home_pass_epa = game['away_passing_epa']
                home_rush_epa = game['away_rushing_epa']
                away_def_pass_epa = game['home_defensive_pass_epa']
                away_def_rush_epa = game['home_defensive_rush_epa']
                home_def_pass_epa = game['away_defensive_pass_epa']
                home_def_rush_epa = game['away_defensive_rush_epa']
                spread_line = -game.get('spread_line', np.nan) if pd.notna(game.get('spread_line')) else np.nan
            else:
                away, home = orig_away, orig_home
                away_data, home_data = orig_away_data, orig_home_data
                away_seed, home_seed = orig_away_seed, orig_home_seed
                away_off_epa = game['away_offensive_epa']
                away_def_epa = game['away_defensive_epa']
                home_off_epa = game['home_offensive_epa']
                home_def_epa = game['home_defensive_epa']
                away_pass_epa = game['away_passing_epa']
                away_rush_epa = game['away_rushing_epa']
                home_pass_epa = game['home_passing_epa']
                home_rush_epa = game['home_rushing_epa']
                away_def_pass_epa = game['away_defensive_pass_epa']
                away_def_rush_epa = game['away_defensive_rush_epa']
                home_def_pass_epa = game['home_defensive_pass_epa']
                home_def_rush_epa = game['home_defensive_rush_epa']
                spread_line = game.get('spread_line', np.nan)
        else:
            away, home = orig_away, orig_home
            away_data, home_data = orig_away_data, orig_home_data
            away_seed, home_seed = orig_away_seed, orig_home_seed
            away_off_epa = game['away_offensive_epa']
            away_def_epa = game['away_defensive_epa']
            home_off_epa = game['home_offensive_epa']
            home_def_epa = game['home_defensive_epa']
            away_pass_epa = game['away_passing_epa']
            away_rush_epa = game['away_rushing_epa']
            home_pass_epa = game['home_passing_epa']
            home_rush_epa = game['home_rushing_epa']
            away_def_pass_epa = game['away_defensive_pass_epa']
            away_def_rush_epa = game['away_defensive_rush_epa']
            home_def_pass_epa = game['home_defensive_pass_epa']
            home_def_rush_epa = game['home_defensive_rush_epa']
            spread_line = game.get('spread_line', np.nan)
        
        # ============================================================
        # CORE STATS
        # ============================================================
        
        away_games = away_data['wins'] + away_data['losses']
        home_games = home_data['wins'] + home_data['losses']
        
        away_pd_pg = away_data['point_differential'] / max(away_games, 1)
        home_pd_pg = home_data['point_differential'] / max(home_games, 1)
        
        away_net_epa = away_off_epa - away_def_epa
        home_net_epa = home_off_epa - home_def_epa
        
        # ============================================================
        # MATCHUP FEATURES (NEW in v9)
        # ============================================================
        
        away_game_epa = {
            'off': away_off_epa, 'pass_off': away_pass_epa, 'rush_off': away_rush_epa,
            'def': away_def_epa, 'pass_def': away_def_pass_epa, 'rush_def': away_def_rush_epa
        }
        home_game_epa = {
            'off': home_off_epa, 'pass_off': home_pass_epa, 'rush_off': home_rush_epa,
            'def': home_def_epa, 'pass_def': home_def_pass_epa, 'rush_def': home_def_rush_epa
        }
        
        matchup_features = compute_matchup_features(away_data, home_data, away_game_epa, home_game_epa)
        
        # ============================================================
        # VULNERABILITY (as residual)
        # ============================================================
        
        away_expected_wpct = epa_model['intercept'] + epa_model['slope'] * away_data['net_epa']
        home_expected_wpct = epa_model['intercept'] + epa_model['slope'] * home_data['net_epa']
        
        away_vulnerability = away_data['win_pct'] - away_expected_wpct
        home_vulnerability = home_data['win_pct'] - home_expected_wpct
        
        # ============================================================
        # UNDERSEEDED (quality vs seed mismatch)
        # ============================================================
        
        season_teams = team_df[team_df['season'] == season].copy()
        season_teams['games'] = season_teams['wins'] + season_teams['losses']
        season_teams['pd_pg'] = season_teams['point_differential'] / season_teams['games'].clip(lower=1)
        season_teams['quality_rank'] = season_teams['pd_pg'].rank(ascending=False)
        
        away_qr = season_teams[season_teams['team'] == away]['quality_rank']
        home_qr = season_teams[season_teams['team'] == home]['quality_rank']
        
        away_quality_rank = int(away_qr.values[0]) if len(away_qr) > 0 else len(season_teams) // 2
        home_quality_rank = int(home_qr.values[0]) if len(home_qr) > 0 else len(season_teams) // 2
        
        away_underseeded = away_seed - away_quality_rank
        home_underseeded = home_seed - home_quality_rank
        
        # ============================================================
        # MOMENTUM
        # ============================================================
        
        away_momentum = away_data.get('momentum_residual', 0)
        home_momentum = home_data.get('momentum_residual', 0)
        if pd.isna(away_momentum): away_momentum = 0
        if pd.isna(home_momentum): home_momentum = 0
        
        # ============================================================
        # BASELINE + SPREAD PROBABILITIES
        # ============================================================
        
        is_neutral_int = 1 if is_neutral else 0
        baseline_prob = baseline_probability(home_seed, away_seed, is_neutral, baselines)
        baseline_logit_val = logit(baseline_prob)
        
        # Learned spread probability
        spread_prob = spread_to_probability_learned(spread_line, spread_model)
        spread_logit_val = logit(spread_prob) if pd.notna(spread_prob) else np.nan
        
        # Choose offset: use spread when available (better prior), else baseline
        if pd.notna(spread_logit_val):
            offset_logit = spread_logit_val
            offset_source = 'spread'
        else:
            offset_logit = baseline_logit_val
            offset_source = 'baseline'
        
        # ============================================================
        # TARGET
        # ============================================================
        
        actual_winner = game['winner']
        home_wins = 1 if actual_winner == home else 0
        
        # ============================================================
        # FEATURE ROW
        # ============================================================
        
        feature_row = {
            'season': season,
            'game_type': game['game_type'],
            'away_team': away,
            'home_team': home,
            'orig_away_team': orig_away,
            'orig_home_team': orig_home,
            'winner': actual_winner,
            'away_score': game['away_score'],
            'home_score': game['home_score'],
            
            'home_wins': home_wins,
            'is_neutral': is_neutral_int,
            
            'away_seed': away_seed,
            'home_seed': home_seed,
            'seed_diff': away_seed - home_seed,
            
            # Core deltas
            'delta_net_epa': home_net_epa - away_net_epa,
            'delta_pd_pg': home_pd_pg - away_pd_pg,
            
            # Vulnerability
            'delta_vulnerability': away_vulnerability - home_vulnerability,
            
            # Underseeded (symmetric)
            'delta_underseeded': away_underseeded - home_underseeded,
            
            # Momentum
            'delta_momentum': home_momentum - away_momentum,
            
            # Baseline
            'baseline_prob': baseline_prob,
            'baseline_logit': baseline_logit_val,
            
            # Spread
            'spread_line': spread_line,
            'spread_prob': spread_prob,
            'spread_logit': spread_logit_val,
            
            # Offset for GLM
            'offset_logit': offset_logit,
            'offset_source': offset_source,
            
            # Raw for analysis
            'away_net_epa': away_net_epa,
            'home_net_epa': home_net_epa,
        }
        
        # Add matchup features
        feature_row.update(matchup_features)
        
        features.append(feature_row)
    
    return pd.DataFrame(features)


# ============================================================
# RIDGE-REGULARIZED OFFSET GLM
# ============================================================

def train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=1.0):
    """
    Train offset GLM with standardization and ridge regularization.
    """
    train_df = features_df[features_df['season'].isin(train_seasons)].copy()
    train_df = train_df.dropna(subset=feature_cols + ['offset_logit'])
    
    if len(train_df) < 30:
        print(f"Warning: Only {len(train_df)} training samples")
        return None
    
    X_raw = train_df[feature_cols].values
    
    # Standardize
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0) + 1e-9
    X_scaled = (X_raw - mu) / sd
    
    X = sm.add_constant(X_scaled)
    y = train_df['home_wins'].values
    offset = train_df['offset_logit'].values
    
    # Fit with ridge regularization
    try:
        glm = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset)
        result = glm.fit_regularized(alpha=alpha, L1_wt=0.0)  # L1_wt=0 is ridge
    except Exception as e:
        print(f"Ridge GLM fit failed: {e}")
        # Fallback to unregularized
        try:
            result = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit()
        except:
            return None
    
    return {
        'model': result,
        'feature_cols': feature_cols,
        'mu': mu,
        'sd': sd,
        'train_samples': len(train_df),
        'alpha': alpha
    }


def calibrate_model_platt(model_dict, features_df, calib_season):
    """
    Platt scaling calibration on holdout season.
    Fits: P(y=1) = sigmoid(a * logit(p_model) + b)
    """
    calib_df = features_df[features_df['season'] == calib_season].copy()
    calib_df = calib_df.dropna(subset=model_dict['feature_cols'] + ['offset_logit'])
    
    if len(calib_df) < 5:
        print(f"Warning: Only {len(calib_df)} calibration samples, skipping Platt scaling")
        return None
    
    # Get model predictions on calibration set
    preds = []
    for _, row in calib_df.iterrows():
        pred = predict_with_ridge_offset(model_dict, row, platt_params=None)
        if pred is not None:
            preds.append({
                'model_prob': pred['home_prob'],
                'actual': row['home_wins']
            })
    
    if len(preds) < 5:
        return None
    
    pred_df = pd.DataFrame(preds)
    
    # Fit Platt scaling: logistic regression of actual on logit(model_prob)
    model_logits = logit(pred_df['model_prob'].clip(0.01, 0.99))
    X_platt = sm.add_constant(model_logits.values)
    y_platt = pred_df['actual'].values
    
    try:
        platt_model = sm.GLM(y_platt, X_platt, family=sm.families.Binomial()).fit()
        platt_params = {'a': platt_model.params[1], 'b': platt_model.params[0]}
        print(f"\nPlatt scaling on {calib_season} ({len(pred_df)} games):")
        print(f"  P_calibrated = sigmoid({platt_params['b']:.3f} + {platt_params['a']:.3f} * logit(P_raw))")
        return platt_params
    except:
        return None


def predict_with_ridge_offset(model_dict, game_features, platt_params=None):
    """Predict using ridge offset model with optional Platt calibration."""
    if model_dict is None:
        return None
    
    try:
        X_raw = np.array([[game_features[col] for col in model_dict['feature_cols']]])
    except KeyError as e:
        return None
    
    if np.any(np.isnan(X_raw)):
        # Fallback to offset-only prediction
        offset = game_features['offset_logit']
        prob = expit(offset)
        return {
            'home_prob': prob,
            'predicted_winner': 'home' if prob > 0.5 else 'away',
            'adjustment': 0,
            'raw_prob': prob
        }
    
    # Standardize
    X_scaled = (X_raw - model_dict['mu']) / model_dict['sd']
    X = sm.add_constant(X_scaled, has_constant='add')
    offset = np.array([game_features['offset_logit']])
    
    # Get raw prediction
    raw_prob = model_dict['model'].predict(X, offset=offset)[0]
    
    # Apply Platt calibration if available
    if platt_params is not None:
        raw_logit = logit(np.clip(raw_prob, 0.01, 0.99))
        calibrated_logit = platt_params['b'] + platt_params['a'] * raw_logit
        final_prob = expit(calibrated_logit)
    else:
        final_prob = raw_prob
    
    adjustment = logit(raw_prob) - game_features['offset_logit']
    
    return {
        'home_prob': final_prob,
        'away_prob': 1 - final_prob,
        'adjustment': adjustment,
        'raw_prob': raw_prob,
        'offset_logit': game_features['offset_logit'],
        'predicted_winner': 'home' if final_prob > 0.5 else 'away'
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(results_df):
    """Comprehensive evaluation metrics"""
    metrics = {}
    
    metrics['accuracy'] = results_df['correct'].mean()
    metrics['n_games'] = len(results_df)
    
    # Model log loss
    actual_probs = results_df.apply(
        lambda r: r['home_prob'] if r['actual_home_wins'] == 1 else 1 - r['home_prob'],
        axis=1
    )
    metrics['log_loss'] = -np.mean(np.log(actual_probs.clip(0.01, 0.99)))
    metrics['brier'] = np.mean((results_df['home_prob'] - results_df['actual_home_wins'])**2)
    
    # Baseline metrics
    baseline_probs = results_df.apply(
        lambda r: r['baseline_prob'] if r['actual_home_wins'] == 1 else 1 - r['baseline_prob'],
        axis=1
    )
    metrics['baseline_log_loss'] = -np.mean(np.log(baseline_probs.clip(0.01, 0.99)))
    metrics['baseline_brier'] = np.mean((results_df['baseline_prob'] - results_df['actual_home_wins'])**2)
    
    # Spread metrics (only on games with spread)
    spread_results = results_df.dropna(subset=['spread_prob'])
    metrics['spread_games'] = len(spread_results)
    if len(spread_results) > 0:
        spread_probs = spread_results.apply(
            lambda r: r['spread_prob'] if r['actual_home_wins'] == 1 else 1 - r['spread_prob'],
            axis=1
        )
        metrics['spread_log_loss'] = -np.mean(np.log(spread_probs.clip(0.01, 0.99)))
        metrics['spread_brier'] = np.mean((spread_results['spread_prob'] - spread_results['actual_home_wins'])**2)
    
    # Upset analysis
    upset_games = results_df[results_df['seed_diff'] > 0]
    if len(upset_games) > 0:
        actual_upsets = upset_games[upset_games['actual_home_wins'] == 0]
        predicted_upsets = upset_games[upset_games['predicted_winner'] == 'away']
        metrics['actual_upsets'] = len(actual_upsets)
        metrics['predicted_upsets'] = len(predicted_upsets)
        if len(actual_upsets) > 0:
            correct_upset_calls = len(actual_upsets[actual_upsets.index.isin(predicted_upsets.index)])
            metrics['upset_recall'] = correct_upset_calls / len(actual_upsets)
    
    return metrics


def validate_on_2024(model_dict, features_df, platt_params):
    """Validate on 2024 playoffs"""
    
    print("\n" + "="*110)
    print("2024 PLAYOFF VALIDATION - MODEL v9 (Ridge Offset GLM + Matchup Features)")
    print("="*110)
    
    test_df = features_df[features_df['season'] == 2024].copy()
    round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
    
    results = []
    
    for _, gf in test_df.iterrows():
        pred = predict_with_ridge_offset(model_dict, gf, platt_params)
        
        if pred is None:
            continue
        
        actual_winner = gf['winner']
        pred_team = gf['home_team'] if pred['predicted_winner'] == 'home' else gf['away_team']
        
        results.append({
            'round': round_names.get(gf['game_type'], gf['game_type']),
            'matchup': f"{gf['orig_away_team']} @ {gf['orig_home_team']}",
            'away_seed': gf['away_seed'],
            'home_seed': gf['home_seed'],
            'seed_diff': gf['seed_diff'],
            'baseline_prob': gf['baseline_prob'],
            'offset_src': gf['offset_source'],
            'adjustment': pred['adjustment'],
            'raw_prob': pred['raw_prob'],
            'home_prob': pred['home_prob'],
            'spread_prob': gf.get('spread_prob', np.nan),
            'predicted': pred_team,
            'predicted_winner': pred['predicted_winner'],
            'actual': actual_winner,
            'correct': pred_team == actual_winner,
            'actual_home_wins': 1 if actual_winner == gf['home_team'] else 0,
            'score': f"{gf['away_score']}-{gf['home_score']}",
            'delta_pass_edge': gf.get('delta_pass_edge', np.nan),
            'delta_pressure_mismatch': gf.get('delta_pressure_mismatch', np.nan),
            'delta_ol_exposed': gf.get('delta_ol_exposed', 0),
        })
    
    results_df = pd.DataFrame(results)
    
    # Display by round
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df[results_df['round'] == round_name]
        if len(rg) == 0:
            continue
        
        print(f"\n{round_name.upper()}")
        print(f"{'Matchup':<16} {'Seeds':<7} {'Offset':<7} {'Adj':>6} {'Raw':>7} {'Final':>7} {'Spread':>7} {'Pred':<5} {'Act':<5} {'✓/✗'}")
        print("-" * 100)
        
        for _, g in rg.iterrows():
            result = "✓" if g['correct'] else "✗"
            seeds = f"{int(g['away_seed'])}v{int(g['home_seed'])}"
            spread_str = f"{g['spread_prob']:.0%}" if pd.notna(g['spread_prob']) else "N/A"
            offset_str = f"{g['baseline_prob']:.0%}{'*' if g['offset_src'] == 'spread' else ''}"
            print(f"{g['matchup']:<16} {seeds:<7} {offset_str:<7} {g['adjustment']:>+5.2f} "
                  f"{g['raw_prob']:>6.1%} {g['home_prob']:>6.1%} {spread_str:>7} "
                  f"{g['predicted']:<5} {g['actual']:<5} {result}")
    
    print("\n  (* = spread used as offset)")
    
    # ============================================================
    # METRICS
    # ============================================================
    
    print(f"\n{'='*110}")
    print("COMPREHENSIVE METRICS - MODEL v9")
    print(f"{'='*110}")
    
    metrics = evaluate_predictions(results_df)
    
    total = len(results_df)
    correct = results_df['correct'].sum()
    
    print(f"\n1. ACCURACY")
    print(f"   Model v9: {correct}/{total} ({metrics['accuracy']:.1%})")
    
    baseline_correct = results_df.apply(
        lambda r: (r['baseline_prob'] > 0.5 and r['actual_home_wins'] == 1) or
                  (r['baseline_prob'] <= 0.5 and r['actual_home_wins'] == 0),
        axis=1
    ).sum()
    print(f"   Baseline: {baseline_correct}/{total} ({baseline_correct/total:.1%})")
    
    print(f"\n   By Round:")
    for rnd in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df[results_df['round'] == rnd]
        if len(rg) > 0:
            print(f"     {rnd:<15} {rg['correct'].sum()}/{len(rg)} ({rg['correct'].mean():.1%})")
    
    print(f"\n2. PROBABILISTIC QUALITY (Lower is better)")
    print(f"   {'Metric':<20} {'Model v9':<12} {'Baseline':<12} {'Spread':<12} {'Spread N'}")
    print(f"   {'-'*68}")
    print(f"   {'Log Loss':<20} {metrics['log_loss']:<12.4f} {metrics['baseline_log_loss']:<12.4f}", end="")
    if 'spread_log_loss' in metrics:
        print(f" {metrics['spread_log_loss']:<12.4f} {metrics['spread_games']}")
    else:
        print(" N/A")
    
    print(f"   {'Brier Score':<20} {metrics['brier']:<12.4f} {metrics['baseline_brier']:<12.4f}", end="")
    if 'spread_brier' in metrics:
        print(f" {metrics['spread_brier']:<12.4f}")
    else:
        print(" N/A")
    
    print(f"\n   Improvement over Baseline (log loss): {metrics['baseline_log_loss'] - metrics['log_loss']:+.4f}")
    if 'spread_log_loss' in metrics:
        print(f"   Improvement over Spread (log loss):   {metrics['spread_log_loss'] - metrics['log_loss']:+.4f}")
    
    print(f"\n3. UPSET ANALYSIS")
    print(f"   Actual upsets: {metrics.get('actual_upsets', 0)}")
    print(f"   Predicted upsets: {metrics.get('predicted_upsets', 0)}")
    print(f"   Upset recall: {metrics.get('upset_recall', 0):.1%}")
    
    # Show matchup features for upsets
    if metrics.get('actual_upsets', 0) > 0:
        upset_games = results_df[(results_df['seed_diff'] > 0) & (results_df['actual_home_wins'] == 0)]
        print(f"\n   Upset Details (with matchup features):")
        for _, g in upset_games.iterrows():
            called = "✓" if g['predicted'] == g['actual'] else "✗"
            print(f"     {g['matchup']}: {g['actual']} won [{called}]")
            print(f"       adj={g['adjustment']:+.2f}, pass_edge={g['delta_pass_edge']:.2f}, "
                  f"pressure={g['delta_pressure_mismatch']:.2f}, ol_exposed={g['delta_ol_exposed']}")
    
    return results_df, metrics


# ============================================================
# MAIN
# ============================================================

def main():
    # Load data
    games_df, team_df, spread_df = load_all_data()
    
    # Merge spread data (fixed method)
    games_df = merge_spread_data_fixed(games_df, spread_df)
    
    # ============================================================
    # SETUP: Use 2023 for baselines (no leakage for 2024 test)
    # ============================================================
    
    train_seasons = list(range(2000, 2023))
    calib_season = 2023
    test_season = 2024
    
    # Baselines can include 2023
    baselines = compute_historical_baselines(games_df, max_season=2023)
    print(f"\nHistorical Baselines (2000-2023):")
    print(f"  Home win rate: {baselines['home_win_rate']:.1%}")
    
    # EPA model
    epa_model = fit_expected_win_pct_model(team_df, max_season=2023)
    print(f"\nEPA -> Win% Model:")
    print(f"  Expected Win% = {epa_model['intercept']:.3f} + {epa_model['slope']:.3f} * net_epa")
    
    # Spread probability model (learned from data)
    # First pass: prepare features without spread model to get spread data
    temp_features = prepare_game_features_v9(games_df, team_df, epa_model, baselines, spread_model=None)
    spread_model = fit_spread_probability_model(temp_features, train_seasons)
    
    # Rebuild features with learned spread model
    print("\nPreparing v9 features with matchup analysis...")
    features_df = prepare_game_features_v9(games_df, team_df, epa_model, baselines, spread_model)
    print(f"Total games with features: {len(features_df)}")
    
    # ============================================================
    # FEATURE SET (with matchup features)
    # ============================================================
    
    feature_cols = [
        # Core deltas
        'delta_net_epa',
        'delta_pd_pg',
        'seed_diff',
        
        # Team quality indicators
        'delta_vulnerability',
        'delta_underseeded',
        'delta_momentum',
        
        # Matchup features (NEW)
        'delta_pass_edge',
        'delta_rush_edge',
        'delta_pressure_mismatch',
        'delta_ol_exposed',
    ]
    
    # Check availability
    available = [f for f in feature_cols if f in features_df.columns]
    missing = [f for f in feature_cols if f not in features_df.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
    feature_cols = available
    
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    
    # ============================================================
    # TRAIN WITH RIDGE REGULARIZATION
    # ============================================================
    
    # Grid search alpha on 2023
    print("\nTuning regularization strength on 2023...")
    best_alpha = 1.0
    best_ll = float('inf')
    
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        model = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=alpha)
        if model is None:
            continue
        
        # Evaluate on 2023
        calib_df = features_df[features_df['season'] == calib_season].dropna(subset=feature_cols + ['offset_logit'])
        preds = []
        for _, row in calib_df.iterrows():
            pred = predict_with_ridge_offset(model, row, platt_params=None)
            if pred:
                actual = row['home_wins']
                prob = pred['home_prob'] if actual == 1 else 1 - pred['home_prob']
                preds.append(prob)
        
        if preds:
            ll = -np.mean(np.log(np.clip(preds, 0.01, 0.99)))
            print(f"  alpha={alpha}: log_loss={ll:.4f}")
            if ll < best_ll:
                best_ll = ll
                best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha}")
    
    # Final model with best alpha, trained on 2000-2022
    model_dict = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=best_alpha)
    
    if model_dict is None:
        print("Training failed!")
        return None, None, None, None
    
    print(f"\nTrained on {model_dict['train_samples']} games with alpha={best_alpha}")
    
    # Show coefficients
    print("\nRidge Offset GLM Coefficients (standardized):")
    print(f"  {'Feature':<30} {'Coef':>10}")
    print(f"  {'-'*42}")
    coef_names = ['const'] + feature_cols
    for name, coef in zip(coef_names, model_dict['model'].params):
        print(f"  {name:<30} {coef:>+10.4f}")
    
    # Calibrate on 2023
    platt_params = calibrate_model_platt(model_dict, features_df, calib_season)
    
    # ============================================================
    # VALIDATE ON 2024
    # ============================================================
    
    results_df, metrics = validate_on_2024(model_dict, features_df, platt_params)
    
    # ============================================================
    # HISTORICAL ROLLING VALIDATION
    # ============================================================
    
    print("\n" + "="*110)
    print("HISTORICAL ROLLING VALIDATION (2015-2024)")
    print("="*110)
    
    all_results = []
    
    for test_year in range(2015, 2025):
        train_yrs = list(range(2000, test_year - 1))  # Leave one year for calibration
        calib_yr = test_year - 1
        
        # Rebuild everything for this window
        window_baselines = compute_historical_baselines(games_df, max_season=test_year - 1)
        window_epa_model = fit_expected_win_pct_model(team_df, max_season=test_year - 1)
        
        # Spread model
        temp_feat = prepare_game_features_v9(games_df, team_df, window_epa_model, window_baselines, None)
        window_spread_model = fit_spread_probability_model(temp_feat, train_yrs)
        
        window_features = prepare_game_features_v9(games_df, team_df, window_epa_model, window_baselines, window_spread_model)
        
        model = train_ridge_offset_model(window_features, feature_cols, train_yrs, alpha=best_alpha)
        if model is None:
            continue
        
        # Calibrate on year before test
        platt = calibrate_model_platt(model, window_features, calib_yr)
        
        test_df = window_features[window_features['season'] == test_year].dropna(subset=feature_cols + ['offset_logit'])
        
        season_results = []
        for _, gf in test_df.iterrows():
            pred = predict_with_ridge_offset(model, gf, platt)
            if pred is None:
                continue
            
            actual_home_wins = 1 if gf['winner'] == gf['home_team'] else 0
            pred_home_wins = 1 if pred['predicted_winner'] == 'home' else 0
            
            season_results.append({
                'correct': actual_home_wins == pred_home_wins,
                'home_prob': pred['home_prob'],
                'baseline_prob': gf['baseline_prob'],
                'actual_home_wins': actual_home_wins
            })
        
        if season_results:
            sdf = pd.DataFrame(season_results)
            
            model_probs = sdf.apply(
                lambda r: r['home_prob'] if r['actual_home_wins'] == 1 else 1 - r['home_prob'], axis=1
            )
            base_probs = sdf.apply(
                lambda r: r['baseline_prob'] if r['actual_home_wins'] == 1 else 1 - r['baseline_prob'], axis=1
            )
            
            all_results.append({
                'season': test_year,
                'games': len(sdf),
                'correct': sdf['correct'].sum(),
                'accuracy': sdf['correct'].mean(),
                'model_ll': -np.mean(np.log(model_probs.clip(0.01, 0.99))),
                'baseline_ll': -np.mean(np.log(base_probs.clip(0.01, 0.99)))
            })
    
    hist_df = pd.DataFrame(all_results)
    hist_df['improvement'] = hist_df['baseline_ll'] - hist_df['model_ll']
    
    print(f"\n{'Season':<10} {'Games':<8} {'Acc':<10} {'Model LL':<12} {'Base LL':<12} {'Δ LL':<12}")
    print("-" * 70)
    for _, row in hist_df.iterrows():
        print(f"{int(row['season']):<10} {int(row['games']):<8} {row['accuracy']:.1%}      "
              f"{row['model_ll']:.4f}       {row['baseline_ll']:.4f}       {row['improvement']:+.4f}")
    
    print("-" * 70)
    total_games = hist_df['games'].sum()
    total_correct = hist_df['correct'].sum()
    avg_model_ll = hist_df['model_ll'].mean()
    avg_base_ll = hist_df['baseline_ll'].mean()
    print(f"{'AVERAGE':<10} {int(total_games):<8} {total_correct/total_games:.1%}      "
          f"{avg_model_ll:.4f}       {avg_base_ll:.4f}       {avg_base_ll - avg_model_ll:+.4f}")
    
    wins = (hist_df['improvement'] > 0).sum()
    print(f"\nModel beats baseline in {wins}/{len(hist_df)} seasons on log loss")
    
    # Save
    results_df.to_csv('../validation_results_2024_v9.csv', index=False)
    print(f"\nResults saved to ../validation_results_2024_v9.csv")
    
    return model_dict, features_df, results_df, metrics


if __name__ == "__main__":
    model_dict, features_df, results, metrics = main()
