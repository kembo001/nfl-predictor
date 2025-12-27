"""
NFL Playoff Model v10 - Fixed Probability Pipeline + Correct Matchup Math

CRITICAL FIXES FROM v9:
1. Spread merge using underdog field (no winner/loser leakage)
2. Use spread linear predictor directly as offset (no logit of saturated prob)
3. Fix matchup edge math (subtract defense strength, not add)
4. Consistent z-score direction (higher = better for all)
5. Multi-year Platt calibration with shrinkage (avoid overfitting)
6. Clip probabilities before logit to avoid infinities
7. Add sanity checks for spread orientation

ARCHITECTURE:
- Offset GLM with ridge regularization
- Offset = spread_linear_predictor when available, else baseline_logit
- Matchup features: pass/rush edges, pressure mismatch, weakness exposure
- Multi-year calibration pool with shrinkage toward identity
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
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


def merge_spread_data_safe(games_df, spread_df):
    """
    Merge spread using underdog field only (no winner/loser to avoid leakage).
    
    Convention: home_spread is from home team perspective
    - Negative = home is favored (laying points)
    - Positive = home is underdog (getting points)
    """
    games_df = games_df.copy()
    
    # The spread file has: underdog, spread_magnitude, spread_line
    # spread_line is already signed (negative = that team favored)
    
    # Build lookup: (season, game_type, team1, team2) -> home_spread
    spread_lookup = {}
    
    for _, row in spread_df.iterrows():
        season = row['season']
        game_type = row['game_type']
        underdog = row['underdog']
        magnitude = abs(row['spread_magnitude'])
        
        # Find the matching game to identify home/away
        # We use underdog + the other team (winner or loser, doesn't matter for matching)
        teams = {row['winner'], row['loser']}
        
        matching = games_df[
            (games_df['season'] == season) & 
            (games_df['game_type'] == game_type) &
            (games_df['home_team'].isin(teams)) &
            (games_df['away_team'].isin(teams))
        ]
        
        if len(matching) > 0:
            game = matching.iloc[0]
            home = game['home_team']
            away = game['away_team']
            
            # Determine home_spread based on who was underdog
            if home == underdog:
                # Home is underdog, getting points -> positive spread
                home_spread = magnitude
            else:
                # Away is underdog, home is favorite -> negative spread
                home_spread = -magnitude
            
            key = (season, game_type, away, home)
            spread_lookup[key] = home_spread
    
    def get_home_spread(row):
        key = (row['season'], row['game_type'], row['away_team'], row['home_team'])
        return spread_lookup.get(key, np.nan)
    
    games_df['home_spread'] = games_df.apply(get_home_spread, axis=1)
    
    matched = games_df['home_spread'].notna().sum()
    print(f"Matched spread data for {matched}/{len(games_df)} games")
    
    # Sanity check: correlation with home wins should be negative
    # (more negative spread = home favored = higher home win rate)
    check_df = games_df.dropna(subset=['home_spread'])
    check_df = check_df.copy()
    check_df['home_win'] = (check_df['winner'] == check_df['home_team']).astype(int)
    corr = check_df['home_spread'].corr(check_df['home_win'])
    print(f"Spread sanity check: corr(home_spread, home_win) = {corr:.3f}")
    if corr > 0:
        print("  WARNING: Positive correlation suggests spread sign may be inverted!")
    else:
        print("  ✓ Negative correlation confirms correct spread orientation")
    
    return games_df


# ============================================================
# SPREAD MODEL (returns linear predictor, not probability)
# ============================================================

def fit_spread_model(features_df, train_seasons):
    """
    Learn spread -> home win probability from historical data.
    Returns model to get LINEAR PREDICTOR (not probability) for use as offset.
    """
    train_df = features_df[
        (features_df['season'].isin(train_seasons)) & 
        (features_df['home_spread'].notna())
    ].copy()
    
    if len(train_df) < 20:
        print(f"Warning: Only {len(train_df)} games with spread for training")
        return None
    
    X = sm.add_constant(train_df['home_spread'].values)
    y = train_df['home_wins'].values
    
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        
        # The slope should be negative (more negative spread = home favored = higher prob)
        slope = model.params[1]
        print(f"\nSpread Model (trained on {len(train_df)} games):")
        print(f"  logit(P(home wins)) = {model.params[0]:.3f} + {slope:.3f} * home_spread")
        
        if slope > 0:
            print("  WARNING: Positive slope suggests spread orientation issue!")
        else:
            print("  ✓ Negative slope confirms correct orientation")
        
        return model
    except Exception as e:
        print(f"Spread model fit failed: {e}")
        return None


def get_spread_offset_logit(home_spread, spread_model):
    """
    Get the LINEAR PREDICTOR from spread model (not probability).
    This avoids infinite logits from saturated probabilities.
    """
    if spread_model is None or pd.isna(home_spread):
        return np.nan
    
    # Linear predictor = b0 + b1 * home_spread
    offset = spread_model.params[0] + spread_model.params[1] * home_spread
    
    # Clip to reasonable range to avoid extreme values
    return np.clip(offset, -4, 4)  # Corresponds to ~1.8% to ~98.2% probability


# ============================================================
# BASELINE MODELS
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
    """Compute baseline win probability for home team."""
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
# SEASON Z-SCORE NORMALIZATION (consistent direction)
# ============================================================

def compute_season_zscores(team_df):
    """
    Compute z-scores for key stats within each season.
    IMPORTANT: All z-scores are oriented so HIGHER = BETTER
    """
    team_df = team_df.copy()
    
    # Stats where higher is already better (no flip needed)
    higher_is_better = [
        'passing_epa', 'rushing_epa', 'total_offensive_epa', 'net_epa',
        'point_differential', 'win_pct',
        'pass_rush_rating', 'pressure_rate', 'sack_rate',  # DL metrics
        'pass_block_rating', 'protection_rate'  # OL metrics
    ]
    
    # Stats where lower is better (need to flip)
    lower_is_better = [
        'defensive_epa', 'defensive_pass_epa', 'defensive_rush_epa',  # Lower EPA allowed = better
        'sacks_allowed_rate'  # Lower sacks allowed = better
    ]
    
    for stat in higher_is_better:
        if stat not in team_df.columns:
            continue
        z_col = f'z_{stat}'
        team_df[z_col] = team_df.groupby('season')[stat].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
    
    for stat in lower_is_better:
        if stat not in team_df.columns:
            continue
        z_col = f'z_{stat}'
        # Flip sign so higher z = better (lower raw value)
        team_df[z_col] = team_df.groupby('season')[stat].transform(
            lambda x: -(x - x.mean()) / (x.std() + 1e-9)
        )
    
    return team_df


# ============================================================
# MATCHUP FEATURES (FIXED MATH)
# ============================================================

def compute_matchup_features(away_data, home_data):
    """
    Compute unit-vs-unit matchup features.
    
    All z-scores are oriented so HIGHER = BETTER.
    Edge = offense_strength - opponent_defense_strength
    (Good offense against weak defense = positive edge)
    """
    features = {}
    
    def get_z(data, stat, default=0):
        z_col = f'z_{stat}'
        if z_col in data.index and pd.notna(data[z_col]):
            return data[z_col]
        return default
    
    # ============================================================
    # PASS/RUSH EDGES
    # Edge = my offense z-score - opponent defense z-score
    # Positive = I have advantage in this phase
    # ============================================================
    
    # Offensive strengths (higher z = better offense)
    home_pass_off = get_z(home_data, 'passing_epa')
    home_rush_off = get_z(home_data, 'rushing_epa')
    away_pass_off = get_z(away_data, 'passing_epa')
    away_rush_off = get_z(away_data, 'rushing_epa')
    
    # Defensive strengths (higher z = better defense, already flipped in z-score)
    home_pass_def = get_z(home_data, 'defensive_pass_epa')
    home_rush_def = get_z(home_data, 'defensive_rush_epa')
    away_pass_def = get_z(away_data, 'defensive_pass_epa')
    away_rush_def = get_z(away_data, 'defensive_rush_epa')
    
    # Pass edge: my passing offense vs their pass defense
    # Positive = my offense is better than their defense can handle
    home_pass_edge = home_pass_off - away_pass_def  # Home passing vs away pass D
    home_rush_edge = home_rush_off - away_rush_def  # Home rushing vs away rush D
    away_pass_edge = away_pass_off - home_pass_def  # Away passing vs home pass D
    away_rush_edge = away_rush_off - home_rush_def  # Away rushing vs home rush D
    
    features['delta_pass_edge'] = home_pass_edge - away_pass_edge
    features['delta_rush_edge'] = home_rush_edge - away_rush_edge
    
    # Style skew: who gets to play their preferred game?
    home_style_lean = home_pass_edge - home_rush_edge
    away_style_lean = away_pass_edge - away_rush_edge
    features['delta_style_skew'] = home_style_lean - away_style_lean
    
    # ============================================================
    # PRESSURE MISMATCH (OL weakness vs DL strength)
    # ============================================================
    
    # OL strength: higher protection_rate z = better OL
    home_ol = get_z(home_data, 'protection_rate')
    away_ol = get_z(away_data, 'protection_rate')
    
    # DL strength: higher pressure_rate z = better pass rush
    home_dl = get_z(home_data, 'pressure_rate')
    away_dl = get_z(away_data, 'pressure_rate')
    
    # Pressure mismatch = opponent DL strength - my OL strength
    # Positive = I'm in trouble (their rush > my protection)
    home_pressure_mismatch = away_dl - home_ol  # Away rushing home's OL
    away_pressure_mismatch = home_dl - away_ol  # Home rushing away's OL
    
    # Delta: positive = away has worse mismatch (good for home)
    features['delta_pressure_mismatch'] = away_pressure_mismatch - home_pressure_mismatch
    
    # ============================================================
    # WEAKNESS EXPOSURE FLAGS
    # Fire when bad unit meets elite opponent unit
    # ============================================================
    
    # Home OL exposed: home OL is bad (z < -0.5) AND away DL is good (z > 0.5)
    features['home_ol_exposed'] = 1 if (home_ol < -0.5 and away_dl > 0.5) else 0
    features['away_ol_exposed'] = 1 if (away_ol < -0.5 and home_dl > 0.5) else 0
    features['delta_ol_exposed'] = features['away_ol_exposed'] - features['home_ol_exposed']
    
    # Pass D exposed: my pass D is bad AND opponent passing is elite
    home_pass_d_exposed = 1 if (home_pass_def < -0.5 and away_pass_off > 0.5) else 0
    away_pass_d_exposed = 1 if (away_pass_def < -0.5 and home_pass_off > 0.5) else 0
    features['delta_pass_d_exposed'] = away_pass_d_exposed - home_pass_d_exposed
    
    return features


# ============================================================
# FEATURE ENGINEERING v10
# ============================================================

def prepare_game_features_v10(games_df, team_df, epa_model, baselines, spread_model):
    """
    Prepare features with v10 fixes:
    - Correct matchup edge math
    - Spread linear predictor as offset (no saturated probs)
    - Consistent z-score orientation
    """
    
    # Z-score team stats by season (with correct orientation)
    team_df = compute_season_zscores(team_df)
    
    features = []
    
    for _, game in games_df.iterrows():
        season = game['season']
        is_neutral = game['location'] == 'Neutral'
        
        orig_away = game['away_team']
        orig_home = game['home_team']
        
        # Get team season data
        orig_away_data = team_df[(team_df['team'] == orig_away) & (team_df['season'] == season)]
        orig_home_data = team_df[(team_df['team'] == orig_home) & (team_df['season'] == season)]
        
        if len(orig_away_data) == 0 or len(orig_home_data) == 0:
            continue
        
        orig_away_data = orig_away_data.iloc[0]
        orig_home_data = orig_home_data.iloc[0]
        
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
                home_spread = -game.get('home_spread', np.nan) if pd.notna(game.get('home_spread')) else np.nan
            else:
                away, home = orig_away, orig_home
                away_data, home_data = orig_away_data, orig_home_data
                away_seed, home_seed = orig_away_seed, orig_home_seed
                home_spread = game.get('home_spread', np.nan)
        else:
            away, home = orig_away, orig_home
            away_data, home_data = orig_away_data, orig_home_data
            away_seed, home_seed = orig_away_seed, orig_home_seed
            home_spread = game.get('home_spread', np.nan)
        
        # ============================================================
        # CORE STATS
        # ============================================================
        
        away_games = away_data['wins'] + away_data['losses']
        home_games = home_data['wins'] + home_data['losses']
        
        away_pd_pg = away_data['point_differential'] / max(away_games, 1)
        home_pd_pg = home_data['point_differential'] / max(home_games, 1)
        
        away_net_epa = away_data['net_epa'] if 'net_epa' in away_data.index else 0
        home_net_epa = home_data['net_epa'] if 'net_epa' in home_data.index else 0
        
        # ============================================================
        # MATCHUP FEATURES
        # ============================================================
        
        matchup_features = compute_matchup_features(away_data, home_data)
        
        # ============================================================
        # VULNERABILITY (as residual)
        # ============================================================
        
        away_expected_wpct = epa_model['intercept'] + epa_model['slope'] * away_net_epa
        home_expected_wpct = epa_model['intercept'] + epa_model['slope'] * home_net_epa
        
        away_vulnerability = away_data['win_pct'] - away_expected_wpct
        home_vulnerability = home_data['win_pct'] - home_expected_wpct
        
        # ============================================================
        # UNDERSEEDED
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
        # BASELINE + SPREAD OFFSETS
        # ============================================================
        
        is_neutral_int = 1 if is_neutral else 0
        baseline_prob = baseline_probability(home_seed, away_seed, is_neutral, baselines)
        baseline_logit_val = logit(baseline_prob)
        
        # Get spread offset as LINEAR PREDICTOR (not logit of probability)
        spread_offset = get_spread_offset_logit(home_spread, spread_model)
        
        # Choose offset: use spread when available
        if pd.notna(spread_offset):
            offset_logit = spread_offset
            offset_source = 'spread'
        else:
            offset_logit = baseline_logit_val
            offset_source = 'baseline'
        
        # For comparison, compute spread probability (clipped)
        if pd.notna(spread_offset):
            spread_prob = expit(spread_offset)
        else:
            spread_prob = np.nan
        
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
            
            # Underseeded
            'delta_underseeded': away_underseeded - home_underseeded,
            
            # Momentum
            'delta_momentum': home_momentum - away_momentum,
            
            # Baseline
            'baseline_prob': baseline_prob,
            'baseline_logit': baseline_logit_val,
            
            # Spread
            'home_spread': home_spread,
            'spread_offset': spread_offset,
            'spread_prob': spread_prob,
            
            # Offset for GLM
            'offset_logit': offset_logit,
            'offset_source': offset_source,
        }
        
        # Add matchup features
        feature_row.update(matchup_features)
        
        features.append(feature_row)
    
    return pd.DataFrame(features)


# ============================================================
# RIDGE OFFSET GLM
# ============================================================

def train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=1.0):
    """Train offset GLM with standardization and ridge regularization."""
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
    
    try:
        glm = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset)
        result = glm.fit_regularized(alpha=alpha, L1_wt=0.0)
    except Exception as e:
        print(f"Ridge GLM fit failed: {e}, trying unregularized")
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


# ============================================================
# MULTI-YEAR PLATT CALIBRATION WITH SHRINKAGE
# ============================================================

def calibrate_model_platt_multiyear(model_dict, features_df, calib_seasons, shrinkage=0.3):
    """
    Platt scaling on multiple years with shrinkage toward identity (a=1, b=0).
    
    P_calibrated = sigmoid(b + a * logit(P_raw))
    Shrinkage pulls toward: a=1, b=0 (identity transformation)
    """
    calib_df = features_df[features_df['season'].isin(calib_seasons)].copy()
    calib_df = calib_df.dropna(subset=model_dict['feature_cols'] + ['offset_logit'])
    
    if len(calib_df) < 10:
        print(f"Warning: Only {len(calib_df)} calibration samples, skipping Platt scaling")
        return {'a': 1.0, 'b': 0.0}  # Identity
    
    # Get model predictions on calibration set
    preds = []
    for _, row in calib_df.iterrows():
        pred = predict_with_offset(model_dict, row, platt_params=None)
        if pred is not None:
            preds.append({
                'model_prob': pred['home_prob'],
                'actual': row['home_wins']
            })
    
    if len(preds) < 10:
        return {'a': 1.0, 'b': 0.0}
    
    pred_df = pd.DataFrame(preds)
    
    # Clip probabilities to avoid infinite logits
    clipped_probs = pred_df['model_prob'].clip(0.05, 0.95)
    model_logits = logit(clipped_probs)
    
    X_platt = sm.add_constant(model_logits.values)
    y_platt = pred_df['actual'].values
    
    try:
        platt_model = sm.GLM(y_platt, X_platt, family=sm.families.Binomial()).fit()
        a_raw = platt_model.params[1]
        b_raw = platt_model.params[0]
        
        # Apply shrinkage toward identity (a=1, b=0)
        a = 1.0 + shrinkage * (a_raw - 1.0)
        b = 0.0 + shrinkage * (b_raw - 0.0)
        
        print(f"\nPlatt scaling on {len(pred_df)} games (seasons {list(calib_seasons)}):")
        print(f"  Raw: a={a_raw:.3f}, b={b_raw:.3f}")
        print(f"  After shrinkage ({shrinkage}): a={a:.3f}, b={b:.3f}")
        
        return {'a': a, 'b': b}
    except:
        return {'a': 1.0, 'b': 0.0}


def predict_with_offset(model_dict, game_features, platt_params=None):
    """Predict using offset model with optional Platt calibration."""
    if model_dict is None:
        return None
    
    try:
        X_raw = np.array([[game_features[col] for col in model_dict['feature_cols']]])
    except KeyError as e:
        return None
    
    if np.any(np.isnan(X_raw)):
        offset = game_features['offset_logit']
        prob = expit(offset)
        return {
            'home_prob': np.clip(prob, 0.01, 0.99),
            'predicted_winner': 'home' if prob > 0.5 else 'away',
            'adjustment': 0,
            'raw_prob': prob
        }
    
    X_scaled = (X_raw - model_dict['mu']) / model_dict['sd']
    X = sm.add_constant(X_scaled, has_constant='add')
    offset = np.array([game_features['offset_logit']])
    
    # Get raw prediction
    raw_prob = model_dict['model'].predict(X, offset=offset)[0]
    raw_prob = np.clip(raw_prob, 0.01, 0.99)
    
    # Apply Platt calibration if available
    if platt_params is not None:
        raw_logit = logit(raw_prob)
        calibrated_logit = platt_params['b'] + platt_params['a'] * raw_logit
        # Clip calibrated logit to avoid extreme probabilities
        calibrated_logit = np.clip(calibrated_logit, -4, 4)
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
    
    print("\n" + "="*115)
    print("2024 PLAYOFF VALIDATION - MODEL v10 (Fixed Pipeline + Correct Matchup Math)")
    print("="*115)
    
    test_df = features_df[features_df['season'] == 2024].copy()
    round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
    
    results = []
    
    for _, gf in test_df.iterrows():
        pred = predict_with_offset(model_dict, gf, platt_params)
        
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
        print(f"{'Matchup':<16} {'Seeds':<7} {'Base':<6} {'Src':<5} {'Adj':>6} {'Raw':>6} {'Final':>6} {'Sprd':>6} {'Pred':<5} {'Act':<5} {'✓'}")
        print("-" * 105)
        
        for _, g in rg.iterrows():
            result = "✓" if g['correct'] else "✗"
            seeds = f"{int(g['away_seed'])}v{int(g['home_seed'])}"
            spread_str = f"{g['spread_prob']:.0%}" if pd.notna(g['spread_prob']) else "N/A"
            src = "sprd" if g['offset_src'] == 'spread' else "base"
            print(f"{g['matchup']:<16} {seeds:<7} {g['baseline_prob']:.0%}   {src:<5} {g['adjustment']:>+5.2f} "
                  f"{g['raw_prob']:>5.0%} {g['home_prob']:>5.0%} {spread_str:>6} "
                  f"{g['predicted']:<5} {g['actual']:<5} {result}")
    
    # ============================================================
    # METRICS
    # ============================================================
    
    print(f"\n{'='*115}")
    print("COMPREHENSIVE METRICS - MODEL v10")
    print(f"{'='*115}")
    
    metrics = evaluate_predictions(results_df)
    
    total = len(results_df)
    correct = results_df['correct'].sum()
    
    print(f"\n1. ACCURACY")
    print(f"   Model v10: {correct}/{total} ({metrics['accuracy']:.1%})")
    
    baseline_correct = results_df.apply(
        lambda r: (r['baseline_prob'] > 0.5 and r['actual_home_wins'] == 1) or
                  (r['baseline_prob'] <= 0.5 and r['actual_home_wins'] == 0),
        axis=1
    ).sum()
    print(f"   Baseline:  {baseline_correct}/{total} ({baseline_correct/total:.1%})")
    
    print(f"\n   By Round:")
    for rnd in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df[results_df['round'] == rnd]
        if len(rg) > 0:
            print(f"     {rnd:<15} {rg['correct'].sum()}/{len(rg)} ({rg['correct'].mean():.1%})")
    
    print(f"\n2. PROBABILISTIC QUALITY (Lower is better)")
    print(f"   {'Metric':<20} {'Model v10':<12} {'Baseline':<12} {'Spread':<12} {'Spread N'}")
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
    
    ll_diff = metrics['baseline_log_loss'] - metrics['log_loss']
    print(f"\n   Improvement over Baseline (log loss): {ll_diff:+.4f} {'✓' if ll_diff > 0 else '✗'}")
    if 'spread_log_loss' in metrics:
        sp_diff = metrics['spread_log_loss'] - metrics['log_loss']
        print(f"   Improvement over Spread (log loss):   {sp_diff:+.4f} {'✓' if sp_diff > 0 else '✗'}")
    
    print(f"\n3. UPSET ANALYSIS")
    print(f"   Actual upsets: {metrics.get('actual_upsets', 0)}")
    print(f"   Predicted upsets: {metrics.get('predicted_upsets', 0)}")
    print(f"   Upset recall: {metrics.get('upset_recall', 0):.1%}")
    
    if metrics.get('actual_upsets', 0) > 0:
        upset_games = results_df[(results_df['seed_diff'] > 0) & (results_df['actual_home_wins'] == 0)]
        print(f"\n   Upset Details:")
        for _, g in upset_games.iterrows():
            called = "✓" if g['predicted'] == g['actual'] else "✗"
            print(f"     {g['matchup']}: {g['actual']} won [{called}]")
            print(f"       adj={g['adjustment']:+.2f}, pass_edge={g['delta_pass_edge']:.2f}, "
                  f"pressure={g['delta_pressure_mismatch']:.2f}")
    
    return results_df, metrics


# ============================================================
# MAIN
# ============================================================

def main():
    # Load data
    games_df, team_df, spread_df = load_all_data()
    
    # Merge spread data (FIXED: using underdog, not winner/loser)
    games_df = merge_spread_data_safe(games_df, spread_df)
    
    # ============================================================
    # SETUP
    # ============================================================
    
    train_seasons = list(range(2000, 2023))
    calib_seasons = [2021, 2022, 2023]  # Multi-year calibration
    test_season = 2024
    
    baselines = compute_historical_baselines(games_df, max_season=2023)
    print(f"\nHistorical Baselines (2000-2023):")
    print(f"  Home win rate: {baselines['home_win_rate']:.1%}")
    
    epa_model = fit_expected_win_pct_model(team_df, max_season=2023)
    print(f"\nEPA -> Win% Model:")
    print(f"  Expected Win% = {epa_model['intercept']:.3f} + {epa_model['slope']:.3f} * net_epa")
    
    # First pass to get spread model
    temp_features = prepare_game_features_v10(games_df, team_df, epa_model, baselines, spread_model=None)
    spread_model = fit_spread_model(temp_features, train_seasons)
    
    # Rebuild features with spread model
    print("\nPreparing v10 features...")
    features_df = prepare_game_features_v10(games_df, team_df, epa_model, baselines, spread_model)
    print(f"Total games with features: {len(features_df)}")
    
    # Additional sanity checks
    print("\nSpread distribution check:")
    spread_stats = features_df['home_spread'].dropna()
    print(f"  N with spread: {len(spread_stats)}")
    print(f"  Mean: {spread_stats.mean():.2f}, Std: {spread_stats.std():.2f}")
    print(f"  Range: [{spread_stats.min():.1f}, {spread_stats.max():.1f}]")
    
    # ============================================================
    # FEATURE SET
    # ============================================================
    
    feature_cols = [
        'delta_net_epa',
        'delta_pd_pg',
        'seed_diff',
        'delta_vulnerability',
        'delta_underseeded',
        'delta_momentum',
        'delta_pass_edge',
        'delta_rush_edge',
        'delta_pressure_mismatch',
        'delta_ol_exposed',
    ]
    
    available = [f for f in feature_cols if f in features_df.columns]
    missing = [f for f in feature_cols if f not in features_df.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
    feature_cols = available
    
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    
    # ============================================================
    # TUNE ALPHA
    # ============================================================
    
    print("\nTuning regularization strength...")
    best_alpha = 1.0
    best_ll = float('inf')
    
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
        model = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=alpha)
        if model is None:
            continue
        
        # Evaluate on 2023
        calib_df = features_df[features_df['season'] == 2023].dropna(subset=feature_cols + ['offset_logit'])
        preds = []
        for _, row in calib_df.iterrows():
            pred = predict_with_offset(model, row, platt_params=None)
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
    
    # Final model
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
        direction = "→ Home" if coef > 0 else "→ Away"
        print(f"  {name:<30} {coef:>+10.4f}  {direction}")
    
    # Multi-year calibration with shrinkage
    platt_params = calibrate_model_platt_multiyear(model_dict, features_df, calib_seasons, shrinkage=0.3)
    
    # ============================================================
    # VALIDATE ON 2024
    # ============================================================
    
    results_df, metrics = validate_on_2024(model_dict, features_df, platt_params)
    
    # ============================================================
    # HISTORICAL VALIDATION
    # ============================================================
    
    print("\n" + "="*115)
    print("HISTORICAL ROLLING VALIDATION (2015-2024)")
    print("="*115)
    
    all_results = []
    
    for test_year in range(2015, 2025):
        train_yrs = list(range(2000, test_year - 2))
        calib_yrs = [test_year - 2, test_year - 1]
        
        window_baselines = compute_historical_baselines(games_df, max_season=test_year - 1)
        window_epa_model = fit_expected_win_pct_model(team_df, max_season=test_year - 1)
        
        temp_feat = prepare_game_features_v10(games_df, team_df, window_epa_model, window_baselines, None)
        window_spread_model = fit_spread_model(temp_feat, train_yrs)
        
        window_features = prepare_game_features_v10(games_df, team_df, window_epa_model, window_baselines, window_spread_model)
        
        model = train_ridge_offset_model(window_features, feature_cols, train_yrs, alpha=best_alpha)
        if model is None:
            continue
        
        platt = calibrate_model_platt_multiyear(model, window_features, calib_yrs, shrinkage=0.3)
        
        test_df = window_features[window_features['season'] == test_year].dropna(subset=feature_cols + ['offset_logit'])
        
        season_results = []
        for _, gf in test_df.iterrows():
            pred = predict_with_offset(model, gf, platt)
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
        better = "✓" if row['improvement'] > 0 else ""
        print(f"{int(row['season']):<10} {int(row['games']):<8} {row['accuracy']:.1%}      "
              f"{row['model_ll']:.4f}       {row['baseline_ll']:.4f}       {row['improvement']:+.4f} {better}")
    
    print("-" * 70)
    total_games = hist_df['games'].sum()
    total_correct = hist_df['correct'].sum()
    avg_model_ll = hist_df['model_ll'].mean()
    avg_base_ll = hist_df['baseline_ll'].mean()
    avg_imp = avg_base_ll - avg_model_ll
    print(f"{'AVERAGE':<10} {int(total_games):<8} {total_correct/total_games:.1%}      "
          f"{avg_model_ll:.4f}       {avg_base_ll:.4f}       {avg_imp:+.4f}")
    
    wins = (hist_df['improvement'] > 0).sum()
    print(f"\nModel beats baseline in {wins}/{len(hist_df)} seasons on log loss")
    
    # Save
    results_df.to_csv('../validation_results_2024_v10.csv', index=False)
    print(f"\nResults saved to ../validation_results_2024_v10.csv")
    
    return model_dict, features_df, results_df, metrics


if __name__ == "__main__":
    model_dict, features_df, results, metrics = main()
