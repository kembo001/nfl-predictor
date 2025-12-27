"""
NFL Playoff Model v11 - Fixed Spread Scale + Continuous Exposure Scores

CRITICAL FIXES FROM v10:
1. Spread model: scale by touchdowns (spread/7) + ridge regularization
2. Wider logit clip (-6, 6) now that spread scale is realistic
3. Continuous exposure scores (not just binary flags)
4. Diagnostic: offset-only vs model comparison
5. Detailed matchup report for upsets

ARCHITECTURE:
- Offset GLM with ridge regularization
- Spread offset properly scaled (~0.3-0.5 per point equivalent)
- Matchup features: edges + continuous exposure scores
- Multi-year calibration with shrinkage
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
    Merge spread using underdog field only (no winner/loser leakage).
    Convention: home_spread negative = home favored
    """
    games_df = games_df.copy()
    spread_lookup = {}
    
    for _, row in spread_df.iterrows():
        season = row['season']
        game_type = row['game_type']
        underdog = row['underdog']
        magnitude = abs(row['spread_magnitude'])
        
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
            
            if home == underdog:
                home_spread = magnitude  # Home getting points
            else:
                home_spread = -magnitude  # Home laying points
            
            key = (season, game_type, game['away_team'], home)
            spread_lookup[key] = home_spread
    
    def get_home_spread(row):
        key = (row['season'], row['game_type'], row['away_team'], row['home_team'])
        return spread_lookup.get(key, np.nan)
    
    games_df['home_spread'] = games_df.apply(get_home_spread, axis=1)
    
    matched = games_df['home_spread'].notna().sum()
    print(f"Matched spread data for {matched}/{len(games_df)} games")
    
    # Sanity check
    check_df = games_df.dropna(subset=['home_spread']).copy()
    check_df['home_win'] = (check_df['winner'] == check_df['home_team']).astype(int)
    corr = check_df['home_spread'].corr(check_df['home_win'])
    print(f"Spread sanity check: corr(home_spread, home_win) = {corr:.3f}")
    if corr > 0:
        print("  WARNING: Positive correlation suggests spread sign may be inverted!")
    else:
        print("  ✓ Negative correlation confirms correct spread orientation")
    
    return games_df


# ============================================================
# SPREAD MODEL (FIXED: scale by touchdowns + ridge)
# ============================================================

def fit_spread_model_scaled(features_df, train_seasons):
    """
    Learn spread -> home win probability with proper scaling.
    Uses spread/7 (touchdowns) to get reasonable coefficient magnitude.
    Ridge regularization prevents explosion.
    """
    train_df = features_df[
        (features_df['season'].isin(train_seasons)) & 
        (features_df['home_spread'].notna())
    ].copy()
    
    if len(train_df) < 20:
        print(f"Warning: Only {len(train_df)} games with spread for training")
        return None
    
    # Scale spread by touchdowns (7 points)
    spread_td = train_df['home_spread'].values / 7.0
    X = sm.add_constant(spread_td)
    y = train_df['home_wins'].values
    
    try:
        # Use ridge regularization to prevent coefficient explosion
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        model = glm.fit_regularized(alpha=0.5, L1_wt=0.0)  # Ridge
        
        # The slope should be negative (more negative spread = home favored = higher prob)
        slope = model.params[1]
        intercept = model.params[0]
        
        # Convert back to per-point interpretation
        slope_per_point = slope / 7.0
        
        print(f"\nSpread Model (trained on {len(train_df)} games, ridge α=0.5):")
        print(f"  logit(P(home)) = {intercept:.3f} + {slope:.3f} * (spread/7)")
        print(f"  Equivalent: {intercept:.3f} + {slope_per_point:.3f} * spread_points")
        
        if slope > 0:
            print("  WARNING: Positive slope suggests spread orientation issue!")
        else:
            print("  ✓ Negative slope confirms correct orientation")
            
        # Show implied probabilities at key spreads
        print(f"\n  Implied probabilities:")
        for pts in [-10, -7, -3, 0, 3, 7, 10]:
            prob = expit(intercept + slope * (pts / 7.0))
            print(f"    Spread {pts:+d}: P(home) = {prob:.1%}")
        
        return {'intercept': intercept, 'slope': slope, 'scale': 7.0}
    except Exception as e:
        print(f"Spread model fit failed: {e}")
        return None


def get_spread_offset_logit(home_spread, spread_model):
    """
    Get linear predictor from spread model.
    Now properly scaled so clipping is rarely needed.
    """
    if spread_model is None or pd.isna(home_spread):
        return np.nan
    
    # Apply the scaled model
    spread_td = home_spread / spread_model['scale']
    offset = spread_model['intercept'] + spread_model['slope'] * spread_td
    
    # Wider clip now that scale is reasonable (-6, 6 = ~0.2% to ~99.8%)
    return np.clip(offset, -6, 6)


# ============================================================
# BASELINE MODELS
# ============================================================

def compute_historical_baselines(games_df, max_season):
    train_games = games_df[games_df['season'] <= max_season]
    home_games = train_games[train_games['location'] == 'Home']
    if len(home_games) == 0:
        return {'home_win_rate': 0.55}
    home_wins = (home_games['winner'] == home_games['home_team']).sum()
    return {'home_win_rate': home_wins / len(home_games)}


def baseline_probability(home_seed, away_seed, is_neutral, baselines):
    seed_diff = away_seed - home_seed
    if is_neutral:
        base_prob = 0.50 + (seed_diff * 0.03)
    else:
        base_prob = baselines['home_win_rate'] + (seed_diff * 0.02)
    return np.clip(base_prob, 0.20, 0.85)


def fit_expected_win_pct_model(team_df, max_season):
    train_data = team_df[team_df['season'] <= max_season].copy()
    train_data = train_data.dropna(subset=['win_pct', 'net_epa'])
    if len(train_data) < 20:
        return {'intercept': 0.5, 'slope': 2.0}
    X = sm.add_constant(train_data['net_epa'].values.reshape(-1, 1))
    y = train_data['win_pct'].values
    model = sm.OLS(y, X).fit()
    return {'intercept': model.params[0], 'slope': model.params[1]}


# ============================================================
# Z-SCORE NORMALIZATION (higher = better for all)
# ============================================================

def compute_season_zscores(team_df):
    team_df = team_df.copy()
    
    higher_is_better = [
        'passing_epa', 'rushing_epa', 'total_offensive_epa', 'net_epa',
        'point_differential', 'win_pct',
        'pass_rush_rating', 'pressure_rate', 'sack_rate',
        'pass_block_rating', 'protection_rate'
    ]
    
    lower_is_better = [
        'defensive_epa', 'defensive_pass_epa', 'defensive_rush_epa',
        'sacks_allowed_rate'
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
        # Flip sign so higher z = better
        team_df[z_col] = team_df.groupby('season')[stat].transform(
            lambda x: -(x - x.mean()) / (x.std() + 1e-9)
        )
    
    return team_df


# ============================================================
# MATCHUP FEATURES (with continuous exposure scores)
# ============================================================

def compute_matchup_features(away_data, home_data):
    """
    Compute matchup features including:
    - Pass/rush edges
    - Continuous exposure scores (not just flags)
    - Binary exposure flags for interpretability
    """
    features = {}
    
    def get_z(data, stat, default=0):
        z_col = f'z_{stat}'
        if z_col in data.index and pd.notna(data[z_col]):
            return data[z_col]
        return default
    
    # ============================================================
    # PASS/RUSH EDGES
    # ============================================================
    
    home_pass_off = get_z(home_data, 'passing_epa')
    home_rush_off = get_z(home_data, 'rushing_epa')
    away_pass_off = get_z(away_data, 'passing_epa')
    away_rush_off = get_z(away_data, 'rushing_epa')
    
    home_pass_def = get_z(home_data, 'defensive_pass_epa')
    home_rush_def = get_z(home_data, 'defensive_rush_epa')
    away_pass_def = get_z(away_data, 'defensive_pass_epa')
    away_rush_def = get_z(away_data, 'defensive_rush_epa')
    
    home_pass_edge = home_pass_off - away_pass_def
    home_rush_edge = home_rush_off - away_rush_def
    away_pass_edge = away_pass_off - home_pass_def
    away_rush_edge = away_rush_off - home_rush_def
    
    features['delta_pass_edge'] = home_pass_edge - away_pass_edge
    features['delta_rush_edge'] = home_rush_edge - away_rush_edge
    
    # Style skew
    home_style_lean = home_pass_edge - home_rush_edge
    away_style_lean = away_pass_edge - away_rush_edge
    features['delta_style_skew'] = home_style_lean - away_style_lean
    
    # ============================================================
    # CONTINUOUS EXPOSURE SCORES (NEW)
    # Exposure = max(0, opponent_strength - my_weakness)
    # Positive = I'm exposed in this matchup
    # ============================================================
    
    home_ol = get_z(home_data, 'protection_rate')
    away_ol = get_z(away_data, 'protection_rate')
    home_dl = get_z(home_data, 'pressure_rate')
    away_dl = get_z(away_data, 'pressure_rate')
    
    # OL exposure: how much opponent's DL exceeds my OL
    home_ol_exposure = np.maximum(0, away_dl - home_ol)
    away_ol_exposure = np.maximum(0, home_dl - away_ol)
    features['delta_ol_exposure'] = away_ol_exposure - home_ol_exposure  # Positive = good for home
    
    # Pass defense exposure: opponent's passing vs my pass D
    home_pass_d_exposure = np.maximum(0, away_pass_off - home_pass_def)
    away_pass_d_exposure = np.maximum(0, home_pass_off - away_pass_def)
    features['delta_pass_d_exposure'] = away_pass_d_exposure - home_pass_d_exposure
    
    # Rush defense exposure
    home_rush_d_exposure = np.maximum(0, away_rush_off - home_rush_def)
    away_rush_d_exposure = np.maximum(0, home_rush_off - away_rush_def)
    features['delta_rush_d_exposure'] = away_rush_d_exposure - home_rush_d_exposure
    
    # ============================================================
    # BINARY FLAGS (for interpretability)
    # ============================================================
    
    features['home_ol_exposed_flag'] = 1 if (home_ol < -0.5 and away_dl > 0.5) else 0
    features['away_ol_exposed_flag'] = 1 if (away_ol < -0.5 and home_dl > 0.5) else 0
    features['delta_ol_exposed_flag'] = features['away_ol_exposed_flag'] - features['home_ol_exposed_flag']
    
    # ============================================================
    # RAW VALUES FOR MATCHUP REPORT
    # ============================================================
    
    features['home_ol_z'] = home_ol
    features['away_ol_z'] = away_ol
    features['home_dl_z'] = home_dl
    features['away_dl_z'] = away_dl
    features['home_pass_off_z'] = home_pass_off
    features['away_pass_off_z'] = away_pass_off
    features['home_pass_def_z'] = home_pass_def
    features['away_pass_def_z'] = away_pass_def
    
    return features


# ============================================================
# FEATURE ENGINEERING v11
# ============================================================

def prepare_game_features_v11(games_df, team_df, epa_model, baselines, spread_model):
    """Prepare features with v11 improvements."""
    
    team_df = compute_season_zscores(team_df)
    features = []
    
    for _, game in games_df.iterrows():
        season = game['season']
        is_neutral = game['location'] == 'Neutral'
        
        orig_away = game['away_team']
        orig_home = game['home_team']
        
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
        
        # Neutral game handling
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
        
        # Core stats
        away_games = away_data['wins'] + away_data['losses']
        home_games = home_data['wins'] + home_data['losses']
        
        away_pd_pg = away_data['point_differential'] / max(away_games, 1)
        home_pd_pg = home_data['point_differential'] / max(home_games, 1)
        
        away_net_epa = away_data['net_epa'] if 'net_epa' in away_data.index else 0
        home_net_epa = home_data['net_epa'] if 'net_epa' in home_data.index else 0
        
        # Matchup features
        matchup_features = compute_matchup_features(away_data, home_data)
        
        # Vulnerability
        away_expected_wpct = epa_model['intercept'] + epa_model['slope'] * away_net_epa
        home_expected_wpct = epa_model['intercept'] + epa_model['slope'] * home_net_epa
        away_vulnerability = away_data['win_pct'] - away_expected_wpct
        home_vulnerability = home_data['win_pct'] - home_expected_wpct
        
        # Underseeded
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
        
        # Momentum
        away_momentum = away_data.get('momentum_residual', 0)
        home_momentum = home_data.get('momentum_residual', 0)
        if pd.isna(away_momentum): away_momentum = 0
        if pd.isna(home_momentum): home_momentum = 0
        
        # Offsets
        is_neutral_int = 1 if is_neutral else 0
        baseline_prob = baseline_probability(home_seed, away_seed, is_neutral, baselines)
        baseline_logit_val = logit(baseline_prob)
        
        spread_offset = get_spread_offset_logit(home_spread, spread_model)
        
        if pd.notna(spread_offset):
            offset_logit = spread_offset
            offset_source = 'spread'
            spread_prob = expit(spread_offset)
        else:
            offset_logit = baseline_logit_val
            offset_source = 'baseline'
            spread_prob = np.nan
        
        # Target
        actual_winner = game['winner']
        home_wins = 1 if actual_winner == home else 0
        
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
            
            'delta_net_epa': home_net_epa - away_net_epa,
            'delta_pd_pg': home_pd_pg - away_pd_pg,
            'delta_vulnerability': away_vulnerability - home_vulnerability,
            'delta_underseeded': away_underseeded - home_underseeded,
            'delta_momentum': home_momentum - away_momentum,
            
            'baseline_prob': baseline_prob,
            'baseline_logit': baseline_logit_val,
            'home_spread': home_spread,
            'spread_offset': spread_offset,
            'spread_prob': spread_prob,
            'offset_logit': offset_logit,
            'offset_source': offset_source,
        }
        
        feature_row.update(matchup_features)
        features.append(feature_row)
    
    return pd.DataFrame(features)


# ============================================================
# RIDGE OFFSET GLM
# ============================================================

def train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=1.0):
    train_df = features_df[features_df['season'].isin(train_seasons)].copy()
    train_df = train_df.dropna(subset=feature_cols + ['offset_logit'])
    
    if len(train_df) < 30:
        print(f"Warning: Only {len(train_df)} training samples")
        return None
    
    X_raw = train_df[feature_cols].values
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
        print(f"Ridge GLM fit failed: {e}")
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
# PLATT CALIBRATION
# ============================================================

def calibrate_model_platt_multiyear(model_dict, features_df, calib_seasons, shrinkage=0.3):
    calib_df = features_df[features_df['season'].isin(calib_seasons)].copy()
    calib_df = calib_df.dropna(subset=model_dict['feature_cols'] + ['offset_logit'])
    
    if len(calib_df) < 10:
        print(f"Warning: Only {len(calib_df)} calibration samples")
        return {'a': 1.0, 'b': 0.0}
    
    preds = []
    for _, row in calib_df.iterrows():
        pred = predict_with_offset(model_dict, row, platt_params=None)
        if pred is not None:
            preds.append({'model_prob': pred['home_prob'], 'actual': row['home_wins']})
    
    if len(preds) < 10:
        return {'a': 1.0, 'b': 0.0}
    
    pred_df = pd.DataFrame(preds)
    clipped_probs = pred_df['model_prob'].clip(0.05, 0.95)
    model_logits = logit(clipped_probs)
    
    X_platt = sm.add_constant(model_logits.values)
    y_platt = pred_df['actual'].values
    
    try:
        platt_model = sm.GLM(y_platt, X_platt, family=sm.families.Binomial()).fit()
        a_raw, b_raw = platt_model.params[1], platt_model.params[0]
        
        a = 1.0 + shrinkage * (a_raw - 1.0)
        b = 0.0 + shrinkage * (b_raw - 0.0)
        
        print(f"\nPlatt scaling on {len(pred_df)} games (seasons {list(calib_seasons)}):")
        print(f"  Raw: a={a_raw:.3f}, b={b_raw:.3f}")
        print(f"  After shrinkage ({shrinkage}): a={a:.3f}, b={b:.3f}")
        
        return {'a': a, 'b': b}
    except:
        return {'a': 1.0, 'b': 0.0}


def predict_with_offset(model_dict, game_features, platt_params=None):
    if model_dict is None:
        return None
    
    try:
        X_raw = np.array([[game_features[col] for col in model_dict['feature_cols']]])
    except KeyError:
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
    
    raw_prob = model_dict['model'].predict(X, offset=offset)[0]
    raw_prob = np.clip(raw_prob, 0.01, 0.99)
    
    if platt_params is not None:
        raw_logit = logit(raw_prob)
        calibrated_logit = platt_params['b'] + platt_params['a'] * raw_logit
        calibrated_logit = np.clip(calibrated_logit, -5, 5)
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
    metrics = {}
    metrics['accuracy'] = results_df['correct'].mean()
    metrics['n_games'] = len(results_df)
    
    actual_probs = results_df.apply(
        lambda r: r['home_prob'] if r['actual_home_wins'] == 1 else 1 - r['home_prob'], axis=1
    )
    metrics['log_loss'] = -np.mean(np.log(actual_probs.clip(0.01, 0.99)))
    metrics['brier'] = np.mean((results_df['home_prob'] - results_df['actual_home_wins'])**2)
    
    baseline_probs = results_df.apply(
        lambda r: r['baseline_prob'] if r['actual_home_wins'] == 1 else 1 - r['baseline_prob'], axis=1
    )
    metrics['baseline_log_loss'] = -np.mean(np.log(baseline_probs.clip(0.01, 0.99)))
    metrics['baseline_brier'] = np.mean((results_df['baseline_prob'] - results_df['actual_home_wins'])**2)
    
    # Offset-only metrics (diagnostic)
    offset_probs = results_df.apply(
        lambda r: expit(r['offset_logit']) if r['actual_home_wins'] == 1 else 1 - expit(r['offset_logit']), axis=1
    )
    metrics['offset_only_log_loss'] = -np.mean(np.log(offset_probs.clip(0.01, 0.99)))
    
    spread_results = results_df.dropna(subset=['spread_prob'])
    metrics['spread_games'] = len(spread_results)
    if len(spread_results) > 0:
        spread_probs = spread_results.apply(
            lambda r: r['spread_prob'] if r['actual_home_wins'] == 1 else 1 - r['spread_prob'], axis=1
        )
        metrics['spread_log_loss'] = -np.mean(np.log(spread_probs.clip(0.01, 0.99)))
        metrics['spread_brier'] = np.mean((spread_results['spread_prob'] - spread_results['actual_home_wins'])**2)
    
    upset_games = results_df[results_df['seed_diff'] > 0]
    if len(upset_games) > 0:
        actual_upsets = upset_games[upset_games['actual_home_wins'] == 0]
        predicted_upsets = upset_games[upset_games['predicted_winner'] == 'away']
        metrics['actual_upsets'] = len(actual_upsets)
        metrics['predicted_upsets'] = len(predicted_upsets)
        if len(actual_upsets) > 0:
            correct = len(actual_upsets[actual_upsets.index.isin(predicted_upsets.index)])
            metrics['upset_recall'] = correct / len(actual_upsets)
    
    return metrics


def print_matchup_report(results_df, features_df):
    """Print detailed matchup report for upsets."""
    
    print(f"\n{'='*100}")
    print("DETAILED MATCHUP REPORT - 2024 UPSETS")
    print(f"{'='*100}")
    
    upset_games = results_df[(results_df['seed_diff'] > 0) & (results_df['actual_home_wins'] == 0)]
    
    for _, g in upset_games.iterrows():
        matchup = g['matchup']
        
        # Get full feature row
        feat = features_df[
            (features_df['season'] == 2024) & 
            (features_df['orig_away_team'] == matchup.split(' @ ')[0]) &
            (features_df['orig_home_team'] == matchup.split(' @ ')[1])
        ]
        
        if len(feat) == 0:
            continue
        f = feat.iloc[0]
        
        called = "✓ CALLED" if g['predicted'] == g['actual'] else "✗ MISSED"
        
        print(f"\n{matchup}: {g['actual']} won [{called}]")
        print(f"  Score: {g['score']}, Seeds: {int(g['away_seed'])} vs {int(g['home_seed'])}")
        print(f"  Model: {g['home_prob']:.0%} home, Baseline: {g['baseline_prob']:.0%}")
        
        print(f"\n  MATCHUP EDGES:")
        print(f"    Pass Edge (home adv):  {f.get('delta_pass_edge', 0):+.2f}")
        print(f"    Rush Edge (home adv):  {f.get('delta_rush_edge', 0):+.2f}")
        
        print(f"\n  EXPOSURE SCORES (positive = away more exposed):")
        print(f"    OL Exposure:      {f.get('delta_ol_exposure', 0):+.2f}")
        print(f"    Pass D Exposure:  {f.get('delta_pass_d_exposure', 0):+.2f}")
        print(f"    Rush D Exposure:  {f.get('delta_rush_d_exposure', 0):+.2f}")
        
        print(f"\n  RAW Z-SCORES:")
        print(f"    Home OL: {f.get('home_ol_z', 0):+.2f}, Away DL: {f.get('away_dl_z', 0):+.2f} → OL exposed: {f.get('home_ol_exposed_flag', 0)}")
        print(f"    Away OL: {f.get('away_ol_z', 0):+.2f}, Home DL: {f.get('home_dl_z', 0):+.2f} → OL exposed: {f.get('away_ol_exposed_flag', 0)}")
        print(f"    Home Pass Off: {f.get('home_pass_off_z', 0):+.2f}, Away Pass Def: {f.get('away_pass_def_z', 0):+.2f}")
        print(f"    Away Pass Off: {f.get('away_pass_off_z', 0):+.2f}, Home Pass Def: {f.get('home_pass_def_z', 0):+.2f}")


def validate_on_2024(model_dict, features_df, platt_params):
    """Validate on 2024 playoffs"""
    
    print("\n" + "="*115)
    print("2024 PLAYOFF VALIDATION - MODEL v11 (Fixed Spread Scale + Continuous Exposure)")
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
            'offset_logit': gf['offset_logit'],
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
            'delta_ol_exposure': gf.get('delta_ol_exposure', np.nan),
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
    
    # Metrics
    print(f"\n{'='*115}")
    print("COMPREHENSIVE METRICS - MODEL v11")
    print(f"{'='*115}")
    
    metrics = evaluate_predictions(results_df)
    
    total = len(results_df)
    correct = results_df['correct'].sum()
    
    print(f"\n1. ACCURACY")
    print(f"   Model v11: {correct}/{total} ({metrics['accuracy']:.1%})")
    
    baseline_correct = results_df.apply(
        lambda r: (r['baseline_prob'] > 0.5 and r['actual_home_wins'] == 1) or
                  (r['baseline_prob'] <= 0.5 and r['actual_home_wins'] == 0), axis=1
    ).sum()
    print(f"   Baseline:  {baseline_correct}/{total} ({baseline_correct/total:.1%})")
    
    print(f"\n   By Round:")
    for rnd in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df[results_df['round'] == rnd]
        if len(rg) > 0:
            print(f"     {rnd:<15} {rg['correct'].sum()}/{len(rg)} ({rg['correct'].mean():.1%})")
    
    print(f"\n2. PROBABILISTIC QUALITY (Lower is better)")
    print(f"   {'Metric':<20} {'Model':<10} {'Offset':<10} {'Baseline':<10} {'Spread':<10} {'N'}")
    print(f"   {'-'*72}")
    print(f"   {'Log Loss':<20} {metrics['log_loss']:<10.4f} {metrics['offset_only_log_loss']:<10.4f} "
          f"{metrics['baseline_log_loss']:<10.4f}", end="")
    if 'spread_log_loss' in metrics:
        print(f" {metrics['spread_log_loss']:<10.4f} {metrics['spread_games']}")
    else:
        print(" N/A")
    
    print(f"   {'Brier Score':<20} {metrics['brier']:<10.4f} {'--':<10} {metrics['baseline_brier']:<10.4f}", end="")
    if 'spread_brier' in metrics:
        print(f" {metrics['spread_brier']:<10.4f}")
    else:
        print(" N/A")
    
    ll_vs_base = metrics['baseline_log_loss'] - metrics['log_loss']
    ll_vs_offset = metrics['offset_only_log_loss'] - metrics['log_loss']
    print(f"\n   Model vs Baseline:    {ll_vs_base:+.4f} {'✓' if ll_vs_base > 0 else '✗'}")
    print(f"   Model vs Offset-only: {ll_vs_offset:+.4f} {'✓' if ll_vs_offset > 0 else '✗'} (tests if features add value)")
    
    print(f"\n3. UPSET ANALYSIS")
    print(f"   Actual upsets: {metrics.get('actual_upsets', 0)}")
    print(f"   Predicted upsets: {metrics.get('predicted_upsets', 0)}")
    print(f"   Upset recall: {metrics.get('upset_recall', 0):.1%}")
    
    return results_df, metrics


# ============================================================
# MAIN
# ============================================================

def main():
    games_df, team_df, spread_df = load_all_data()
    games_df = merge_spread_data_safe(games_df, spread_df)
    
    train_seasons = list(range(2000, 2023))
    calib_seasons = [2021, 2022, 2023]
    test_season = 2024
    
    baselines = compute_historical_baselines(games_df, max_season=2023)
    print(f"\nHistorical Baselines: Home win rate = {baselines['home_win_rate']:.1%}")
    
    epa_model = fit_expected_win_pct_model(team_df, max_season=2023)
    print(f"EPA -> Win%: {epa_model['intercept']:.3f} + {epa_model['slope']:.3f} * net_epa")
    
    # First pass for spread model
    temp_features = prepare_game_features_v11(games_df, team_df, epa_model, baselines, spread_model=None)
    spread_model = fit_spread_model_scaled(temp_features, train_seasons)
    
    # Rebuild features
    print("\nPreparing v11 features...")
    features_df = prepare_game_features_v11(games_df, team_df, epa_model, baselines, spread_model)
    print(f"Total games: {len(features_df)}")
    
    # Features (now with continuous exposure scores)
    feature_cols = [
        'delta_net_epa',
        'delta_pd_pg',
        'seed_diff',
        'delta_vulnerability',
        'delta_underseeded',
        'delta_momentum',
        'delta_pass_edge',
        'delta_rush_edge',
        'delta_ol_exposure',        # Continuous
        'delta_pass_d_exposure',    # Continuous
    ]
    
    available = [f for f in feature_cols if f in features_df.columns]
    print(f"\nFeatures ({len(available)}): {available}")
    feature_cols = available
    
    # Tune alpha
    print("\nTuning regularization...")
    best_alpha, best_ll = 1.0, float('inf')
    
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        model = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=alpha)
        if model is None:
            continue
        
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
                best_ll, best_alpha = ll, alpha
    
    print(f"\nBest alpha: {best_alpha}")
    
    model_dict = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=best_alpha)
    print(f"Trained on {model_dict['train_samples']} games")
    
    # Coefficients
    print("\nRidge Offset GLM Coefficients (standardized):")
    print(f"  {'Feature':<25} {'Coef':>10} {'Direction'}")
    print(f"  {'-'*50}")
    coef_names = ['const'] + feature_cols
    for name, coef in zip(coef_names, model_dict['model'].params):
        direction = "→ Home" if coef > 0 else "→ Away"
        print(f"  {name:<25} {coef:>+10.4f}  {direction}")
    
    platt_params = calibrate_model_platt_multiyear(model_dict, features_df, calib_seasons, shrinkage=0.3)
    
    results_df, metrics = validate_on_2024(model_dict, features_df, platt_params)
    
    # Matchup report for upsets
    print_matchup_report(results_df, features_df)
    
    # Historical validation
    print("\n" + "="*115)
    print("HISTORICAL ROLLING VALIDATION (2015-2024)")
    print("="*115)
    
    all_results = []
    
    for test_year in range(2015, 2025):
        train_yrs = list(range(2000, test_year - 2))
        calib_yrs = [test_year - 2, test_year - 1]
        
        window_baselines = compute_historical_baselines(games_df, max_season=test_year - 1)
        window_epa_model = fit_expected_win_pct_model(team_df, max_season=test_year - 1)
        
        temp_feat = prepare_game_features_v11(games_df, team_df, window_epa_model, window_baselines, None)
        window_spread_model = fit_spread_model_scaled(temp_feat, train_yrs)
        
        window_features = prepare_game_features_v11(games_df, team_df, window_epa_model, window_baselines, window_spread_model)
        
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
                'offset_logit': gf['offset_logit'],
                'actual_home_wins': actual_home_wins
            })
        
        if season_results:
            sdf = pd.DataFrame(season_results)
            
            model_probs = sdf.apply(lambda r: r['home_prob'] if r['actual_home_wins'] == 1 else 1 - r['home_prob'], axis=1)
            base_probs = sdf.apply(lambda r: r['baseline_prob'] if r['actual_home_wins'] == 1 else 1 - r['baseline_prob'], axis=1)
            offset_probs = sdf.apply(lambda r: expit(r['offset_logit']) if r['actual_home_wins'] == 1 else 1 - expit(r['offset_logit']), axis=1)
            
            all_results.append({
                'season': test_year,
                'games': len(sdf),
                'correct': sdf['correct'].sum(),
                'accuracy': sdf['correct'].mean(),
                'model_ll': -np.mean(np.log(model_probs.clip(0.01, 0.99))),
                'offset_ll': -np.mean(np.log(offset_probs.clip(0.01, 0.99))),
                'baseline_ll': -np.mean(np.log(base_probs.clip(0.01, 0.99)))
            })
    
    hist_df = pd.DataFrame(all_results)
    hist_df['vs_base'] = hist_df['baseline_ll'] - hist_df['model_ll']
    hist_df['vs_offset'] = hist_df['offset_ll'] - hist_df['model_ll']
    
    print(f"\n{'Season':<8} {'Games':<7} {'Acc':<8} {'Model':<9} {'Offset':<9} {'Base':<9} {'vs Base':<10} {'vs Off':<10}")
    print("-" * 85)
    for _, row in hist_df.iterrows():
        print(f"{int(row['season']):<8} {int(row['games']):<7} {row['accuracy']:.1%}    "
              f"{row['model_ll']:.4f}    {row['offset_ll']:.4f}    {row['baseline_ll']:.4f}    "
              f"{row['vs_base']:+.4f} {'✓' if row['vs_base'] > 0 else ''}    "
              f"{row['vs_offset']:+.4f} {'✓' if row['vs_offset'] > 0 else ''}")
    
    print("-" * 85)
    avg_model = hist_df['model_ll'].mean()
    avg_offset = hist_df['offset_ll'].mean()
    avg_base = hist_df['baseline_ll'].mean()
    print(f"{'AVG':<8} {int(hist_df['games'].sum()):<7} {hist_df['correct'].sum()/hist_df['games'].sum():.1%}    "
          f"{avg_model:.4f}    {avg_offset:.4f}    {avg_base:.4f}    "
          f"{avg_base - avg_model:+.4f}      {avg_offset - avg_model:+.4f}")
    
    wins_base = (hist_df['vs_base'] > 0).sum()
    wins_offset = (hist_df['vs_offset'] > 0).sum()
    print(f"\nModel beats baseline in {wins_base}/{len(hist_df)} seasons")
    print(f"Model beats offset-only in {wins_offset}/{len(hist_df)} seasons (features add value)")
    
    results_df.to_csv('../validation_results_2024_v11.csv', index=False)
    print(f"\nResults saved to ../validation_results_2024_v11.csv")
    
    return model_dict, features_df, results_df, metrics


if __name__ == "__main__":
    model_dict, features_df, results, metrics = main()
