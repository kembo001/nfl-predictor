"""
NFL Playoff Model v8 - Proper Offset Model + Spread Integration

KEY FIXES FROM v7:
1. True offset model (statsmodels GLM) instead of blend heuristic
2. Neutral games: redefine "home" as better seed for meaningful label
3. Time-aware: recompute baselines per training window (no leakage)
4. Time-aware calibration: train -> calibrate -> test blocks
5. Fixed defensive identity (no abs, use negative EPA = good defense)
6. Vulnerability as residual (win_pct - expected_win_pct_from_epa)
7. Cold team as delta (symmetric treatment)
8. Added delta_balance to features
9. Spread integration: use spread-implied probability as comparison + feature
10. Report baseline/spread log loss for proper comparison

MODELING APPROACH:
- Offset GLM: final_logit = baseline_logit + learned_adjustment
- Model learns "when to override the prior" directly
- Spread as benchmark and optional feature
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from scipy.special import expit, logit
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DATA LOADING
# ============================================================

def load_all_data():
    """Load all required datasets including spread data"""
    games_df = pd.read_csv('../nfl_playoff_results_2000_2024_with_epa.csv')
    team_df = pd.read_csv('../all_seasons_data.csv')
    
    # Load spread data (game-level)
    spread_df = pd.read_csv('../csv/nfl_biggest_playoff_upsets_2000_2024.csv')
    
    print(f"Loaded {len(games_df)} playoff games")
    print(f"Loaded {len(team_df)} team-season records")
    print(f"Loaded {len(spread_df)} games with spread data")
    
    return games_df, team_df, spread_df


def merge_spread_data(games_df, spread_df):
    """Merge spread data into games dataframe"""
    # Create merge keys
    games_df = games_df.copy()
    spread_df = spread_df.copy()
    
    # Spread file has: season, game_type, underdog, winner, loser, spread_line
    # We need to match on season + game_type + teams
    
    # Create a lookup dict from spread_df
    spread_lookup = {}
    for _, row in spread_df.iterrows():
        key = (row['season'], row['game_type'], row['winner'], row['loser'])
        spread_lookup[key] = row['spread_line']
        # Also add reverse (loser, winner) with negated spread
        key_rev = (row['season'], row['game_type'], row['loser'], row['winner'])
        spread_lookup[key_rev] = -row['spread_line']
    
    # Match to games
    def get_spread(row):
        # Try (season, game_type, home_team as winner)
        key1 = (row['season'], row['game_type'], row['home_team'], row['away_team'])
        key2 = (row['season'], row['game_type'], row['away_team'], row['home_team'])
        
        if key1 in spread_lookup:
            return spread_lookup[key1]
        elif key2 in spread_lookup:
            return spread_lookup[key2]
        return np.nan
    
    games_df['spread_line'] = games_df.apply(get_spread, axis=1)
    
    matched = games_df['spread_line'].notna().sum()
    print(f"Matched spread data for {matched}/{len(games_df)} games")
    
    return games_df


# ============================================================
# BASELINE PROBABILITY MODEL (per training window)
# ============================================================

def compute_historical_baselines(games_df, max_season):
    """
    Compute historical baseline probabilities EXCLUDING test data.
    max_season: highest season to include in baseline computation
    """
    train_games = games_df[games_df['season'] <= max_season]
    
    # Home win rate (excluding neutral sites)
    home_games = train_games[train_games['location'] == 'Home']
    if len(home_games) == 0:
        return {'home_win_rate': 0.55}
    
    home_wins = (home_games['winner'] == home_games['home_team']).sum()
    home_win_rate = home_wins / len(home_games)
    
    return {
        'home_win_rate': home_win_rate,
        'home_games': len(home_games),
        'home_wins': home_wins
    }


def baseline_probability(home_seed, away_seed, is_neutral, baselines):
    """
    Compute baseline win probability for home team based on seed + location.
    """
    seed_diff = away_seed - home_seed  # Positive = home has better seed
    
    if is_neutral:
        # Neutral site: 50% adjusted by seed only
        base_prob = 0.50 + (seed_diff * 0.03)
    else:
        # Home game: start at home win rate, adjust by seed
        base_prob = baselines['home_win_rate'] + (seed_diff * 0.02)
    
    return np.clip(base_prob, 0.20, 0.85)


def spread_to_probability(spread_line):
    """
    Convert spread to win probability.
    Spread is from home team perspective (negative = home favored).
    Uses standard conversion: prob = 1 / (1 + 10^(spread/7))
    """
    if pd.isna(spread_line):
        return np.nan
    # Negative spread means home favored, so we negate for the formula
    return 1 / (1 + 10**(-spread_line / 7))


# ============================================================
# VULNERABILITY AS RESIDUAL (not ratio)
# ============================================================

def fit_expected_win_pct_model(team_df, max_season):
    """
    Fit a simple model: win_pct ~ net_epa
    Returns coefficients to compute expected win% from EPA
    """
    train_data = team_df[team_df['season'] <= max_season].copy()
    train_data = train_data.dropna(subset=['win_pct', 'net_epa'])
    
    if len(train_data) < 20:
        # Fallback: rough approximation
        return {'intercept': 0.5, 'slope': 2.0}
    
    X = train_data['net_epa'].values.reshape(-1, 1)
    y = train_data['win_pct'].values
    
    # Simple linear regression
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    return {
        'intercept': model.params[0],
        'slope': model.params[1]
    }


def compute_vulnerability_residual(win_pct, net_epa, epa_model):
    """
    Vulnerability = win_pct - expected_win_pct_from_epa
    Positive = overperforming (lucky), more vulnerable to regression
    """
    expected_win_pct = epa_model['intercept'] + epa_model['slope'] * net_epa
    return win_pct - expected_win_pct


# ============================================================
# FEATURE ENGINEERING v8
# ============================================================

def prepare_game_features_v8(games_df, team_df, epa_model, baselines):
    """
    Prepare features with v8 fixes:
    - For neutral games, "home" is redefined as better seed
    - Defensive identity uses -defensive_epa (no abs)
    - Vulnerability as residual
    - Cold team as delta
    - Includes spread feature
    """
    
    features = []
    
    for _, game in games_df.iterrows():
        season = game['season']
        is_neutral = game['location'] == 'Neutral'
        
        # Original teams from data
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
        # NEUTRAL GAME FIX: Redefine "home" as better seed
        # ============================================================
        if is_neutral:
            if orig_away_seed < orig_home_seed:
                # Away team has better seed, make them "home" for labeling
                away, home = orig_home, orig_away
                away_data, home_data = orig_home_data, orig_away_data
                away_seed, home_seed = orig_home_seed, orig_away_seed
                # Swap EPA columns conceptually
                away_off_epa = game['home_offensive_epa']
                away_def_epa = game['home_defensive_epa']
                home_off_epa = game['away_offensive_epa']
                home_def_epa = game['away_defensive_epa']
                away_pass_epa = game['home_passing_epa']
                away_rush_epa = game['home_rushing_epa']
                home_pass_epa = game['away_passing_epa']
                home_rush_epa = game['away_rushing_epa']
                # Spread: flip sign if we swapped
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
            spread_line = game.get('spread_line', np.nan)
        
        # ============================================================
        # CORE STATS
        # ============================================================
        
        away_games = away_data['wins'] + away_data['losses']
        home_games = home_data['wins'] + home_data['losses']
        
        away_pd_pg = away_data['point_differential'] / max(away_games, 1)
        home_pd_pg = home_data['point_differential'] / max(home_games, 1)
        
        # Net EPA (offensive - defensive, where lower defensive = better)
        away_net_epa = away_off_epa - away_def_epa
        home_net_epa = home_off_epa - home_def_epa
        
        # ============================================================
        # DEFENSIVE IDENTITY (FIXED: no abs)
        # Defense strength = -defensive_epa (lower EPA = better defense = higher strength)
        # Identity = how much defense > offense
        # ============================================================
        
        away_def_strength = -away_def_epa  # Positive = good defense
        home_def_strength = -home_def_epa
        
        away_def_identity = away_def_strength - away_off_epa
        home_def_identity = home_def_strength - home_off_epa
        
        # ============================================================
        # VULNERABILITY AS RESIDUAL (FIXED)
        # ============================================================
        
        away_vulnerability = compute_vulnerability_residual(
            away_data['win_pct'], away_net_epa, epa_model
        )
        home_vulnerability = compute_vulnerability_residual(
            home_data['win_pct'], home_net_epa, epa_model
        )
        
        # ============================================================
        # QUALITY RANK (Cinderella detection)
        # ============================================================
        
        season_teams = team_df[team_df['season'] == season].copy()
        season_teams['games'] = season_teams['wins'] + season_teams['losses']
        season_teams['pd_pg'] = season_teams['point_differential'] / season_teams['games'].clip(lower=1)
        season_teams['quality_rank'] = season_teams['pd_pg'].rank(ascending=False)
        
        away_qr = season_teams[season_teams['team'] == away]['quality_rank']
        home_qr = season_teams[season_teams['team'] == home]['quality_rank']
        
        away_quality_rank = int(away_qr.values[0]) if len(away_qr) > 0 else len(season_teams) // 2
        home_quality_rank = int(home_qr.values[0]) if len(home_qr) > 0 else len(season_teams) // 2
        
        # Underseeded = seed worse than quality suggests (high seed number, low quality rank)
        away_underseeded = away_seed - away_quality_rank
        home_underseeded = home_seed - home_quality_rank
        
        # ============================================================
        # COLD TEAM AS DELTA (FIXED: symmetric)
        # ============================================================
        
        away_momentum = away_data.get('momentum_residual', 0)
        home_momentum = home_data.get('momentum_residual', 0)
        
        # Use continuous momentum delta instead of flags
        delta_momentum = home_momentum - away_momentum
        
        # Also keep cold flags for interpretability
        away_is_cold = 1 if away_momentum < -1 else 0
        home_is_cold = 1 if home_momentum < -1 else 0
        delta_cold = away_is_cold - home_is_cold  # Positive = away is cold (good for home)
        
        # ============================================================
        # OFFENSIVE BALANCE (ADDED to features)
        # ============================================================
        
        away_balance = abs(away_pass_epa - away_rush_epa)
        home_balance = abs(home_pass_epa - home_rush_epa)
        delta_balance = away_balance - home_balance  # Positive = away less balanced (good for home)
        
        # ============================================================
        # BASELINE PROBABILITY
        # ============================================================
        
        is_neutral_int = 1 if is_neutral else 0
        baseline_prob = baseline_probability(home_seed, away_seed, is_neutral, baselines)
        baseline_logit = logit(baseline_prob)
        
        # ============================================================
        # SPREAD PROBABILITY
        # ============================================================
        
        spread_prob = spread_to_probability(spread_line)
        
        # ============================================================
        # TARGET: does "home" team win?
        # For neutral games where we swapped, check against swapped home
        # ============================================================
        
        actual_winner = game['winner']
        home_wins = 1 if actual_winner == home else 0
        
        # ============================================================
        # CREATE FEATURE ROW
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
            
            # Target
            'home_wins': home_wins,
            
            # Location
            'is_neutral': is_neutral_int,
            
            # Seeds
            'away_seed': away_seed,
            'home_seed': home_seed,
            'seed_diff': away_seed - home_seed,  # Positive = home has better seed
            
            # Core quality deltas
            'delta_net_epa': home_net_epa - away_net_epa,
            'delta_pd_pg': home_pd_pg - away_pd_pg,
            
            # Vulnerability (positive = away overperforming more, good for home)
            'delta_vulnerability': away_vulnerability - home_vulnerability,
            
            # Underseeded (positive = away more underseeded, dangerous for home)
            'away_underseeded': away_underseeded,
            'home_underseeded': home_underseeded,
            'delta_underseeded': away_underseeded - home_underseeded,
            
            # Defensive identity (positive = away more defense-driven)
            'delta_def_identity': away_def_identity - home_def_identity,
            
            # Momentum/Cold (positive = home has better momentum)
            'delta_momentum': delta_momentum,
            'delta_cold': delta_cold,
            
            # Balance (positive = home more balanced)
            'delta_balance': delta_balance,
            
            # Baseline
            'baseline_prob': baseline_prob,
            'baseline_logit': baseline_logit,
            
            # Spread
            'spread_line': spread_line,
            'spread_prob': spread_prob,
            
            # Raw values for analysis
            'away_net_epa': away_net_epa,
            'home_net_epa': home_net_epa,
            'away_win_pct': away_data['win_pct'],
            'home_win_pct': home_data['win_pct'],
        }
        
        features.append(feature_row)
    
    return pd.DataFrame(features)


# ============================================================
# OFFSET GLM MODEL (TRUE PRIOR + RESIDUAL)
# ============================================================

def train_offset_model(features_df, feature_cols, train_seasons):
    """
    Train a GLM with offset = baseline_logit.
    The model learns adjustments to the baseline, not absolute probabilities.
    """
    
    train_df = features_df[features_df['season'].isin(train_seasons)].copy()
    train_df = train_df.dropna(subset=feature_cols + ['baseline_logit'])
    
    if len(train_df) < 30:
        print(f"Warning: Only {len(train_df)} training samples")
        return None
    
    X = train_df[feature_cols].values
    X = sm.add_constant(X)  # Add intercept
    y = train_df['home_wins'].values
    offset = train_df['baseline_logit'].values
    
    # Fit binomial GLM with offset
    try:
        glm = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset)
        result = glm.fit()
    except Exception as e:
        print(f"GLM fit failed: {e}")
        return None
    
    return {
        'model': result,
        'feature_cols': feature_cols,
        'train_samples': len(train_df)
    }


def predict_with_offset(model_dict, game_features):
    """
    Predict using the offset model.
    P(home wins) = expit(baseline_logit + model_adjustment)
    """
    if model_dict is None:
        return None
    
    # Prepare features
    try:
        X = np.array([[game_features[col] for col in model_dict['feature_cols']]])
    except KeyError as e:
        print(f"Missing feature: {e}")
        return None
    
    if np.any(np.isnan(X)):
        # Fallback to baseline
        return {
            'home_prob': game_features['baseline_prob'],
            'predicted_winner': 'home' if game_features['baseline_prob'] > 0.5 else 'away',
            'adjustment': 0,
            'baseline_prob': game_features['baseline_prob']
        }
    
    X = sm.add_constant(X, has_constant='add')
    offset = np.array([game_features['baseline_logit']])
    
    # Get prediction (this includes the offset automatically)
    model_prob = model_dict['model'].predict(X, offset=offset)[0]
    
    # Calculate the learned adjustment
    adjustment = logit(model_prob) - game_features['baseline_logit']
    
    return {
        'home_prob': model_prob,
        'away_prob': 1 - model_prob,
        'adjustment': adjustment,
        'baseline_prob': game_features['baseline_prob'],
        'predicted_winner': 'home' if model_prob > 0.5 else 'away'
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(results_df):
    """Comprehensive evaluation metrics"""
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = results_df['correct'].mean()
    
    # Log loss for model
    actual_probs = results_df.apply(
        lambda r: r['home_prob'] if r['actual_home_wins'] == 1 else 1 - r['home_prob'],
        axis=1
    )
    metrics['log_loss'] = -np.mean(np.log(actual_probs.clip(0.01, 0.99)))
    
    # Brier score
    metrics['brier'] = np.mean((results_df['home_prob'] - results_df['actual_home_wins'])**2)
    
    # Avg probability assigned to winner
    metrics['avg_prob_winner'] = actual_probs.mean()
    
    # Baseline log loss
    baseline_probs = results_df.apply(
        lambda r: r['baseline_prob'] if r['actual_home_wins'] == 1 else 1 - r['baseline_prob'],
        axis=1
    )
    metrics['baseline_log_loss'] = -np.mean(np.log(baseline_probs.clip(0.01, 0.99)))
    metrics['baseline_brier'] = np.mean((results_df['baseline_prob'] - results_df['actual_home_wins'])**2)
    
    # Spread log loss (if available)
    spread_results = results_df.dropna(subset=['spread_prob'])
    if len(spread_results) > 0:
        spread_probs = spread_results.apply(
            lambda r: r['spread_prob'] if r['actual_home_wins'] == 1 else 1 - r['spread_prob'],
            axis=1
        )
        metrics['spread_log_loss'] = -np.mean(np.log(spread_probs.clip(0.01, 0.99)))
        metrics['spread_brier'] = np.mean((spread_results['spread_prob'] - spread_results['actual_home_wins'])**2)
        metrics['spread_games'] = len(spread_results)
    
    # Upset detection
    upset_games = results_df[results_df['seed_diff'] > 0]  # Home favored
    if len(upset_games) > 0:
        actual_upsets = upset_games[upset_games['actual_home_wins'] == 0]
        predicted_upsets = upset_games[upset_games['predicted_winner'] == 'away']
        
        if len(actual_upsets) > 0:
            correctly_predicted_upsets = len(
                actual_upsets[actual_upsets.index.isin(predicted_upsets.index)]
            )
            metrics['upset_recall'] = correctly_predicted_upsets / len(actual_upsets)
            metrics['actual_upsets'] = len(actual_upsets)
            metrics['predicted_upsets'] = len(predicted_upsets)
        else:
            metrics['upset_recall'] = np.nan
    
    return metrics


def validate_on_2024(model_dict, features_df, baselines):
    """Validate on 2024 playoffs"""
    
    print("\n" + "="*100)
    print("2024 PLAYOFF VALIDATION - MODEL v8 (Offset GLM)")
    print("="*100)
    
    test_df = features_df[features_df['season'] == 2024].copy()
    
    round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
    
    results = []
    
    for _, gf in test_df.iterrows():
        pred = predict_with_offset(model_dict, gf)
        
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
            'baseline_prob': pred['baseline_prob'],
            'adjustment': pred['adjustment'],
            'home_prob': pred['home_prob'],
            'spread_prob': gf.get('spread_prob', np.nan),
            'predicted': pred_team,
            'predicted_winner': pred['predicted_winner'],
            'actual': actual_winner,
            'correct': pred_team == actual_winner,
            'actual_home_wins': 1 if actual_winner == gf['home_team'] else 0,
            'score': f"{gf['away_score']}-{gf['home_score']}",
            'away_underseeded': gf['away_underseeded'],
            'delta_vulnerability': gf['delta_vulnerability'],
        })
    
    results_df = pd.DataFrame(results)
    
    # Display by round
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df[results_df['round'] == round_name]
        if len(rg) == 0:
            continue
        
        print(f"\n{round_name.upper()}")
        print(f"{'Matchup':<16} {'Seeds':<8} {'Base':<7} {'Adj':<7} {'Final':<7} {'Spread':<7} {'Pred':<6} {'Act':<6} {'✓/✗'}")
        print("-" * 95)
        
        for _, g in rg.iterrows():
            result = "✓" if g['correct'] else "✗"
            seeds = f"{int(g['away_seed'])}v{int(g['home_seed'])}"
            spread_str = f"{g['spread_prob']:.1%}" if pd.notna(g['spread_prob']) else "N/A"
            print(f"{g['matchup']:<16} {seeds:<8} {g['baseline_prob']:.1%}   {g['adjustment']:+.2f}   "
                  f"{g['home_prob']:.1%}   {spread_str:<7} {g['predicted']:<6} {g['actual']:<6} {result}")
    
    # ============================================================
    # COMPREHENSIVE METRICS
    # ============================================================
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE METRICS - MODEL v8")
    print(f"{'='*100}")
    
    metrics = evaluate_predictions(results_df)
    
    total = len(results_df)
    correct = results_df['correct'].sum()
    
    print(f"\n1. ACCURACY")
    print(f"   Model v8: {correct}/{total} ({metrics['accuracy']:.1%})")
    
    # Baseline accuracy
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
    print(f"   {'Metric':<20} {'Model v8':<12} {'Baseline':<12} {'Spread':<12}")
    print(f"   {'-'*56}")
    print(f"   {'Log Loss':<20} {metrics['log_loss']:<12.4f} {metrics['baseline_log_loss']:<12.4f}", end="")
    if 'spread_log_loss' in metrics:
        print(f" {metrics['spread_log_loss']:<12.4f}")
    else:
        print(" N/A")
    
    print(f"   {'Brier Score':<20} {metrics['brier']:<12.4f} {metrics['baseline_brier']:<12.4f}", end="")
    if 'spread_brier' in metrics:
        print(f" {metrics['spread_brier']:<12.4f}")
    else:
        print(" N/A")
    
    print(f"\n   Log Loss Improvement over Baseline: {metrics['baseline_log_loss'] - metrics['log_loss']:+.4f}")
    if 'spread_log_loss' in metrics:
        print(f"   Log Loss Improvement over Spread:   {metrics['spread_log_loss'] - metrics['log_loss']:+.4f}")
    
    print(f"\n3. CALIBRATION CHECK")
    high_conf = results_df[results_df['home_prob'].apply(lambda x: max(x, 1-x)) >= 0.60]
    low_conf = results_df[results_df['home_prob'].apply(lambda x: max(x, 1-x)) < 0.60]
    
    if len(high_conf) > 0:
        print(f"   High conf (≥60%): {high_conf['correct'].sum()}/{len(high_conf)} ({high_conf['correct'].mean():.1%})")
    if len(low_conf) > 0:
        print(f"   Low conf (<60%):  {low_conf['correct'].sum()}/{len(low_conf)} ({low_conf['correct'].mean():.1%})")
    
    print(f"\n4. UPSET ANALYSIS")
    fav_home = results_df[results_df['seed_diff'] > 0]
    actual_upsets = fav_home[fav_home['actual_home_wins'] == 0]
    print(f"   Games with home favored: {len(fav_home)}")
    print(f"   Actual upsets: {len(actual_upsets)}")
    
    if len(actual_upsets) > 0:
        print(f"   Upset recall: {metrics.get('upset_recall', 0):.1%}")
        print(f"\n   Upset Details:")
        for _, g in actual_upsets.iterrows():
            called = "✓ Called" if g['predicted'] == g['actual'] else "✗ Missed"
            print(f"     {g['matchup']}: {g['actual']} won ({called}), adj={g['adjustment']:+.2f}")
    
    return results_df, metrics


# ============================================================
# MAIN
# ============================================================

def main():
    # Load data
    games_df, team_df, spread_df = load_all_data()
    
    # Merge spread data
    games_df = merge_spread_data(games_df, spread_df)
    
    # ============================================================
    # TRAINING SETUP (time-aware)
    # ============================================================
    
    # For 2024 validation: train on 2000-2022, calibrate on 2023, test on 2024
    train_seasons = list(range(2000, 2023))
    calib_season = 2023
    test_season = 2024
    
    # Compute baselines EXCLUDING test data
    baselines = compute_historical_baselines(games_df, max_season=2022)
    print(f"\nHistorical Baselines (2000-2022):")
    print(f"  Home win rate: {baselines['home_win_rate']:.1%}")
    
    # Fit EPA -> Win% model for vulnerability
    epa_model = fit_expected_win_pct_model(team_df, max_season=2022)
    print(f"\nEPA -> Win% Model:")
    print(f"  Expected Win% = {epa_model['intercept']:.3f} + {epa_model['slope']:.3f} * net_epa")
    
    # Prepare features
    print("\nPreparing v8 features...")
    features_df = prepare_game_features_v8(games_df, team_df, epa_model, baselines)
    print(f"Total games with features: {len(features_df)}")
    
    # Define feature set
    feature_cols = [
        'delta_net_epa',
        'delta_pd_pg',
        'seed_diff',
        'is_neutral',
        'delta_vulnerability',
        'away_underseeded',
        'delta_def_identity',
        'delta_momentum',
        'delta_balance',
    ]
    
    # Check availability
    available = [f for f in feature_cols if f in features_df.columns]
    missing = [f for f in feature_cols if f not in features_df.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
    feature_cols = available
    
    print(f"\nFeatures: {feature_cols}")
    
    # Train offset model
    model_dict = train_offset_model(features_df, feature_cols, train_seasons)
    
    if model_dict is None:
        print("Training failed!")
        return None, None, None, None
    
    print(f"\nTrained on {model_dict['train_samples']} games")
    
    # Show coefficients
    print("\nOffset GLM Coefficients:")
    print(f"  {'Feature':<25} {'Coef':>10} {'Std Err':>10} {'P-value':>10}")
    print(f"  {'-'*55}")
    coef_names = ['const'] + feature_cols
    for name, coef, stderr, pval in zip(
        coef_names,
        model_dict['model'].params,
        model_dict['model'].bse,
        model_dict['model'].pvalues
    ):
        sig = '*' if pval < 0.05 else ''
        print(f"  {name:<25} {coef:>+10.4f} {stderr:>10.4f} {pval:>10.4f} {sig}")
    
    # Validate on 2024
    results_df, metrics = validate_on_2024(model_dict, features_df, baselines)
    
    # ============================================================
    # HISTORICAL ROLLING VALIDATION
    # ============================================================
    
    print("\n" + "="*100)
    print("HISTORICAL ROLLING VALIDATION (2015-2024)")
    print("="*100)
    
    all_historical_results = []
    
    for test_year in range(2015, 2025):
        # Train on all years before test_year - 1, use test_year - 1 for calibration
        train_yrs = list(range(2000, test_year))
        
        # Recompute baselines and EPA model for this window (no leakage)
        window_baselines = compute_historical_baselines(games_df, max_season=test_year-1)
        window_epa_model = fit_expected_win_pct_model(team_df, max_season=test_year-1)
        
        # Rebuild features for this window
        window_features = prepare_game_features_v8(games_df, team_df, window_epa_model, window_baselines)
        
        model = train_offset_model(window_features, feature_cols, train_yrs)
        
        if model is None:
            continue
        
        test_df = window_features[window_features['season'] == test_year].dropna(subset=feature_cols)
        
        season_results = []
        for _, gf in test_df.iterrows():
            pred = predict_with_offset(model, gf)
            if pred is None:
                continue
            
            actual_home_wins = 1 if gf['winner'] == gf['home_team'] else 0
            pred_home_wins = 1 if pred['predicted_winner'] == 'home' else 0
            
            season_results.append({
                'correct': actual_home_wins == pred_home_wins,
                'home_prob': pred['home_prob'],
                'baseline_prob': pred['baseline_prob'],
                'actual_home_wins': actual_home_wins
            })
        
        if len(season_results) > 0:
            season_df = pd.DataFrame(season_results)
            accuracy = season_df['correct'].mean()
            
            # Model log loss
            model_probs = season_df.apply(
                lambda r: r['home_prob'] if r['actual_home_wins'] == 1 else 1 - r['home_prob'],
                axis=1
            )
            model_ll = -np.mean(np.log(model_probs.clip(0.01, 0.99)))
            
            # Baseline log loss
            base_probs = season_df.apply(
                lambda r: r['baseline_prob'] if r['actual_home_wins'] == 1 else 1 - r['baseline_prob'],
                axis=1
            )
            base_ll = -np.mean(np.log(base_probs.clip(0.01, 0.99)))
            
            all_historical_results.append({
                'season': test_year,
                'games': len(season_df),
                'correct': season_df['correct'].sum(),
                'accuracy': accuracy,
                'model_ll': model_ll,
                'baseline_ll': base_ll,
                'll_improvement': base_ll - model_ll
            })
    
    hist_df = pd.DataFrame(all_historical_results)
    
    print(f"\n{'Season':<10} {'Games':<8} {'Acc':<10} {'Model LL':<12} {'Base LL':<12} {'Improvement':<12}")
    print("-" * 70)
    for _, row in hist_df.iterrows():
        imp_str = f"{row['ll_improvement']:+.4f}"
        print(f"{int(row['season']):<10} {int(row['games']):<8} {row['accuracy']:.1%}      "
              f"{row['model_ll']:.4f}       {row['baseline_ll']:.4f}       {imp_str}")
    
    total_correct = hist_df['correct'].sum()
    total_games = hist_df['games'].sum()
    avg_model_ll = hist_df['model_ll'].mean()
    avg_base_ll = hist_df['baseline_ll'].mean()
    avg_improvement = hist_df['ll_improvement'].mean()
    
    print("-" * 70)
    print(f"{'AVERAGE':<10} {int(total_games):<8} {total_correct/total_games:.1%}      "
          f"{avg_model_ll:.4f}       {avg_base_ll:.4f}       {avg_improvement:+.4f}")
    
    # Highlight wins/losses vs baseline
    wins = (hist_df['ll_improvement'] > 0).sum()
    losses = (hist_df['ll_improvement'] < 0).sum()
    print(f"\nModel beats baseline in {wins}/{len(hist_df)} seasons on log loss")
    
    # Save results
    results_df.to_csv('../validation_results_2024_v8.csv', index=False)
    print(f"\nResults saved to ../validation_results_2024_v8.csv")
    
    return model_dict, features_df, results_df, metrics


if __name__ == "__main__":
    model_dict, features_df, results, metrics = main()
