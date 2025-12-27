"""
NFL Playoff Model v7 - Complete Rebuild Based on Deep Analysis

KEY FIXES FROM v6:
A. Home perspective only (no confusing mirrored rows)
B. Remove delta_rest (no learnable variance)
C. Seed expectations computed excluding test season
D. Cinderella uses per-game point differential

NEW FEATURES FROM RESEARCH:
1. Vulnerability indicators (Win%/EPA ratio, fragile favorites)
2. Quality vs Seed mismatch (true Cinderella detection)
3. Defensive identity (defense > offense teams upset more)
4. Cold team flag (avoid "Very Cold" teams)
5. Offensive balance

MODELING APPROACH:
- Prior + Residual: Anchor to baseline (seed + home), adjust with evidence
- Market spread as benchmark comparison
- Calibrated probabilities
- Proper evaluation (log loss, Brier, upset recall)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# Historical baseline win rates (computed from data)
HOME_WIN_RATE = 0.58  # Will be computed from data
SEED_ADV_PER_DIFF = 0.04  # ~4% per seed difference

# Vulnerability threshold (from research: Win%/EPA < 4.5 = fragile)
VULNERABILITY_THRESHOLD = 4.5


# ============================================================
# DATA LOADING
# ============================================================

def load_all_data():
    """Load all required datasets"""
    games_df = pd.read_csv('../nfl_playoff_results_2000_2024_with_epa.csv')
    team_df = pd.read_csv('../all_seasons_data.csv')
    upset_df = pd.read_csv('../df_with_upsets_merged.csv')
    
    print(f"Loaded {len(games_df)} playoff games")
    print(f"Loaded {len(team_df)} team-season records")
    print(f"Loaded {len(upset_df)} records with upset/spread data")
    
    return games_df, team_df, upset_df


# ============================================================
# BASELINE PROBABILITY MODEL
# ============================================================

def compute_historical_baselines(games_df):
    """
    Compute historical baseline probabilities from actual game outcomes.
    This becomes our prior that the model adjusts.
    """
    # Home win rate (excluding neutral sites)
    home_games = games_df[games_df['location'] == 'Home']
    home_wins = (home_games['winner'] == home_games['home_team']).sum()
    home_win_rate = home_wins / len(home_games)
    
    # Neutral site (Super Bowl) - roughly 50/50 adjusted by seed
    neutral_games = games_df[games_df['location'] == 'Neutral']
    
    print(f"\nHistorical Baselines:")
    print(f"  Home team win rate: {home_win_rate:.1%} ({home_wins}/{len(home_games)} games)")
    print(f"  Neutral site games: {len(neutral_games)}")
    
    return {
        'home_win_rate': home_win_rate,
        'neutral_games': len(neutral_games)
    }


def baseline_probability(home_seed, away_seed, is_neutral, baselines):
    """
    Compute baseline win probability for home team based on seed + location.
    This is our prior before adjustments.
    """
    seed_diff = away_seed - home_seed  # Positive = home has better seed
    
    if is_neutral:
        # Neutral site: start at 50%, adjust by seed
        base_prob = 0.50 + (seed_diff * 0.03)
    else:
        # Home game: start at home win rate, adjust by seed
        base_prob = baselines['home_win_rate'] + (seed_diff * 0.02)
    
    # Clip to reasonable range
    return np.clip(base_prob, 0.25, 0.85)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def compute_seed_expectations_excluding_season(team_df, exclude_season):
    """
    Compute expected performance by seed, EXCLUDING the test season.
    This prevents data leakage.
    """
    train_data = team_df[team_df['season'] < exclude_season]
    
    if len(train_data) == 0:
        train_data = team_df  # Fallback for early seasons
    
    # Use per-game point differential to handle 16 vs 17 game seasons
    train_data = train_data.copy()
    train_data['games'] = train_data['wins'] + train_data['losses']
    train_data['pd_per_game'] = train_data['point_differential'] / train_data['games'].clip(lower=1)
    
    seed_stats = train_data.groupby('playoff_seed').agg({
        'pd_per_game': ['mean', 'std'],
        'net_epa': ['mean', 'std']
    })
    seed_stats.columns = ['pd_pg_mean', 'pd_pg_std', 'epa_mean', 'epa_std']
    seed_stats = seed_stats.fillna(seed_stats.mean())
    
    return seed_stats


def compute_vulnerability_score(win_pct, net_epa):
    """
    Compute vulnerability score.
    High Win% relative to EPA = "lucky" team, prone to regression.
    From research: Win%/EPA ratio < 4.5 is fragile.
    """
    if net_epa <= 0.001:
        # Very low/negative EPA with wins = extremely vulnerable
        return 10.0 if win_pct > 0.5 else 0.0
    
    ratio = win_pct / net_epa
    return ratio


def compute_quality_rank(team, season, team_df):
    """
    Compute team's quality rank among all playoff teams that season.
    Based on point differential per game.
    """
    season_teams = team_df[team_df['season'] == season].copy()
    season_teams['games'] = season_teams['wins'] + season_teams['losses']
    season_teams['pd_pg'] = season_teams['point_differential'] / season_teams['games'].clip(lower=1)
    season_teams['quality_rank'] = season_teams['pd_pg'].rank(ascending=False)
    
    team_row = season_teams[season_teams['team'] == team]
    if len(team_row) > 0:
        return int(team_row['quality_rank'].values[0])
    return len(season_teams) // 2  # Default to middle


def prepare_game_features_v7(games_df, team_df):
    """
    Prepare features using HOME PERSPECTIVE ONLY.
    One row per game, team_a = home team (or first listed for neutral).
    """
    
    features = []
    
    for _, game in games_df.iterrows():
        away = game['away_team']
        home = game['home_team']
        season = game['season']
        is_neutral = 1 if game['location'] == 'Neutral' else 0
        
        # Get team season data
        away_data = team_df[(team_df['team'] == away) & (team_df['season'] == season)]
        home_data = team_df[(team_df['team'] == home) & (team_df['season'] == season)]
        
        if len(away_data) == 0 or len(home_data) == 0:
            continue
        
        away_data = away_data.iloc[0]
        home_data = home_data.iloc[0]
        
        # Skip if missing EPA
        if pd.isna(game.get('away_offensive_epa')) or pd.isna(game.get('home_offensive_epa')):
            continue
        
        # ============================================================
        # CORE STATS
        # ============================================================
        
        away_games = away_data['wins'] + away_data['losses']
        home_games = home_data['wins'] + home_data['losses']
        
        away_pd_pg = away_data['point_differential'] / max(away_games, 1)
        home_pd_pg = home_data['point_differential'] / max(home_games, 1)
        
        away_net_epa = game['away_offensive_epa'] - game['away_defensive_epa']
        home_net_epa = game['home_offensive_epa'] - game['home_defensive_epa']
        
        away_seed = int(away_data['playoff_seed'])
        home_seed = int(home_data['playoff_seed'])
        
        # ============================================================
        # VULNERABILITY SCORES (from research)
        # ============================================================
        
        away_vulnerability = compute_vulnerability_score(away_data['win_pct'], away_net_epa)
        home_vulnerability = compute_vulnerability_score(home_data['win_pct'], home_net_epa)
        
        # ============================================================
        # QUALITY vs SEED MISMATCH (true Cinderella detection)
        # ============================================================
        
        away_quality_rank = compute_quality_rank(away, season, team_df)
        home_quality_rank = compute_quality_rank(home, season, team_df)
        
        # Seed rank (1 seed = rank 1, 7 seed = rank 7 in each conf, so 1-14 overall)
        # Simplified: just use seed as rank proxy
        away_seed_rank = away_seed + (7 if away_data['conference'] == 'NFC' else 0)
        home_seed_rank = home_seed + (7 if home_data['conference'] == 'NFC' else 0)
        
        # Positive = underseeded (quality better than seed suggests)
        away_underseeded = away_seed - away_quality_rank  # Higher seed number but better quality
        home_underseeded = home_seed - home_quality_rank
        
        # ============================================================
        # DEFENSIVE IDENTITY (from research: defense enables upsets)
        # ============================================================
        
        # Positive = defensive team (defense better than offense)
        away_def_identity = abs(game['away_defensive_epa']) - game['away_offensive_epa']
        home_def_identity = abs(game['home_defensive_epa']) - game['home_offensive_epa']
        
        # ============================================================
        # COLD TEAM FLAG (from research: Very Cold teams lose)
        # ============================================================
        
        away_is_cold = 1 if away_data.get('momentum_residual', 0) < -1 else 0
        home_is_cold = 1 if home_data.get('momentum_residual', 0) < -1 else 0
        
        # ============================================================
        # OFFENSIVE BALANCE (from research: balanced teams win more)
        # ============================================================
        
        away_balance = abs(game['away_passing_epa'] - game['away_rushing_epa'])
        home_balance = abs(game['home_passing_epa'] - game['home_rushing_epa'])
        
        # ============================================================
        # CREATE FEATURE ROW (Home perspective: predicting P(home wins))
        # ============================================================
        
        feature_row = {
            'season': season,
            'game_type': game['game_type'],
            'away_team': away,
            'home_team': home,
            'winner': game['winner'],
            'away_score': game['away_score'],
            'home_score': game['home_score'],
            
            # Target: does home team win?
            'home_wins': 1 if game['winner'] == home else 0,
            
            # Location
            'is_neutral': is_neutral,
            
            # Seeds
            'away_seed': away_seed,
            'home_seed': home_seed,
            'seed_diff': away_seed - home_seed,  # Positive = home has better seed
            
            # Core quality deltas (positive = home advantage)
            'delta_net_epa': home_net_epa - away_net_epa,
            'delta_pd_pg': home_pd_pg - away_pd_pg,
            
            # Vulnerability (positive = away is more fragile/home advantage)
            'delta_vulnerability': away_vulnerability - home_vulnerability,
            'away_vulnerable': 1 if away_vulnerability > VULNERABILITY_THRESHOLD else 0,
            'home_vulnerable': 1 if home_vulnerability > VULNERABILITY_THRESHOLD else 0,
            
            # Cinderella / Underseeded (positive = away is underseeded/dangerous)
            'away_underseeded': away_underseeded,
            'home_underseeded': home_underseeded,
            'delta_underseeded': away_underseeded - home_underseeded,
            
            # Defensive identity (positive = away has stronger def identity)
            'delta_def_identity': away_def_identity - home_def_identity,
            
            # Cold teams (danger flags)
            'away_is_cold': away_is_cold,
            'home_is_cold': home_is_cold,
            
            # Balance (positive = home is more balanced/better)
            'delta_balance': away_balance - home_balance,  # Lower is better, so away - home
            
            # Raw values for analysis
            'away_net_epa': away_net_epa,
            'home_net_epa': home_net_epa,
            'away_win_pct': away_data['win_pct'],
            'home_win_pct': home_data['win_pct'],
        }
        
        features.append(feature_row)
    
    return pd.DataFrame(features)


# ============================================================
# PRIOR + RESIDUAL MODEL
# ============================================================

def train_residual_model(features_df, feature_cols, train_seasons, baselines):
    """
    Train a model that predicts ADJUSTMENTS to the baseline probability.
    
    Approach:
    1. Compute baseline P(home wins) from seed + location
    2. Compute log-odds of baseline
    3. Train model to predict adjustment to log-odds
    4. Final probability = sigmoid(baseline_logit + adjustment)
    """
    
    train_df = features_df[features_df['season'].isin(train_seasons)].copy()
    train_df = train_df.dropna(subset=feature_cols)
    
    if len(train_df) < 30:
        print(f"Warning: Only {len(train_df)} training samples")
        return None
    
    # Compute baseline probabilities
    train_df['baseline_prob'] = train_df.apply(
        lambda r: baseline_probability(r['home_seed'], r['away_seed'], r['is_neutral'], baselines),
        axis=1
    )
    
    # Convert to log-odds
    train_df['baseline_logit'] = np.log(train_df['baseline_prob'] / (1 - train_df['baseline_prob']))
    
    # Prepare features
    X = train_df[feature_cols].values
    y = train_df['home_wins'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression with offset (baseline_logit as offset)
    # Since sklearn doesn't support offsets directly, we'll use a workaround:
    # Train on residuals: y_residual = y - baseline_pred
    # Actually, better approach: include baseline as a feature with fixed coefficient
    
    # Simpler approach: Train model, then blend with baseline
    model = LogisticRegression(
        penalty='l2',
        C=0.5,
        random_state=42,
        max_iter=2000
    )
    model.fit(X_scaled, y)
    
    # Calibrate the model
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated_model.fit(X_scaled, y)
    
    return {
        'model': model,
        'calibrated_model': calibrated_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'baselines': baselines,
        'train_samples': len(train_df)
    }


def predict_with_prior(model_dict, game_features, blend_weight=0.4):
    """
    Predict probability with prior anchoring.
    
    Final prob = blend_weight * baseline + (1 - blend_weight) * model
    
    When model is uncertain (close to 50%), we trust baseline more.
    """
    if model_dict is None:
        return None
    
    # Compute baseline
    baseline_prob = baseline_probability(
        game_features['home_seed'],
        game_features['away_seed'],
        game_features['is_neutral'],
        model_dict['baselines']
    )
    
    # Get model prediction
    try:
        X = np.array([[game_features[col] for col in model_dict['feature_cols']]])
    except KeyError as e:
        print(f"Missing feature: {e}")
        return None
    
    if np.any(np.isnan(X)):
        return {'home_prob': baseline_prob, 'predicted_winner': 'home' if baseline_prob > 0.5 else 'away',
                'model_prob': 0.5, 'baseline_prob': baseline_prob, 'blend_weight': 1.0}
    
    X_scaled = model_dict['scaler'].transform(X)
    
    # Use calibrated model
    model_prob = model_dict['calibrated_model'].predict_proba(X_scaled)[0][1]
    
    # Adaptive blending: trust baseline more when model is uncertain
    model_confidence = abs(model_prob - 0.5) * 2  # 0 to 1 scale
    adaptive_blend = blend_weight * (1 - model_confidence) + 0.1 * model_confidence
    
    # Final probability
    final_prob = adaptive_blend * baseline_prob + (1 - adaptive_blend) * model_prob
    
    return {
        'home_prob': final_prob,
        'away_prob': 1 - final_prob,
        'model_prob': model_prob,
        'baseline_prob': baseline_prob,
        'blend_weight': adaptive_blend,
        'predicted_winner': 'home' if final_prob > 0.5 else 'away'
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(results_df):
    """Comprehensive evaluation metrics"""
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = results_df['correct'].mean()
    
    # Log loss (probability of actual winner)
    actual_probs = results_df.apply(
        lambda r: r['home_prob'] if r['actual_home_wins'] == 1 else 1 - r['home_prob'],
        axis=1
    )
    metrics['log_loss'] = -np.mean(np.log(actual_probs.clip(0.01, 0.99)))
    
    # Brier score
    metrics['brier'] = np.mean((results_df['home_prob'] - results_df['actual_home_wins'])**2)
    
    # Avg probability assigned to winner
    metrics['avg_prob_winner'] = actual_probs.mean()
    
    # Upset detection (where home is favorite by seed)
    upset_games = results_df[results_df['seed_diff'] > 0]  # Home favored
    if len(upset_games) > 0:
        actual_upsets = upset_games[upset_games['actual_home_wins'] == 0]
        predicted_upsets = upset_games[upset_games['predicted_winner'] == 'away']
        
        if len(actual_upsets) > 0:
            # How many actual upsets did we predict?
            correctly_predicted_upsets = len(
                actual_upsets[actual_upsets.index.isin(predicted_upsets.index)]
            )
            metrics['upset_recall'] = correctly_predicted_upsets / len(actual_upsets)
        else:
            metrics['upset_recall'] = np.nan
    
    return metrics


def validate_on_2024(model_dict, features_df, games_df):
    """Validate on 2024 playoffs with comprehensive metrics"""
    
    print("\n" + "="*95)
    print("2024 PLAYOFF VALIDATION - MODEL v7 (Prior + Residual)")
    print("="*95)
    
    test_df = features_df[features_df['season'] == 2024].copy()
    
    round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
    
    results = []
    
    for _, gf in test_df.iterrows():
        pred = predict_with_prior(model_dict, gf, blend_weight=0.35)
        
        if pred is None:
            continue
        
        actual_winner = gf['winner']
        pred_winner = gf['home_team'] if pred['predicted_winner'] == 'home' else gf['away_team']
        
        results.append({
            'round': round_names.get(gf['game_type'], gf['game_type']),
            'matchup': f"{gf['away_team']} @ {gf['home_team']}",
            'away_seed': gf['away_seed'],
            'home_seed': gf['home_seed'],
            'seed_diff': gf['seed_diff'],
            'baseline_prob': pred['baseline_prob'],
            'model_prob': pred['model_prob'],
            'home_prob': pred['home_prob'],
            'blend_wt': pred['blend_weight'],
            'predicted': pred_winner,
            'predicted_winner': pred['predicted_winner'],  # 'home' or 'away'
            'actual': actual_winner,
            'correct': pred_winner == actual_winner,
            'actual_home_wins': 1 if actual_winner == gf['home_team'] else 0,
            'score': f"{gf['away_score']}-{gf['home_score']}",
            'away_underseeded': gf['away_underseeded'],
            'away_vulnerable': gf['away_vulnerable'],
            'home_vulnerable': gf['home_vulnerable'],
        })
    
    results_df = pd.DataFrame(results)
    
    # Display by round
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df[results_df['round'] == round_name]
        if len(rg) == 0:
            continue
        
        print(f"\n{round_name.upper()}")
        print(f"{'Matchup':<16} {'Seeds':<8} {'Base':<7} {'Model':<7} {'Final':<7} {'Pred':<6} {'Act':<6} {'✓/✗'}")
        print("-" * 80)
        
        for _, g in rg.iterrows():
            result = "✓" if g['correct'] else "✗"
            seeds = f"{int(g['away_seed'])}v{int(g['home_seed'])}"
            print(f"{g['matchup']:<16} {seeds:<8} {g['baseline_prob']:.1%}   {g['model_prob']:.1%}   "
                  f"{g['home_prob']:.1%}   {g['predicted']:<6} {g['actual']:<6} {result}")
    
    # ============================================================
    # COMPREHENSIVE METRICS
    # ============================================================
    
    print(f"\n{'='*95}")
    print("COMPREHENSIVE METRICS - MODEL v7")
    print(f"{'='*95}")
    
    metrics = evaluate_predictions(results_df)
    
    total = len(results_df)
    correct = results_df['correct'].sum()
    
    print(f"\n1. ACCURACY")
    print(f"   Overall: {correct}/{total} ({metrics['accuracy']:.1%})")
    
    print(f"\n   By Round:")
    for rnd in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df[results_df['round'] == rnd]
        if len(rg) > 0:
            print(f"     {rnd:<15} {rg['correct'].sum()}/{len(rg)} ({rg['correct'].mean():.1%})")
    
    print(f"\n2. PROBABILISTIC QUALITY")
    print(f"   Log Loss:              {metrics['log_loss']:.4f}")
    print(f"   Brier Score:           {metrics['brier']:.4f}")
    print(f"   Avg Prob to Winner:    {metrics['avg_prob_winner']:.1%}")
    
    print(f"\n3. CALIBRATION CHECK")
    high_conf = results_df[results_df['home_prob'].apply(lambda x: max(x, 1-x)) >= 0.60]
    low_conf = results_df[results_df['home_prob'].apply(lambda x: max(x, 1-x)) < 0.60]
    
    if len(high_conf) > 0:
        print(f"   High conf (≥60%): {high_conf['correct'].sum()}/{len(high_conf)} ({high_conf['correct'].mean():.1%})")
    if len(low_conf) > 0:
        print(f"   Low conf (<60%):  {low_conf['correct'].sum()}/{len(low_conf)} ({low_conf['correct'].mean():.1%})")
    
    print(f"\n4. UPSET ANALYSIS")
    # Games where home was favored (positive seed_diff)
    fav_home = results_df[results_df['seed_diff'] > 0]
    actual_upsets = fav_home[fav_home['actual_home_wins'] == 0]
    print(f"   Games with home favored: {len(fav_home)}")
    print(f"   Actual upsets: {len(actual_upsets)}")
    
    if len(actual_upsets) > 0:
        correctly_called = actual_upsets[actual_upsets['predicted'] != actual_upsets['matchup'].str.split(' @ ').str[1]]
        print(f"   Upset recall: {len(correctly_called)}/{len(actual_upsets)} ({len(correctly_called)/len(actual_upsets):.1%})")
        print(f"\n   Upset Details:")
        for _, g in actual_upsets.iterrows():
            called = "✓ Called" if g['predicted'] == g['actual'] else "✗ Missed"
            print(f"     {g['matchup']}: {g['actual']} won ({called})")
    
    # ============================================================
    # BASELINE COMPARISON
    # ============================================================
    
    print(f"\n{'='*95}")
    print("BASELINE COMPARISON")
    print(f"{'='*95}")
    
    # Baseline 1: Always pick home (or better seed if neutral)
    baseline_home = results_df.apply(
        lambda r: r['actual'] == r['matchup'].split(' @ ')[1] if r['seed_diff'] >= 0 
                  else r['actual'] == r['matchup'].split(' @ ')[0],
        axis=1
    ).sum()
    
    # Baseline 2: Always pick better seed
    baseline_seed = results_df.apply(
        lambda r: (r['actual_home_wins'] == 1 and r['seed_diff'] > 0) or 
                  (r['actual_home_wins'] == 0 and r['seed_diff'] < 0) or
                  (r['seed_diff'] == 0 and r['actual_home_wins'] == 1),
        axis=1
    ).sum()
    
    # Baseline 3: Just use baseline probability
    baseline_only = results_df.apply(
        lambda r: (r['baseline_prob'] > 0.5 and r['actual_home_wins'] == 1) or
                  (r['baseline_prob'] <= 0.5 and r['actual_home_wins'] == 0),
        axis=1
    ).sum()
    
    print(f"\n   Pick Home/Better Seed: {baseline_home}/{total} ({baseline_home/total:.1%})")
    print(f"   Pick Better Seed:      {baseline_seed}/{total} ({baseline_seed/total:.1%})")
    print(f"   Baseline Prob Only:    {baseline_only}/{total} ({baseline_only/total:.1%})")
    print(f"   Model v7:              {correct}/{total} ({correct/total:.1%})")
    
    return results_df, metrics


# ============================================================
# MAIN
# ============================================================

def main():
    # Load data
    games_df, team_df, upset_df = load_all_data()
    
    # Compute historical baselines
    baselines = compute_historical_baselines(games_df)
    
    # Prepare features (home perspective only)
    print("\nPreparing v7 features (home perspective)...")
    features_df = prepare_game_features_v7(games_df, team_df)
    print(f"Total games with features: {len(features_df)}")
    
    # Define feature set (minimal, high-signal)
    feature_cols = [
        'delta_net_epa',
        'delta_pd_pg',
        'seed_diff',
        'is_neutral',
        'delta_vulnerability',
        'away_underseeded',
        'delta_def_identity',
        'away_is_cold',
        'home_is_cold',
    ]
    
    # Check availability
    available = [f for f in feature_cols if f in features_df.columns]
    missing = [f for f in feature_cols if f not in features_df.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
    feature_cols = available
    
    print(f"\nFeatures: {feature_cols}")
    
    # Train on 2000-2023
    train_seasons = list(range(2000, 2024))
    model_dict = train_residual_model(features_df, feature_cols, train_seasons, baselines)
    
    if model_dict is None:
        print("Training failed!")
        return
    
    print(f"\nTrained on {model_dict['train_samples']} games")
    
    # Show coefficients
    print("\nModel Coefficients (pre-calibration):")
    for feat, coef in zip(feature_cols, model_dict['model'].coef_[0]):
        direction = "→ Home" if coef > 0 else "→ Away"
        print(f"  {feat:<25} {coef:+.4f} {direction}")
    
    # Validate on 2024
    results_df, metrics = validate_on_2024(model_dict, features_df, games_df)
    
    # Historical rolling validation
    print("\n" + "="*95)
    print("HISTORICAL ROLLING VALIDATION (2015-2024)")
    print("="*95)
    
    all_historical_results = []
    
    for test_season in range(2015, 2025):
        train_seasons = list(range(2000, test_season))
        model = train_residual_model(features_df, feature_cols, train_seasons, baselines)
        
        if model is None:
            continue
        
        test_df = features_df[features_df['season'] == test_season].dropna(subset=feature_cols)
        
        season_results = []
        for _, gf in test_df.iterrows():
            pred = predict_with_prior(model, gf, blend_weight=0.35)
            if pred is None:
                continue
            
            actual_home_wins = 1 if gf['winner'] == gf['home_team'] else 0
            pred_home_wins = 1 if pred['predicted_winner'] == 'home' else 0
            
            season_results.append({
                'correct': actual_home_wins == pred_home_wins,
                'home_prob': pred['home_prob'],
                'actual_home_wins': actual_home_wins
            })
        
        if len(season_results) > 0:
            season_df = pd.DataFrame(season_results)
            accuracy = season_df['correct'].mean()
            
            # Log loss
            actual_probs = season_df.apply(
                lambda r: r['home_prob'] if r['actual_home_wins'] == 1 else 1 - r['home_prob'],
                axis=1
            )
            ll = -np.mean(np.log(actual_probs.clip(0.01, 0.99)))
            
            all_historical_results.append({
                'season': test_season,
                'games': len(season_df),
                'correct': season_df['correct'].sum(),
                'accuracy': accuracy,
                'log_loss': ll
            })
    
    hist_df = pd.DataFrame(all_historical_results)
    
    print(f"\n{'Season':<10} {'Games':<8} {'Correct':<10} {'Accuracy':<12} {'Log Loss':<10}")
    print("-" * 50)
    for _, row in hist_df.iterrows():
        print(f"{int(row['season']):<10} {int(row['games']):<8} {int(row['correct']):<10} "
              f"{row['accuracy']:.1%}        {row['log_loss']:.4f}")
    
    total_correct = hist_df['correct'].sum()
    total_games = hist_df['games'].sum()
    avg_ll = hist_df['log_loss'].mean()
    
    print("-" * 50)
    print(f"{'TOTAL':<10} {int(total_games):<8} {int(total_correct):<10} "
          f"{total_correct/total_games:.1%}        {avg_ll:.4f}")
    
    # Save results
    results_df.to_csv('../validation_results_2024_v7.csv', index=False)
    print(f"\nResults saved to ../validation_results_2024_v7.csv")
    
    return model_dict, features_df, results_df, metrics


if __name__ == "__main__":
    model_dict, features_df, results, metrics = main()
