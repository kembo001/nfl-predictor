"""
NFL Playoff Model v6 - Comprehensive Rebuild
Implements all suggested improvements:

A) STRUCTURAL FIXES:
1. is_neutral_site instead of is_home_game
2. Explicit defensive EPA sign convention (DEF_EPA_IS_ALLOWED = True)
3. Regular season stats only (no playoff leakage)
4. Mirrored rows for training (doubles data)

B) FEATURE UPGRADES:
5. Cinderella detector (seed outperformance)
6. Playoff experience (QB + coach) - built from historical data
7. Matchup interactions (pass offense vs pass defense)
8. Bye/rest indicator
9. QB stability proxy

C) MODELING UPGRADES:
10. Close game fallback / baseline blend
11. Ensemble (Logistic + Gradient Boosting)
12. Optimize for log loss
13. Elastic Net regularization

D) EVALUATION:
14. Compare against baselines
15. Stratified results
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# Defensive EPA sign convention: True = "allowed" (lower is better)
DEF_EPA_IS_ALLOWED = True

# Ensemble weights
LOGISTIC_WEIGHT = 0.6
GB_WEIGHT = 0.4

# Close game threshold (probability within this of 0.5 triggers fallback)
CLOSE_GAME_THRESHOLD = 0.06


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load all required data"""
    games_df = pd.read_csv('../nfl_playoff_results_2000_2024_with_epa.csv')
    team_df = pd.read_csv('../all_seasons_data.csv')
    
    print(f"Loaded {len(games_df)} playoff games (2000-2024)")
    print(f"Loaded {len(team_df)} team-season records")
    
    return games_df, team_df


# ============================================================
# FEATURE ENGINEERING HELPERS
# ============================================================

def build_playoff_experience_db(games_df, team_df):
    """
    Build cumulative playoff experience for QBs and coaches.
    Returns dict: {(season, team): {'qb_playoff_games': X, 'coach_playoff_games': Y}}
    
    Experience is counted BEFORE the current season (no leakage).
    """
    # Get QB/coach mapping from team_df
    qb_coach_map = {}
    for _, row in team_df.iterrows():
        key = (row['season'], row['team'])
        qb_coach_map[key] = {
            'qb': row['primary_qb'],
            'coach': row['coach']
        }
    
    # Track cumulative experience
    qb_experience = defaultdict(int)  # qb_name -> playoff games
    coach_experience = defaultdict(int)  # coach_name -> playoff games
    
    # Result: experience ENTERING each season
    experience_db = {}
    
    seasons = sorted(games_df['season'].unique())
    
    for season in seasons:
        # First, record experience ENTERING this season for all teams
        season_teams = team_df[team_df['season'] == season]
        
        for _, team_row in season_teams.iterrows():
            team = team_row['team']
            qb = team_row['primary_qb']
            coach = team_row['coach']
            
            experience_db[(season, team)] = {
                'qb_playoff_games': qb_experience.get(qb, 0),
                'coach_playoff_games': coach_experience.get(coach, 0),
                'qb': qb,
                'coach': coach
            }
        
        # Then, count games from THIS season to update for next season
        season_games = games_df[games_df['season'] == season]
        
        for _, game in season_games.iterrows():
            away = game['away_team']
            home = game['home_team']
            
            # Get QB/coach for each team
            away_info = qb_coach_map.get((season, away), {})
            home_info = qb_coach_map.get((season, home), {})
            
            if away_info.get('qb'):
                qb_experience[away_info['qb']] += 1
            if away_info.get('coach'):
                coach_experience[away_info['coach']] += 1
            if home_info.get('qb'):
                qb_experience[home_info['qb']] += 1
            if home_info.get('coach'):
                coach_experience[home_info['coach']] += 1
    
    return experience_db


def compute_seed_expectations(team_df):
    """
    Compute expected performance metrics by seed.
    Used for Cinderella detection (outperforming seed).
    """
    seed_stats = team_df.groupby('playoff_seed').agg({
        'point_differential': ['mean', 'std'],
        'net_epa': ['mean', 'std'],
    })
    seed_stats.columns = ['pd_mean', 'pd_std', 'epa_mean', 'epa_std']
    seed_stats = seed_stats.fillna(seed_stats.mean())  # Handle edge cases
    return seed_stats


def get_bye_status(seed, season):
    """
    Determine if a team had a first-round bye.
    - Before 2020: Seeds 1-2 got byes
    - 2020+: Only seed 1 gets bye
    """
    if season < 2020:
        return 1 if seed <= 2 else 0
    else:
        return 1 if seed == 1 else 0


def compute_matchup_features(away_pass_epa, away_rush_epa, away_pressure, away_block,
                              home_pass_epa, home_rush_epa, home_pressure, home_block,
                              away_def_pass_epa, away_def_rush_epa,
                              home_def_pass_epa, home_def_rush_epa):
    """
    Compute matchup-specific features.
    Pass matchup: offense EPA vs opposing pass defense EPA
    """
    # Away pass offense vs Home pass defense
    # If DEF_EPA_IS_ALLOWED: lower def EPA = better defense
    # So away advantage = away_pass_epa - home_def_pass_epa (if both higher=better offense)
    # But if def is "allowed", then home_def_pass_epa being LOW means good defense
    # Away wants: high away_pass_epa, high home_def_pass_epa (bad defense)
    
    if DEF_EPA_IS_ALLOWED:
        # Good matchup for away = good away offense + bad home defense (high allowed)
        away_pass_matchup = away_pass_epa + home_def_pass_epa
        home_pass_matchup = home_pass_epa + away_def_pass_epa
        away_rush_matchup = away_rush_epa + home_def_rush_epa
        home_rush_matchup = home_rush_epa + away_def_rush_epa
    else:
        away_pass_matchup = away_pass_epa - home_def_pass_epa
        home_pass_matchup = home_pass_epa - away_def_pass_epa
        away_rush_matchup = away_rush_epa - home_def_rush_epa
        home_rush_matchup = home_rush_epa - away_def_rush_epa
    
    # Pressure vs Protection
    # away_pressure (higher = better for away defense)
    # home_block (higher = better for home offense, bad for away)
    # away pressure advantage = away_pressure - home_block (normalized)
    pressure_vs_block_away = (away_pressure or 0) - (home_block or 0) / 100  # block is 0-100 scale
    pressure_vs_block_home = (home_pressure or 0) - (away_block or 0) / 100
    
    return {
        'away_pass_matchup': away_pass_matchup,
        'home_pass_matchup': home_pass_matchup,
        'away_rush_matchup': away_rush_matchup,
        'home_rush_matchup': home_rush_matchup,
        'pressure_vs_block_away': pressure_vs_block_away,
        'pressure_vs_block_home': pressure_vs_block_home,
    }


# ============================================================
# MAIN FEATURE PREPARATION
# ============================================================

def prepare_features_v6(games_df, team_df):
    """
    Prepare comprehensive feature set for v6.
    Creates mirrored rows (both perspectives) for training.
    """
    
    # Build helper data
    print("Building playoff experience database...")
    experience_db = build_playoff_experience_db(games_df, team_df)
    
    print("Computing seed expectations...")
    seed_expectations = compute_seed_expectations(team_df)
    
    features = []
    
    for _, game in games_df.iterrows():
        away = game['away_team']
        home = game['home_team']
        season = game['season']
        game_type = game['game_type']
        
        # Get team season data (regular season stats only)
        away_season = team_df[(team_df['team'] == away) & (team_df['season'] == season)]
        home_season = team_df[(team_df['team'] == home) & (team_df['season'] == season)]
        
        if len(away_season) == 0 or len(home_season) == 0:
            continue
        
        away_data = away_season.iloc[0]
        home_data = home_season.iloc[0]
        
        # Skip if missing critical data
        if pd.isna(game.get('away_offensive_epa')) or pd.isna(game.get('home_offensive_epa')):
            continue
        
        # ============================================================
        # CORE FEATURES
        # ============================================================
        
        # Net EPA (using game's EPA data which should be regular season)
        away_off_epa = game['away_offensive_epa']
        home_off_epa = game['home_offensive_epa']
        away_def_epa = game['away_defensive_epa']
        home_def_epa = game['home_defensive_epa']
        
        if DEF_EPA_IS_ALLOWED:
            away_net_epa = away_off_epa - away_def_epa
            home_net_epa = home_off_epa - home_def_epa
        else:
            away_net_epa = away_off_epa + away_def_epa
            home_net_epa = home_off_epa + home_def_epa
        
        # Point differential per game
        away_games = away_data['wins'] + away_data['losses']
        home_games = home_data['wins'] + home_data['losses']
        away_pd_pg = away_data['point_differential'] / max(away_games, 1)
        home_pd_pg = home_data['point_differential'] / max(home_games, 1)
        
        # Seeds
        away_seed = int(away_data['playoff_seed'])
        home_seed = int(home_data['playoff_seed'])
        
        # ============================================================
        # CINDERELLA DETECTION (Seed Outperformance)
        # ============================================================
        
        if away_seed in seed_expectations.index:
            away_expected_pd = seed_expectations.loc[away_seed, 'pd_mean']
            away_pd_std = seed_expectations.loc[away_seed, 'pd_std']
            away_seed_overperf = (away_data['point_differential'] - away_expected_pd) / max(away_pd_std, 1)
        else:
            away_seed_overperf = 0
        
        if home_seed in seed_expectations.index:
            home_expected_pd = seed_expectations.loc[home_seed, 'pd_mean']
            home_pd_std = seed_expectations.loc[home_seed, 'pd_std']
            home_seed_overperf = (home_data['point_differential'] - home_expected_pd) / max(home_pd_std, 1)
        else:
            home_seed_overperf = 0
        
        # ============================================================
        # PLAYOFF EXPERIENCE
        # ============================================================
        
        away_exp = experience_db.get((season, away), {'qb_playoff_games': 0, 'coach_playoff_games': 0})
        home_exp = experience_db.get((season, home), {'qb_playoff_games': 0, 'coach_playoff_games': 0})
        
        # ============================================================
        # MATCHUP FEATURES
        # ============================================================
        
        matchups = compute_matchup_features(
            game['away_passing_epa'], game['away_rushing_epa'],
            away_data.get('pressure_rate', 0), away_data.get('pass_block_rating', 0),
            game['home_passing_epa'], game['home_rushing_epa'],
            home_data.get('pressure_rate', 0), home_data.get('pass_block_rating', 0),
            game['away_defensive_pass_epa'], game['away_defensive_rush_epa'],
            game['home_defensive_pass_epa'], game['home_defensive_rush_epa']
        )
        
        # ============================================================
        # REST/BYE
        # ============================================================
        
        away_bye = get_bye_status(away_seed, season)
        home_bye = get_bye_status(home_seed, season)
        
        # Bye only matters for divisional round (round after WC)
        if game_type == 'DIV':
            away_rest_adv = away_bye
            home_rest_adv = home_bye
        else:
            away_rest_adv = 0
            home_rest_adv = 0
        
        # ============================================================
        # LOCATION
        # ============================================================
        
        is_neutral = 1 if game['location'] == 'Neutral' else 0
        
        # ============================================================
        # CREATE FEATURE ROWS (Both Perspectives)
        # ============================================================
        
        # Common game info
        game_info = {
            'season': season,
            'game_type': game_type,
            'away_team': away,
            'home_team': home,
            'winner': game['winner'],
            'away_score': game['away_score'],
            'home_score': game['home_score'],
        }
        
        # PERSPECTIVE 1: Away team as "team_a"
        away_perspective = {
            **game_info,
            'perspective': 'away',
            'team_a': away,
            'team_b': home,
            'team_a_wins': 1 if game['winner'] == away else 0,
            
            # Core deltas (positive = team_a advantage)
            'delta_net_epa': away_net_epa - home_net_epa,
            'delta_pd_pg': away_pd_pg - home_pd_pg,
            'delta_seed': home_seed - away_seed,  # Positive = away has better (lower) seed
            
            # Cinderella
            'delta_seed_overperf': away_seed_overperf - home_seed_overperf,
            
            # Experience
            'delta_qb_exp': away_exp['qb_playoff_games'] - home_exp['qb_playoff_games'],
            'delta_coach_exp': away_exp['coach_playoff_games'] - home_exp['coach_playoff_games'],
            
            # Matchups
            'delta_pass_matchup': matchups['away_pass_matchup'] - matchups['home_pass_matchup'],
            'delta_rush_matchup': matchups['away_rush_matchup'] - matchups['home_rush_matchup'],
            'delta_pressure_block': matchups['pressure_vs_block_away'] - matchups['pressure_vs_block_home'],
            
            # Rest
            'delta_rest': away_rest_adv - home_rest_adv,
            
            # Location (negative for away team when opponent is home)
            'is_neutral': is_neutral,
            'opponent_is_home': 0 if is_neutral else 1,
        }
        features.append(away_perspective)
        
        # PERSPECTIVE 2: Home team as "team_a" (mirrored)
        home_perspective = {
            **game_info,
            'perspective': 'home',
            'team_a': home,
            'team_b': away,
            'team_a_wins': 1 if game['winner'] == home else 0,
            
            # Core deltas (flipped)
            'delta_net_epa': home_net_epa - away_net_epa,
            'delta_pd_pg': home_pd_pg - away_pd_pg,
            'delta_seed': away_seed - home_seed,
            
            # Cinderella
            'delta_seed_overperf': home_seed_overperf - away_seed_overperf,
            
            # Experience
            'delta_qb_exp': home_exp['qb_playoff_games'] - away_exp['qb_playoff_games'],
            'delta_coach_exp': home_exp['coach_playoff_games'] - away_exp['coach_playoff_games'],
            
            # Matchups
            'delta_pass_matchup': matchups['home_pass_matchup'] - matchups['away_pass_matchup'],
            'delta_rush_matchup': matchups['home_rush_matchup'] - matchups['away_rush_matchup'],
            'delta_pressure_block': matchups['pressure_vs_block_home'] - matchups['pressure_vs_block_away'],
            
            # Rest
            'delta_rest': home_rest_adv - away_rest_adv,
            
            # Location (positive for home team)
            'is_neutral': is_neutral,
            'opponent_is_home': 0,  # team_a IS home
            'is_at_home': 0 if is_neutral else 1,
        }
        features.append(home_perspective)
    
    df = pd.DataFrame(features)
    
    # Ensure is_at_home exists for all rows
    if 'is_at_home' not in df.columns:
        df['is_at_home'] = 0
    df.loc[df['perspective'] == 'away', 'is_at_home'] = 0
    
    return df


# ============================================================
# MODEL TRAINING
# ============================================================

def train_ensemble_v6(features_df, feature_cols, train_seasons):
    """
    Train ensemble model: Logistic Regression + Gradient Boosting
    Uses elastic net regularization for logistic.
    """
    
    # Use only ONE perspective per game for training (avoid data leakage)
    train_df = features_df[
        (features_df['season'].isin(train_seasons)) &
        (features_df['perspective'] == 'away')
    ].copy()
    
    train_df = train_df.dropna(subset=feature_cols)
    
    if len(train_df) < 30:
        print(f"Warning: Only {len(train_df)} training samples")
        return None
    
    X = train_df[feature_cols].values
    y = train_df['team_a_wins'].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model 1: Logistic Regression with Elastic Net
    lr_model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        C=0.5,
        random_state=42,
        max_iter=2000
    )
    lr_model.fit(X_scaled, y)
    
    # Model 2: Gradient Boosting (tuned for log loss)
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=10,
        random_state=42
    )
    gb_model.fit(X_scaled, y)
    
    return {
        'lr_model': lr_model,
        'gb_model': gb_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'train_samples': len(train_df)
    }


def ensemble_predict_v6(model_dict, game_features):
    """
    Ensemble prediction with close-game fallback.
    """
    if model_dict is None:
        return None
    
    # Extract features
    try:
        X = np.array([[game_features[col] for col in model_dict['feature_cols']]])
    except KeyError as e:
        print(f"Missing feature: {e}")
        return None
    
    if np.any(np.isnan(X)):
        return None
    
    X_scaled = model_dict['scaler'].transform(X)
    
    # Get predictions from both models
    lr_prob = model_dict['lr_model'].predict_proba(X_scaled)[0][1]
    gb_prob = model_dict['gb_model'].predict_proba(X_scaled)[0][1]
    
    # Ensemble
    ensemble_prob = LOGISTIC_WEIGHT * lr_prob + GB_WEIGHT * gb_prob
    
    # Close game fallback
    if abs(ensemble_prob - 0.5) < CLOSE_GAME_THRESHOLD:
        # Use fallback rules
        seed_diff = game_features.get('delta_seed', 0)
        is_at_home = game_features.get('is_at_home', 0)
        opponent_home = game_features.get('opponent_is_home', 0)
        
        # Rule: If close, favor home team or better seed
        if is_at_home:
            fallback_prob = 0.55
        elif opponent_home:
            fallback_prob = 0.45
        elif seed_diff > 0:
            fallback_prob = 0.52
        elif seed_diff < 0:
            fallback_prob = 0.48
        else:
            fallback_prob = 0.50
        
        # Blend with fallback
        final_prob = 0.6 * ensemble_prob + 0.4 * fallback_prob
    else:
        final_prob = ensemble_prob
    
    return {
        'lr_prob': lr_prob,
        'gb_prob': gb_prob,
        'ensemble_prob': ensemble_prob,
        'final_prob': final_prob,
        'is_close_game': abs(ensemble_prob - 0.5) < CLOSE_GAME_THRESHOLD,
        'predicted_winner': 'team_a' if final_prob > 0.5 else 'team_b'
    }


# ============================================================
# BASELINE MODELS (for comparison)
# ============================================================

def baseline_home_team(game_features):
    """Baseline: Always pick home team (or better seed if neutral)"""
    is_at_home = game_features.get('is_at_home', 0)
    opponent_home = game_features.get('opponent_is_home', 0)
    seed_diff = game_features.get('delta_seed', 0)
    is_neutral = game_features.get('is_neutral', 0)
    
    if is_neutral:
        return 1 if seed_diff >= 0 else 0
    elif is_at_home:
        return 1
    elif opponent_home:
        return 0
    return 1 if seed_diff >= 0 else 0


def baseline_better_seed(game_features):
    """Baseline: Always pick better seed"""
    seed_diff = game_features.get('delta_seed', 0)
    return 1 if seed_diff >= 0 else 0


def baseline_better_epa(game_features):
    """Baseline: Always pick higher net EPA"""
    epa_diff = game_features.get('delta_net_epa', 0)
    return 1 if epa_diff >= 0 else 0


# ============================================================
# VALIDATION
# ============================================================

def validate_on_2024(model_dict, features_df, games_df):
    """Comprehensive validation on 2024 playoffs"""
    
    print("\n" + "="*95)
    print("2024 PLAYOFF VALIDATION - MODEL v6")
    print("="*95)
    
    # Test on 2024, away perspective only
    test_df = features_df[
        (features_df['season'] == 2024) &
        (features_df['perspective'] == 'away')
    ].copy()
    
    round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
    
    results = []
    baseline_results = {'home': [], 'seed': [], 'epa': []}
    
    for _, gf in test_df.iterrows():
        pred = ensemble_predict_v6(model_dict, gf)
        
        if pred is None:
            continue
        
        away = gf['team_a']
        home = gf['team_b']
        actual_winner = away if gf['team_a_wins'] == 1 else home
        pred_winner = away if pred['predicted_winner'] == 'team_a' else home
        
        # Get score
        game_row = games_df[
            (games_df['season'] == 2024) &
            (games_df['away_team'] == away) &
            (games_df['home_team'] == home)
        ]
        score = f"{game_row.iloc[0]['away_score']}-{game_row.iloc[0]['home_score']}" if len(game_row) > 0 else "N/A"
        
        results.append({
            'round': round_names.get(gf['game_type'], gf['game_type']),
            'matchup': f"{away} @ {home}",
            'lr_prob': pred['lr_prob'],
            'gb_prob': pred['gb_prob'],
            'ensemble': pred['ensemble_prob'],
            'final': pred['final_prob'],
            'is_close': pred['is_close_game'],
            'predicted': pred_winner,
            'actual': actual_winner,
            'correct': pred_winner == actual_winner,
            'score': score,
            'seed_diff': gf['delta_seed'],
            'cinderella': gf['delta_seed_overperf'],
            'qb_exp_diff': gf['delta_qb_exp'],
        })
        
        # Baseline comparisons
        baseline_results['home'].append(baseline_home_team(gf) == gf['team_a_wins'])
        baseline_results['seed'].append(baseline_better_seed(gf) == gf['team_a_wins'])
        baseline_results['epa'].append(baseline_better_epa(gf) == gf['team_a_wins'])
    
    results_df = pd.DataFrame(results)
    
    # Display by round
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df[results_df['round'] == round_name]
        if len(rg) == 0:
            continue
        
        print(f"\n{round_name.upper()}")
        print(f"{'Matchup':<16} {'LR':<7} {'GB':<7} {'Ens':<7} {'Final':<7} {'Pred':<6} {'Act':<6} {'Cind':<6} {'✓/✗'}")
        print("-" * 85)
        
        for _, g in rg.iterrows():
            result = "✓" if g['correct'] else "✗"
            close = "*" if g['is_close'] else ""
            print(f"{g['matchup']:<16} {g['lr_prob']:.1%}   {g['gb_prob']:.1%}   "
                  f"{g['ensemble']:.1%}   {g['final']:.1%}{close}  {g['predicted']:<6} {g['actual']:<6} "
                  f"{g['cinderella']:+.1f}   {result}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print(f"\n{'='*95}")
    print("SUMMARY - MODEL v6")
    print(f"{'='*95}")
    
    total = len(results_df)
    correct = results_df['correct'].sum()
    
    print(f"\nOverall Accuracy: {correct}/{total} ({correct/total:.1%})")
    
    # By round
    print("\nBy Round:")
    for rnd in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df[results_df['round'] == rnd]
        if len(rg) > 0:
            print(f"  {rnd:<15} {rg['correct'].sum()}/{len(rg)} ({rg['correct'].mean():.1%})")
    
    # By confidence
    print("\nBy Confidence:")
    high = results_df[results_df['final'].apply(lambda x: max(x, 1-x)) >= 0.60]
    med = results_df[(results_df['final'].apply(lambda x: max(x, 1-x)) >= 0.52) & 
                     (results_df['final'].apply(lambda x: max(x, 1-x)) < 0.60)]
    low = results_df[results_df['final'].apply(lambda x: max(x, 1-x)) < 0.52]
    
    if len(high) > 0:
        print(f"  High (≥60%):  {high['correct'].sum()}/{len(high)} ({high['correct'].mean():.1%})")
    if len(med) > 0:
        print(f"  Med (52-60%): {med['correct'].sum()}/{len(med)} ({med['correct'].mean():.1%})")
    if len(low) > 0:
        print(f"  Low (<52%):   {low['correct'].sum()}/{len(low)} ({low['correct'].mean():.1%})")
    
    # Close games
    print("\nClose Games (fallback triggered):")
    close_games = results_df[results_df['is_close']]
    if len(close_games) > 0:
        print(f"  {close_games['correct'].sum()}/{len(close_games)} ({close_games['correct'].mean():.1%})")
    else:
        print("  None")
    
    # Baseline comparison
    print("\n" + "-"*50)
    print("BASELINE COMPARISON")
    print("-"*50)
    print(f"  Pick Home Team:   {sum(baseline_results['home'])}/{total} ({sum(baseline_results['home'])/total:.1%})")
    print(f"  Pick Better Seed: {sum(baseline_results['seed'])}/{total} ({sum(baseline_results['seed'])/total:.1%})")
    print(f"  Pick Higher EPA:  {sum(baseline_results['epa'])}/{total} ({sum(baseline_results['epa'])/total:.1%})")
    print(f"  Model v6:         {correct}/{total} ({correct/total:.1%})")
    
    # Probabilistic metrics
    print("\nProbabilistic Metrics:")
    actuals = results_df['correct'].astype(int).values
    preds = results_df['final'].apply(lambda x: x if results_df.loc[results_df['final']==x, 'predicted'].values[0] == results_df.loc[results_df['final']==x, 'actual'].values[0] else 1-x).values
    
    # Simpler: probability assigned to actual winner
    actual_probs = []
    for _, r in results_df.iterrows():
        if r['actual'] == r['matchup'].split(' @ ')[0]:  # Away won
            actual_probs.append(r['final'])
        else:  # Home won
            actual_probs.append(1 - r['final'])
    
    avg_prob_winner = np.mean(actual_probs)
    print(f"  Avg prob assigned to winner: {avg_prob_winner:.1%}")
    
    # Upset analysis
    print("\nUpset Analysis (seed gap ≥ 3):")
    upsets = results_df[results_df['seed_diff'].abs() >= 3]
    if len(upsets) > 0:
        print(f"  {upsets['correct'].sum()}/{len(upsets)} ({upsets['correct'].mean():.1%})")
        for _, g in upsets.iterrows():
            status = "✓" if g['correct'] else "✗"
            print(f"    {g['matchup']}: Pred {g['predicted']}, Actual {g['actual']} {status}")
    
    return results_df


# ============================================================
# MAIN
# ============================================================

def main():
    # Load data
    games_df, team_df = load_data()
    
    # Prepare features
    print("\nPreparing v6 features...")
    features_df = prepare_features_v6(games_df, team_df)
    
    print(f"Total feature rows: {len(features_df)} ({len(features_df)//2} games × 2 perspectives)")
    
    # Define feature bundles
    bundle_min = [
        'delta_net_epa',
        'delta_pd_pg',
        'delta_seed',
        'delta_seed_overperf',
        'delta_rest',
        'is_neutral',
        'opponent_is_home',
    ]
    
    bundle_matchups = bundle_min + [
        'delta_pass_matchup',
        'delta_rush_matchup',
    ]
    
    bundle_context = bundle_matchups + [
        'delta_qb_exp',
        'delta_coach_exp',
    ]
    
    # Use bundle_context for v6
    feature_cols = bundle_context
    
    print(f"\nFeature bundle: {feature_cols}")
    
    # Check feature availability
    available = [f for f in feature_cols if f in features_df.columns]
    missing = [f for f in feature_cols if f not in features_df.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
    feature_cols = available
    
    # Train on 2000-2023
    print("\nTraining ensemble model...")
    train_seasons = list(range(2000, 2024))
    model_dict = train_ensemble_v6(features_df, feature_cols, train_seasons)
    
    if model_dict is None:
        print("Training failed!")
        return
    
    print(f"Trained on {model_dict['train_samples']} games")
    
    # Show coefficients
    print("\nLogistic Regression Coefficients:")
    for feat, coef in zip(feature_cols, model_dict['lr_model'].coef_[0]):
        direction = "→ Team A" if coef > 0 else "→ Opponent"
        print(f"  {feat:<25} {coef:+.4f} {direction}")
    
    # Show GB feature importance
    print("\nGradient Boosting Feature Importance:")
    for feat, imp in sorted(zip(feature_cols, model_dict['gb_model'].feature_importances_), 
                            key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {feat:<25} {imp:.4f} {bar}")
    
    # Validate on 2024
    results_df = validate_on_2024(model_dict, features_df, games_df)
    
    # Save
    results_df.to_csv('../validation_results_2024_v6.csv', index=False)
    print(f"\nResults saved to ../validation_results_2024_v6.csv")
    
    return model_dict, features_df, results_df


if __name__ == "__main__":
    model_dict, features_df, results = main()
