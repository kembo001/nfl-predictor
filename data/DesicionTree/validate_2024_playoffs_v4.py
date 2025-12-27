"""
NFL Playoff Model v4 - Refined Feature Weights
Based on deeper analysis and domain expertise

Feature Weight Distribution:
1. Δ Net team strength (net_epa)           — 28%
2. Home field / Seed advantage             — 18%
3. Δ Point differential                    — 14%
4. Δ Passing efficiency                    — 10%
5. Δ Pass defense (EPA allowed)            — 8%
6. Δ Pressure / pass rush                  — 6%
7. Δ Overall defense (EPA allowed)         — 5%
8. Δ Pass protection                       — 5%
9. Δ Rushing efficiency                    — 3%
10. Δ Run defense (EPA allowed)            — 2%
11. Style/volume controls                  — 1%
12. Momentum                               — 0% (excluded - proven noise)

Sign Conventions:
- Positive differential = Team A is better
- For defensive EPA (allowed): LOWER is better, so we flip: B - A
- For playoff_seed: LOWER is better, so we flip: B - A
- For sacks_allowed_rate: LOWER is better, so we flip: B - A
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_CONFIG = {
    # Feature: (weight, higher_is_better)
    # If higher_is_better=False, we compute B - A instead of A - B
    
    # 1. Net team strength - 28%
    'net_epa': (0.28, True),
    
    # 2. Seed advantage - 18% (lower seed is better)
    'playoff_seed': (0.18, False),
    
    # 3. Point differential - 14%
    'point_differential': (0.14, True),
    
    # 4. Passing efficiency - 10%
    'passing_epa': (0.10, True),
    
    # 5. Pass defense - 8% (lower EPA allowed is better)
    'defensive_pass_epa': (0.08, False),
    
    # 6. Pressure/pass rush - 6%
    'pressure_rate': (0.06, True),
    
    # 7. Overall defense - 5% (lower EPA allowed is better)
    'defensive_epa': (0.05, False),
    
    # 8. Pass protection - 5%
    'pass_block_rating': (0.05, True),
    
    # 9. Rushing efficiency - 3%
    'rushing_epa': (0.03, True),
    
    # 10. Run defense - 2% (lower EPA allowed is better)
    'defensive_rush_epa': (0.02, False),
    
    # 11. Volume/style - 1% (split across metrics)
    'total_plays': (0.005, True),
    'total_pass_plays': (0.005, True),
}

# Home field advantage boost (applied separately)
HOME_FIELD_BOOST = 0.03


def load_data():
    """Load data"""
    teams_df = pd.read_csv('../all_seasons_data.csv')
    results_df = pd.read_csv('../playoff_results_2024.csv')
    print(f"Loaded {len(teams_df)} team-season records")
    print(f"Loaded {len(results_df)} playoff games")
    return teams_df, results_df


def validate_features(df):
    """Check which features are available"""
    available = {}
    missing = []
    
    for feat, (weight, _) in FEATURE_CONFIG.items():
        if feat in df.columns:
            available[feat] = FEATURE_CONFIG[feat]
        else:
            missing.append(feat)
    
    if missing:
        print(f"⚠ Missing features: {missing}")
    
    # Normalize weights for available features
    total_weight = sum(w for w, _ in available.values())
    normalized = {f: (w/total_weight, hib) for f, (w, hib) in available.items()}
    
    print(f"\nUsing {len(available)} features (weights normalized to sum=1.0)")
    return normalized


def compute_differentials(team_a_stats, team_b_stats, feature_config):
    """
    Compute feature differentials between two teams.
    Returns dict of differentials and weighted composite score.
    
    Sign convention:
    - Positive = Team A advantage
    - For higher_is_better=True: A - B
    - For higher_is_better=False: B - A
    """
    differentials = {}
    weighted_score = 0
    
    for feat, (weight, higher_is_better) in feature_config.items():
        a_val = team_a_stats.get(feat, 0) or 0
        b_val = team_b_stats.get(feat, 0) or 0
        
        if higher_is_better:
            diff = a_val - b_val
        else:
            diff = b_val - a_val  # Flip so positive = A advantage
        
        differentials[f'{feat}_diff'] = diff
        weighted_score += weight * diff
    
    differentials['weighted_score'] = weighted_score
    
    return differentials


def generate_training_data(teams_df, feature_config, max_season=2023):
    """Generate matchup training data from historical seasons"""
    matchups = []
    
    for season in teams_df[teams_df['season'] <= max_season]['season'].unique():
        season_teams = teams_df[teams_df['season'] == season]
        
        for conf in ['AFC', 'NFC']:
            conf_teams = season_teams[season_teams['conference'] == conf]
            teams_list = conf_teams.to_dict('records')
            
            for i, team_a in enumerate(teams_list):
                for team_b in teams_list[i+1:]:
                    diff = compute_differentials(team_a, team_b, feature_config)
                    
                    # Determine winner by playoff finish
                    finish_order = {'0.8-1.0': 5, '0.6-0.8': 4, '0.4-0.6': 3, 
                                   '0.2-0.4': 2, '0.0-0.2': 1}
                    
                    a_finish = finish_order.get(team_a.get('finish_bin', '0.0-0.2'), 0)
                    b_finish = finish_order.get(team_b.get('finish_bin', '0.0-0.2'), 0)
                    
                    if a_finish != b_finish:
                        winner = 1 if a_finish > b_finish else 0
                    else:
                        # Tiebreaker: better seed wins
                        winner = 1 if team_a.get('playoff_seed', 4) < team_b.get('playoff_seed', 4) else 0
                    
                    matchups.append({
                        'team_a_wins': winner,
                        'season': season,
                        **diff
                    })
    
    return pd.DataFrame(matchups)


def train_model_v4(teams_df, feature_config, max_season=2023):
    """Train v4 model"""
    
    matchup_df = generate_training_data(teams_df, feature_config, max_season)
    
    diff_cols = [f'{feat}_diff' for feat in feature_config.keys()]
    X = matchup_df[diff_cols].fillna(0)
    y = matchup_df['team_a_wins']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Logistic Regression
    model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    print(f"\n{'='*80}")
    print("MODEL v4 TRAINING RESULTS")
    print(f"{'='*80}")
    print(f"Training samples: {len(matchup_df)} matchups (2000-{max_season})")
    print(f"Training accuracy: {model.score(X_scaled, y):.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    
    # Coefficients
    print(f"\nLearned Coefficients (standardized):")
    
    # Extract original feature names (remove only the trailing '_diff')
    def get_original_feature(diff_col):
        if diff_col.endswith('_diff'):
            return diff_col[:-5]  # Remove last 5 characters ('_diff')
        return diff_col
    
    coef_df = pd.DataFrame({
        'feature': diff_cols,
        'coefficient': model.coef_[0],
        'config_weight': [feature_config[get_original_feature(f)][0] for f in diff_cols]
    }).sort_values('coefficient', ascending=False)
    
    print(f"{'Feature':<30} {'Coef':<10} {'Config Wt':<10} {'Direction':<15}")
    print("-" * 65)
    for _, row in coef_df.iterrows():
        direction = "→ Team A" if row['coefficient'] > 0 else "→ Team B"
        print(f"{row['feature']:<30} {row['coefficient']:+.4f}    {row['config_weight']:.3f}      {direction}")
    
    return {
        'model': model,
        'scaler': scaler,
        'diff_cols': diff_cols,
        'feature_config': feature_config
    }


def predict_matchup_v4(models, teams_df, team_a, team_b, season, home_team=None):
    """
    Predict matchup with home field advantage.
    
    Args:
        team_a: Away team (or first team listed)
        team_b: Home team (or second team listed)
        home_team: Which team has home field (team_a, team_b, or None for neutral)
    """
    team_a_data = teams_df[(teams_df['team'] == team_a) & (teams_df['season'] == season)]
    team_b_data = teams_df[(teams_df['team'] == team_b) & (teams_df['season'] == season)]
    
    if len(team_a_data) == 0 or len(team_b_data) == 0:
        return None
    
    team_a_stats = team_a_data.iloc[0].to_dict()
    team_b_stats = team_b_data.iloc[0].to_dict()
    
    # Compute differentials
    diff = compute_differentials(team_a_stats, team_b_stats, models['feature_config'])
    
    # Prepare for model
    X = pd.DataFrame([{k: v for k, v in diff.items() if k in models['diff_cols']}])[models['diff_cols']].fillna(0)
    X_scaled = models['scaler'].transform(X)
    
    # Get base probability
    prob = models['model'].predict_proba(X_scaled)[0]
    
    # Apply home field advantage
    if home_team == team_a:
        prob = [prob[0] - HOME_FIELD_BOOST, prob[1] + HOME_FIELD_BOOST]
    elif home_team == team_b:
        prob = [prob[0] + HOME_FIELD_BOOST, prob[1] - HOME_FIELD_BOOST]
    
    # Clip and normalize
    prob = np.clip(prob, 0.01, 0.99)
    prob = prob / prob.sum()
    
    predicted_winner = team_a if prob[1] > 0.5 else team_b
    confidence = max(prob)
    
    return {
        'predicted_winner': predicted_winner,
        'confidence': confidence,
        f'{team_a}_prob': prob[1],
        f'{team_b}_prob': prob[0],
        'weighted_score': diff['weighted_score'],
        'differentials': {k: v for k, v in diff.items() if k != 'weighted_score'}
    }


def validate_v4(models, teams_df, results_df):
    """Validate against 2024 playoffs"""
    
    print(f"\n{'='*90}")
    print("2024 PLAYOFF VALIDATION: MODEL v4")
    print(f"{'='*90}")
    
    round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
    
    results = []
    
    for _, game in results_df.iterrows():
        away = game['away_team']
        home = game['home_team']
        actual_winner = game['winner']
        round_name = round_names.get(game['game_type'], game['game_type'])
        location = game['location']
        
        # Determine home field
        home_team = home if location == 'Home' else None
        
        pred = predict_matchup_v4(models, teams_df, away, home, 2024, home_team)
        
        if pred is None:
            continue
        
        # Key differentials for display
        net_epa_diff = pred['differentials'].get('net_epa_diff', 0)
        seed_diff = pred['differentials'].get('playoff_seed_diff', 0)
        pd_diff = pred['differentials'].get('point_differential_diff', 0)
        
        results.append({
            'round': round_name,
            'matchup': f"{away} @ {home}",
            'home': home if location == 'Home' else 'Neutral',
            'predicted': pred['predicted_winner'],
            'actual': actual_winner,
            'confidence': pred['confidence'],
            'correct': pred['predicted_winner'] == actual_winner,
            f'{away}_prob': pred[f'{away}_prob'],
            f'{home}_prob': pred[f'{home}_prob'],
            'net_epa_diff': net_epa_diff,
            'seed_diff': seed_diff,
            'pd_diff': pd_diff,
            'score': f"{game['away_score']}-{game['home_score']}"
        })
    
    results_df_out = pd.DataFrame(results)
    
    # Display by round
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        round_games = results_df_out[results_df_out['round'] == round_name]
        if len(round_games) == 0:
            continue
        
        print(f"\n{round_name.upper()}")
        print("-" * 95)
        print(f"{'Matchup':<16} {'Home':<8} {'Pred':<5} {'Actual':<6} {'Conf':<7} {'Score':<8} "
              f"{'NetEPA':<8} {'Seed':<6} {'PD':<8} {'✓/✗':<4}")
        print("-" * 95)
        
        for _, g in round_games.iterrows():
            result = "✓" if g['correct'] else "✗"
            print(f"{g['matchup']:<16} {g['home']:<8} {g['predicted']:<5} {g['actual']:<6} "
                  f"{g['confidence']:.1%}   {g['score']:<8} "
                  f"{g['net_epa_diff']:+.3f}   {g['seed_diff']:+.0f}     {g['pd_diff']:+.0f}      {result}")
    
    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY - MODEL v4")
    print(f"{'='*90}")
    
    total_correct = results_df_out['correct'].sum()
    total_games = len(results_df_out)
    
    print(f"\nOverall Accuracy: {total_correct}/{total_games} ({total_correct/total_games:.1%})")
    
    print("\nBy Round:")
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df_out[results_df_out['round'] == round_name]
        if len(rg) > 0:
            rc = rg['correct'].sum()
            print(f"  {round_name:<15} {rc}/{len(rg)} ({rg['correct'].mean():.1%})")
    
    print("\nBy Confidence:")
    high = results_df_out[results_df_out['confidence'] >= 0.65]
    med = results_df_out[(results_df_out['confidence'] >= 0.55) & (results_df_out['confidence'] < 0.65)]
    low = results_df_out[results_df_out['confidence'] < 0.55]
    
    if len(high) > 0:
        print(f"  High (≥65%):    {high['correct'].sum()}/{len(high)} ({high['correct'].mean():.1%})")
    if len(med) > 0:
        print(f"  Medium (55-65%): {med['correct'].sum()}/{len(med)} ({med['correct'].mean():.1%})")
    if len(low) > 0:
        print(f"  Low (<55%):     {low['correct'].sum()}/{len(low)} ({low['correct'].mean():.1%})")
    
    # Missed predictions analysis
    print("\nMissed Predictions:")
    missed = results_df_out[~results_df_out['correct']]
    for _, g in missed.iterrows():
        print(f"  {g['matchup']}: Predicted {g['predicted']} ({g['confidence']:.1%}), "
              f"Actual {g['actual']} | NetEPA: {g['net_epa_diff']:+.3f}, Seed: {g['seed_diff']:+.0f}, PD: {g['pd_diff']:+.0f}")
    
    return results_df_out


def super_bowl_deep_dive(models, teams_df):
    """Detailed breakdown of Super Bowl prediction"""
    
    print(f"\n{'='*90}")
    print("SUPER BOWL DEEP DIVE: KC vs PHI")
    print(f"{'='*90}")
    
    kc = teams_df[(teams_df['team'] == 'KC') & (teams_df['season'] == 2024)].iloc[0].to_dict()
    phi = teams_df[(teams_df['team'] == 'PHI') & (teams_df['season'] == 2024)].iloc[0].to_dict()
    
    print(f"\n{'Feature':<25} {'Weight':<8} {'KC':<12} {'PHI':<12} {'Diff':<12} {'Contribution':<12} {'Adv':<6}")
    print("-" * 95)
    
    total_contribution = 0
    
    for feat, (weight, higher_is_better) in models['feature_config'].items():
        kc_val = kc.get(feat, 0) or 0
        phi_val = phi.get(feat, 0) or 0
        
        # KC is team_a (away), PHI is team_b (home) in Super Bowl
        if higher_is_better:
            diff = kc_val - phi_val
            adv = "KC" if diff > 0 else "PHI"
        else:
            diff = phi_val - kc_val  # Flipped so positive = KC advantage
            adv = "KC" if diff > 0 else "PHI"
        
        contribution = weight * diff
        total_contribution += contribution
        
        print(f"{feat:<25} {weight:<8.3f} {kc_val:<12.3f} {phi_val:<12.3f} {diff:+<12.3f} {contribution:+<12.4f} {adv:<6}")
    
    print("-" * 95)
    print(f"{'TOTAL':<25} {'1.000':<8} {'':<12} {'':<12} {'':<12} {total_contribution:+.4f}")
    
    # Get model prediction
    pred = predict_matchup_v4(models, teams_df, 'KC', 'PHI', 2024, home_team=None)
    
    print(f"\n{'='*50}")
    print("MODEL PREDICTION (Neutral Site)")
    print(f"{'='*50}")
    print(f"  Weighted Score: {pred['weighted_score']:+.4f} {'(favors KC)' if pred['weighted_score'] > 0 else '(favors PHI)'}")
    print(f"  KC Win Probability:  {pred['KC_prob']:.1%}")
    print(f"  PHI Win Probability: {pred['PHI_prob']:.1%}")
    print(f"  Predicted Winner: {pred['predicted_winner']}")
    print(f"  Confidence: {pred['confidence']:.1%}")
    print(f"\n  Actual Result: PHI 40, KC 22")


def model_comparison_summary():
    """Print comparison of all models"""
    print(f"\n{'='*90}")
    print("MODEL VERSION COMPARISON")
    print(f"{'='*90}")
    
    comparison = """
    Model    | Overall | Wild Card | Divisional | Conference | Super Bowl | Key Issue
    ---------|---------|-----------|------------|------------|------------|------------------
    v1       | 61.5%   | 66.7%     | 75.0%      | 50.0%      | 0%         | Overfit to momentum
    v2       | 76.9%   | 100%      | 75.0%      | 50.0%      | 0%         | last_5_win% dominated (noise)
    v3       | 69.2%   | 66.7%     | 75.0%      | 50.0%      | 100%       | Too much seed weight
    v4       | ???     | ???       | ???        | ???        | ???        | Balanced weights
    """
    print(comparison)


def main():
    # Load data
    teams_df, results_df = load_data()
    
    # Validate and normalize feature config
    feature_config = validate_features(teams_df)
    
    # Display weight distribution
    print(f"\n{'='*80}")
    print("FEATURE WEIGHTS (v4)")
    print(f"{'='*80}")
    print(f"{'Feature':<30} {'Weight':<10} {'Higher=Better':<15}")
    print("-" * 55)
    for feat, (weight, hib) in sorted(feature_config.items(), key=lambda x: -x[1][0]):
        bar = "█" * int(weight * 50)
        hib_str = "Yes" if hib else "No (flipped)"
        print(f"{feat:<30} {weight:.3f}      {hib_str:<15} {bar}")
    
    # Train model
    models = train_model_v4(teams_df, feature_config, max_season=2023)
    
    # Validate
    v4_results = validate_v4(models, teams_df, results_df)
    
    # Super Bowl deep dive
    super_bowl_deep_dive(models, teams_df)
    
    # Model comparison
    model_comparison_summary()
    
    # Save results
    v4_results.to_csv('../validation_results_2024_v4.csv', index=False)
    print(f"\nResults saved to ../validation_results_2024_v4.csv")
    
    return models, teams_df, v4_results


if __name__ == "__main__":
    models, teams_df, v4_results = main()
