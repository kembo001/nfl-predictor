"""
NFL Playoff Model v3 - Feature-Prioritized Model
Based on statistical analysis and domain knowledge:
- Removes momentum/last_5_win_pct (proven noise)
- Emphasizes quality metrics in priority order
- Uses weighted feature approach

Priority Features (most to least important):
1. playoff_seed - Historical SB winners are typically 1-2 seeds
2. point_differential - Best predictor of true team quality
3. passing_epa - Passing wins championships
4. defensive_pass_epa - Stopping the pass in playoffs
5. pressure_rate - Modern game demands pass rush
6. sacks_allowed_rate - Protecting your QB
7. total_offensive_epa - Overall efficiency
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# Feature priority weights (sum to 1.0)
FEATURE_WEIGHTS = {
    'playoff_seed': 0.25,           # 1st priority - seed matters most
    'point_differential': 0.22,      # 2nd priority - true team quality
    'passing_epa': 0.18,             # 3rd priority - passing wins
    'defensive_pass_epa': 0.15,      # 4th priority - stopping the pass
    'pressure_rate': 0.08,           # 5th priority - pass rush
    'sacks_allowed_rate': 0.07,      # 6th priority - protecting QB
    'total_offensive_epa': 0.05,     # 7th priority - overall efficiency
}

# Features where LOWER is better (need to flip sign for team A advantage)
LOWER_IS_BETTER = ['playoff_seed', 'defensive_pass_epa', 'sacks_allowed_rate']


def load_data():
    """Load data"""
    teams_df = pd.read_csv('../all_seasons_data.csv')
    results_df = pd.read_csv('../playoff_results_2024.csv')
    print(f"Loaded {len(teams_df)} team-season records")
    print(f"Loaded {len(results_df)} playoff games")
    return teams_df, results_df


def get_feature_cols(df):
    """Get available priority features"""
    priority_features = list(FEATURE_WEIGHTS.keys())
    available = [f for f in priority_features if f in df.columns]
    missing = [f for f in priority_features if f not in df.columns]
    
    if missing:
        print(f"Warning: Missing features: {missing}")
    
    print(f"Using {len(available)} priority features: {available}")
    return available


def create_weighted_matchup_score(team_a_stats, team_b_stats, feature_cols):
    """
    Create a weighted composite score for the matchup.
    Positive score = Team A favored
    Negative score = Team B favored
    
    Also returns individual feature differentials for the ML model.
    """
    differentials = {}
    weighted_score = 0
    
    for col in feature_cols:
        a_val = team_a_stats.get(col, 0)
        b_val = team_b_stats.get(col, 0)
        
        # Calculate raw differential
        if col in LOWER_IS_BETTER:
            # For these features, Team A wants LOWER value
            # So if A < B, that's good for A (positive differential)
            diff = b_val - a_val
        else:
            # For these features, Team A wants HIGHER value
            diff = a_val - b_val
        
        differentials[f'{col}_diff'] = diff
        
        # Add weighted contribution to composite score
        weight = FEATURE_WEIGHTS.get(col, 0)
        weighted_score += weight * diff
    
    differentials['weighted_score'] = weighted_score
    
    return differentials


def generate_training_matchups(teams_df, feature_cols, train_seasons_end=2023):
    """Generate matchup training data"""
    matchups = []
    
    for season in teams_df[teams_df['season'] <= train_seasons_end]['season'].unique():
        season_teams = teams_df[teams_df['season'] == season]
        
        for conf in ['AFC', 'NFC']:
            conf_teams = season_teams[season_teams['conference'] == conf]
            teams_list = conf_teams.to_dict('records')
            
            for i, team_a in enumerate(teams_list):
                for team_b in teams_list[i+1:]:
                    diff = create_weighted_matchup_score(team_a, team_b, feature_cols)
                    
                    # Determine winner by playoff finish
                    finish_order = {'0.8-1.0': 5, '0.6-0.8': 4, '0.4-0.6': 3, 
                                   '0.2-0.4': 2, '0.0-0.2': 1}
                    
                    a_finish = finish_order.get(team_a.get('finish_bin', '0.0-0.2'), 0)
                    b_finish = finish_order.get(team_b.get('finish_bin', '0.0-0.2'), 0)
                    
                    if a_finish != b_finish:
                        winner = 1 if a_finish > b_finish else 0
                    else:
                        winner = 1 if team_a['playoff_seed'] < team_b['playoff_seed'] else 0
                    
                    matchup = {
                        'team_a_wins': winner,
                        'season': season,
                        **diff
                    }
                    matchups.append(matchup)
    
    return pd.DataFrame(matchups)


def train_model_v3(teams_df, feature_cols, train_seasons_end=2023):
    """
    Train v3 model using two approaches:
    1. Weighted composite score (simple, interpretable)
    2. ML model with individual features (more flexible)
    """
    matchup_df = generate_training_matchups(teams_df, feature_cols, train_seasons_end)
    
    print(f"\nModel v3 trained on {len(matchup_df)} matchups (seasons 2000-{train_seasons_end})")
    
    # Approach 1: Simple weighted score model
    # If weighted_score > threshold, predict Team A wins
    # Find optimal threshold
    scores = matchup_df['weighted_score'].values
    y = matchup_df['team_a_wins'].values
    
    best_threshold = 0
    best_acc = 0
    for threshold in np.arange(-50, 50, 1):
        preds = (scores > threshold).astype(int)
        acc = (preds == y).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    print(f"Weighted Score Model: Best threshold = {best_threshold}, Training Acc = {best_acc:.3f}")
    
    # Approach 2: Logistic Regression on individual features (for probabilities)
    diff_cols = [f'{col}_diff' for col in feature_cols]
    X = matchup_df[diff_cols].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Logistic regression with L2 regularization
    lr_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    lr_model.fit(X_scaled, y)
    
    lr_acc = lr_model.score(X_scaled, y)
    print(f"Logistic Regression Model: Training Acc = {lr_acc:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"Logistic Regression CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    
    # Show coefficients (feature importance)
    print("\nLogistic Regression Coefficients (standardized):")
    coef_df = pd.DataFrame({
        'feature': diff_cols,
        'coefficient': lr_model.coef_[0],
        'abs_coef': np.abs(lr_model.coef_[0])
    }).sort_values('abs_coef', ascending=False)
    
    for _, row in coef_df.iterrows():
        direction = "→ Team A" if row['coefficient'] > 0 else "→ Team B"
        print(f"  {row['feature']:<30} {row['coefficient']:+.4f} {direction}")
    
    return {
        'weighted_threshold': best_threshold,
        'lr_model': lr_model,
        'scaler': scaler,
        'diff_cols': diff_cols
    }


def predict_matchup_v3(models, teams_df, team_a, team_b, season, feature_cols, home_team=None):
    """
    Predict using v3 model.
    Returns prediction from both weighted score and logistic regression.
    """
    team_a_data = teams_df[(teams_df['team'] == team_a) & (teams_df['season'] == season)]
    team_b_data = teams_df[(teams_df['team'] == team_b) & (teams_df['season'] == season)]
    
    if len(team_a_data) == 0 or len(team_b_data) == 0:
        return None
    
    team_a_stats = team_a_data.iloc[0].to_dict()
    team_b_stats = team_b_data.iloc[0].to_dict()
    
    # Get differentials
    diff = create_weighted_matchup_score(team_a_stats, team_b_stats, feature_cols)
    
    # Method 1: Weighted score
    weighted_score = diff['weighted_score']
    weighted_pred = 1 if weighted_score > models['weighted_threshold'] else 0
    
    # Method 2: Logistic regression probability
    diff_cols = models['diff_cols']
    X = pd.DataFrame([{k: v for k, v in diff.items() if k in diff_cols}])[diff_cols].fillna(0)
    X_scaled = models['scaler'].transform(X)
    
    prob = models['lr_model'].predict_proba(X_scaled)[0]
    
    # Apply home field advantage (~3% boost)
    home_boost = 0.03
    if home_team == team_a:
        prob = [prob[0] - home_boost, prob[1] + home_boost]
    elif home_team == team_b:
        prob = [prob[0] + home_boost, prob[1] - home_boost]
    
    prob = np.clip(prob, 0.01, 0.99)
    prob = prob / prob.sum()
    
    lr_pred = 1 if prob[1] > 0.5 else 0
    
    # Final prediction: Use LR (has probabilities)
    prediction = lr_pred
    predicted_winner = team_a if prediction == 1 else team_b
    confidence = prob[1] if prediction == 1 else prob[0]
    
    return {
        'predicted_winner': predicted_winner,
        'confidence': confidence,
        f'{team_a}_prob': prob[1],
        f'{team_b}_prob': prob[0],
        'weighted_score': weighted_score,
        'weighted_pred': team_a if weighted_pred == 1 else team_b,
        'feature_diffs': {k: v for k, v in diff.items() if k != 'weighted_score'}
    }


def validate_v3(models, teams_df, results_df, feature_cols):
    """Validate v3 model against 2024 playoffs"""
    
    print("\n" + "="*90)
    print("2024 PLAYOFF VALIDATION: MODEL v3 (QUALITY-FOCUSED)")
    print("="*90)
    
    round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
    
    results = []
    
    for _, game in results_df.iterrows():
        away = game['away_team']
        home = game['home_team']
        actual_winner = game['winner']
        round_name = round_names.get(game['game_type'], game['game_type'])
        location = game['location']
        
        home_team = home if location == 'Home' else None
        
        pred = predict_matchup_v3(models, teams_df, away, home, 2024, feature_cols, home_team)
        
        if pred is None:
            continue
        
        results.append({
            'round': round_name,
            'matchup': f"{away} @ {home}",
            'home_team': home_team if home_team else 'Neutral',
            'predicted': pred['predicted_winner'],
            'actual': actual_winner,
            'confidence': pred['confidence'],
            'correct': pred['predicted_winner'] == actual_winner,
            'weighted_pred': pred['weighted_pred'],
            'weighted_correct': pred['weighted_pred'] == actual_winner,
            'away_score': game['away_score'],
            'home_score': game['home_score'],
            **{k: f"{v:.3f}" for k, v in pred['feature_diffs'].items()}
        })
    
    results_df_display = pd.DataFrame(results)
    
    # Display by round
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        round_games = results_df_display[results_df_display['round'] == round_name]
        if len(round_games) == 0:
            continue
            
        print(f"\n{round_name.upper()} ROUND")
        print("-" * 90)
        print(f"{'Matchup':<18} {'Pred':<6} {'Actual':<6} {'Conf':<7} {'Score':<8} {'Result':<6} {'Key Differentials':<30}")
        print("-" * 90)
        
        for _, game in round_games.iterrows():
            result_str = "✓" if game['correct'] else "✗"
            score = f"{game['away_score']}-{game['home_score']}"
            
            # Show top differential
            seed_diff = float(game['playoff_seed_diff'])
            pd_diff = float(game['point_differential_diff'])
            key_diff = f"Seed:{seed_diff:+.0f} PD:{pd_diff:+.0f}"
            
            print(f"{game['matchup']:<18} {game['predicted']:<6} {game['actual']:<6} "
                  f"{game['confidence']:.1%}   {score:<8} {result_str:<6} {key_diff}")
    
    # Summary
    print("\n" + "="*90)
    print("SUMMARY - MODEL v3")
    print("="*90)
    
    total_correct = results_df_display['correct'].sum()
    total_games = len(results_df_display)
    overall_accuracy = total_correct / total_games
    
    weighted_correct = results_df_display['weighted_correct'].sum()
    
    print(f"\nLogistic Regression Accuracy: {total_correct}/{total_games} ({overall_accuracy:.1%})")
    print(f"Weighted Score Accuracy:      {weighted_correct}/{total_games} ({weighted_correct/total_games:.1%})")
    
    print("\nAccuracy by Round (LR Model):")
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        round_games = results_df_display[results_df_display['round'] == round_name]
        if len(round_games) > 0:
            round_acc = round_games['correct'].mean()
            round_correct = round_games['correct'].sum()
            print(f"  {round_name:<15} {round_correct}/{len(round_games)} ({round_acc:.1%})")
    
    # Confidence analysis
    print("\nConfidence Analysis:")
    high_conf = results_df_display[results_df_display['confidence'] >= 0.6]
    med_conf = results_df_display[(results_df_display['confidence'] >= 0.5) & (results_df_display['confidence'] < 0.6)]
    low_conf = results_df_display[results_df_display['confidence'] < 0.5]
    
    if len(high_conf) > 0:
        print(f"  High (≥60%):   {high_conf['correct'].sum()}/{len(high_conf)} ({high_conf['correct'].mean():.1%})")
    if len(med_conf) > 0:
        print(f"  Medium (50-60%): {med_conf['correct'].sum()}/{len(med_conf)} ({med_conf['correct'].mean():.1%})")
    if len(low_conf) > 0:
        print(f"  Low (<50%):    {low_conf['correct'].sum()}/{len(low_conf)}")
    
    return results_df_display


def analyze_super_bowl_prediction(models, teams_df, feature_cols):
    """Deep dive into Super Bowl prediction"""
    
    print("\n" + "="*90)
    print("SUPER BOWL DEEP DIVE: KC vs PHI")
    print("="*90)
    
    kc = teams_df[(teams_df['team'] == 'KC') & (teams_df['season'] == 2024)].iloc[0]
    phi = teams_df[(teams_df['team'] == 'PHI') & (teams_df['season'] == 2024)].iloc[0]
    
    print(f"\n{'Feature':<25} {'KC':<12} {'PHI':<12} {'Diff (PHI-KC)':<15} {'Weight':<10} {'Contribution':<12}")
    print("-" * 90)
    
    total_weighted = 0
    
    for col in feature_cols:
        kc_val = kc[col]
        phi_val = phi[col]
        weight = FEATURE_WEIGHTS.get(col, 0)
        
        if col in LOWER_IS_BETTER:
            # Lower is better, so PHI wants lower value
            # Diff for PHI advantage = KC - PHI (PHI lower = positive)
            diff = kc_val - phi_val
            contribution = weight * diff
        else:
            # Higher is better for PHI
            diff = phi_val - kc_val
            contribution = weight * diff
        
        total_weighted += contribution
        
        # Who has advantage?
        if col in LOWER_IS_BETTER:
            advantage = "PHI ✓" if phi_val < kc_val else "KC ✓"
        else:
            advantage = "PHI ✓" if phi_val > kc_val else "KC ✓"
        
        print(f"{col:<25} {kc_val:<12.3f} {phi_val:<12.3f} {diff:+<15.3f} {weight:<10.2f} {contribution:+.4f} {advantage}")
    
    print("-" * 90)
    print(f"{'TOTAL WEIGHTED SCORE':<25} {'':<12} {'':<12} {'':<15} {'':<10} {total_weighted:+.4f}")
    
    if total_weighted > 0:
        print(f"\n→ Weighted score favors PHI by {total_weighted:.4f}")
    else:
        print(f"\n→ Weighted score favors KC by {-total_weighted:.4f}")
    
    # Get LR prediction
    pred = predict_matchup_v3(models, teams_df, 'KC', 'PHI', 2024, feature_cols, home_team=None)
    print(f"\nLogistic Regression Prediction:")
    print(f"  KC Win Probability:  {pred['KC_prob']:.1%}")
    print(f"  PHI Win Probability: {pred['PHI_prob']:.1%}")
    print(f"  Predicted Winner: {pred['predicted_winner']}")
    print(f"  Actual Winner: PHI (40-22)")


def main():
    # Load data
    teams_df, results_df = load_data()
    
    # Get available priority features
    feature_cols = get_feature_cols(teams_df)
    
    print("\n" + "="*90)
    print("FEATURE PRIORITY WEIGHTS")
    print("="*90)
    for feat, weight in FEATURE_WEIGHTS.items():
        bar = "█" * int(weight * 40)
        status = "✓" if feat in feature_cols else "✗ MISSING"
        print(f"  {feat:<25} {weight:.2f} {bar} {status}")
    
    # Train model
    models = train_model_v3(teams_df, feature_cols, train_seasons_end=2023)
    
    # Validate
    v3_results = validate_v3(models, teams_df, results_df, feature_cols)
    
    # Super Bowl deep dive
    analyze_super_bowl_prediction(models, teams_df, feature_cols)
    
    # Save results
    v3_results.to_csv('../validation_results_2024_v3.csv', index=False)
    print(f"\nResults saved to ../validation_results_2024_v3.csv")
    
    return models, teams_df, feature_cols, v3_results


if __name__ == "__main__":
    models, teams_df, feature_cols, v3_results = main()
