"""
NFL Playoff Model v2 - Improved Based on EDA Insights
Incorporates:
1. Balanced offense detection (positive in both pass AND rush EPA)
2. Championship zone indicator (good offense + good defense)
3. Recency weighting (recent seasons matter more)
4. Home field advantage
5. Evolved game features (pass rush emphasis)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load regular season and playoff results data"""
    teams_df = pd.read_csv('../all_seasons_data.csv')
    results_df = pd.read_csv('../playoff_results_2024.csv')
    
    print(f"Loaded {len(teams_df)} team-season records")
    print(f"Loaded {len(results_df)} playoff games from 2024")
    
    return teams_df, results_df


def engineer_features(df):
    """
    Create new features based on EDA insights
    """
    df = df.copy()
    
    # 1. BALANCED OFFENSE - positive in BOTH pass and rush EPA (rare, PHI 2024 style)
    df['balanced_elite_offense'] = (
        (df['passing_epa'] > 0) & (df['rushing_epa'] > 0)
    ).astype(int)
    
    # 2. CHAMPIONSHIP ZONE - good offense AND good defense
    # Good offense = positive total_offensive_epa
    # Good defense = negative defensive_epa (allowing fewer points than expected)
    df['championship_zone'] = (
        (df['total_offensive_epa'] > 0) & (df['defensive_epa'] < 0)
    ).astype(int)
    
    # 3. ELITE CHAMPIONSHIP ZONE - top quartile offense + defense
    off_75 = df['total_offensive_epa'].quantile(0.75)
    def_25 = df['defensive_epa'].quantile(0.25)  # Lower is better for defense
    df['elite_championship_zone'] = (
        (df['total_offensive_epa'] > off_75) & (df['defensive_epa'] < def_25)
    ).astype(int)
    
    # 4. OFFENSIVE BALANCE SCORE - how balanced is the offense
    # Teams with both positive or both close to each other are more balanced
    df['offensive_balance'] = 1 - abs(df['passing_epa'] - df['rushing_epa'])
    
    # 5. PASS-RUSH DOMINANCE (evolved game favors this)
    df['pass_rush_dominance'] = df['pass_rush_rating'] * df['sack_rate']
    
    # 6. PROTECTION QUALITY (inverse - lower sacks allowed is better)
    df['protection_quality'] = df['pass_block_rating'] * (1 - df.get('sacks_allowed_rate', 0.05))
    
    # 7. NET EPA (offense - defense, higher is better)
    # Already exists but let's make sure
    if 'net_epa' not in df.columns:
        df['net_epa'] = df['total_offensive_epa'] - df['defensive_epa']
    
    # 8. SEED ADVANTAGE (lower seed = better, invert for modeling)
    df['seed_advantage'] = 8 - df['playoff_seed']  # 1 seed -> 7, 7 seed -> 1
    
    # 9. POINT DIFFERENTIAL PER GAME (normalized)
    games_played = df['wins'] + df['losses'] + df.get('ties', 0)
    df['point_diff_per_game'] = df['point_differential'] / games_played.replace(0, 1)
    
    # 10. MOMENTUM SCORE (already exists as momentum_residual, but let's enhance)
    df['hot_streak'] = (df['last_5_win_pct'] >= 0.8).astype(int)
    
    return df


def prepare_features_v2(df):
    """Prepare enhanced feature columns"""
    
    # Original features
    base_features = [
        'passing_epa', 'rushing_epa', 'total_offensive_epa',
        'defensive_epa', 'defensive_pass_epa', 'defensive_rush_epa',
        'win_pct', 'point_differential', 'net_epa',
        'last_5_win_pct', 'momentum_residual',
        'pass_rush_rating', 'sack_rate', 'pass_block_rating',
        'playoff_seed'
    ]
    
    # New engineered features
    new_features = [
        'balanced_elite_offense',
        'championship_zone',
        'elite_championship_zone', 
        'offensive_balance',
        'pass_rush_dominance',
        'seed_advantage',
        'point_diff_per_game',
        'hot_streak'
    ]
    
    all_features = base_features + new_features
    return [c for c in all_features if c in df.columns]


def create_matchup_features_v2(team_a_stats, team_b_stats, feature_cols):
    """Create differential features for a matchup with special handling"""
    differential = {}
    
    for col in feature_cols:
        # Binary features - take the difference (1-0, 0-1, etc.)
        if col in ['balanced_elite_offense', 'championship_zone', 'elite_championship_zone', 'hot_streak']:
            differential[f'{col}_diff'] = team_a_stats.get(col, 0) - team_b_stats.get(col, 0)
        
        # Defensive stats - lower is better, flip sign
        elif 'defensive' in col.lower() or 'allowed' in col.lower():
            differential[f'{col}_diff'] = team_b_stats.get(col, 0) - team_a_stats.get(col, 0)
        
        # Seed - lower is better, flip sign  
        elif col == 'playoff_seed':
            differential[f'{col}_diff'] = team_b_stats.get(col, 4) - team_a_stats.get(col, 4)
        
        # All other stats - higher is better for team A
        else:
            differential[f'{col}_diff'] = team_a_stats.get(col, 0) - team_b_stats.get(col, 0)
    
    return differential


def train_model_v2(teams_df, feature_cols, train_seasons_end=2023, recency_weight=True):
    """
    Train model with optional recency weighting.
    Recent seasons weighted more heavily.
    """
    matchups = []
    sample_weights = []
    
    seasons = teams_df[teams_df['season'] <= train_seasons_end]['season'].unique()
    max_season = max(seasons)
    
    for season in seasons:
        season_teams = teams_df[teams_df['season'] == season]
        
        # Recency weight: exponential decay, most recent = 1.0
        if recency_weight:
            years_ago = max_season - season
            weight = np.exp(-0.1 * years_ago)  # e^(-0.1 * years_ago)
        else:
            weight = 1.0
        
        for conf in ['AFC', 'NFC']:
            conf_teams = season_teams[season_teams['conference'] == conf]
            teams_list = conf_teams.to_dict('records')
            
            for i, team_a in enumerate(teams_list):
                for team_b in teams_list[i+1:]:
                    diff = create_matchup_features_v2(team_a, team_b, feature_cols)
                    
                    # Determine winner by playoff finish
                    finish_order = {'0.8-1.0': 5, '0.6-0.8': 4, '0.4-0.6': 3, 
                                   '0.2-0.4': 2, '0.0-0.2': 1}
                    
                    a_finish = finish_order.get(team_a.get('finish_bin', '0.0-0.2'), 0)
                    b_finish = finish_order.get(team_b.get('finish_bin', '0.0-0.2'), 0)
                    
                    if a_finish != b_finish:
                        winner = 1 if a_finish > b_finish else 0
                    else:
                        winner = 1 if team_a['playoff_seed'] < team_b['playoff_seed'] else 0
                    
                    matchup = {'team_a_wins': winner, 'season': season, **diff}
                    matchups.append(matchup)
                    sample_weights.append(weight)
    
    matchup_df = pd.DataFrame(matchups)
    diff_cols = [f'{col}_diff' for col in feature_cols]
    
    X = matchup_df[diff_cols].fillna(0)
    y = matchup_df['team_a_wins']
    
    # Use Gradient Boosting - often better for this type of problem
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        random_state=42
    )
    
    # Fit with sample weights
    model.fit(X, y, sample_weight=sample_weights)
    
    print(f"\nModel v2 trained on {len(matchup_df)} matchups")
    print(f"Recency weighting: {'Enabled' if recency_weight else 'Disabled'}")
    print(f"Most recent season weight: 1.0, Oldest ({min(seasons)}) weight: {np.exp(-0.1 * (max_season - min(seasons))):.3f}")
    
    return model, diff_cols


def predict_matchup_v2(model, teams_df, team_a, team_b, season, feature_cols, diff_cols, home_team=None):
    """
    Predict a single matchup with optional home field advantage.
    home_team: which team (team_a or team_b) is at home
    """
    team_a_data = teams_df[(teams_df['team'] == team_a) & (teams_df['season'] == season)]
    team_b_data = teams_df[(teams_df['team'] == team_b) & (teams_df['season'] == season)]
    
    if len(team_a_data) == 0 or len(team_b_data) == 0:
        return None
    
    team_a_stats = team_a_data.iloc[0].to_dict()
    team_b_stats = team_b_data.iloc[0].to_dict()
    
    diff = create_matchup_features_v2(team_a_stats, team_b_stats, feature_cols)
    X = pd.DataFrame([diff])[diff_cols].fillna(0)
    
    prob = model.predict_proba(X)[0]
    
    # Apply home field advantage adjustment (historically ~3% boost)
    home_boost = 0.03
    if home_team == team_a:
        prob = [prob[0] - home_boost, prob[1] + home_boost]
    elif home_team == team_b:
        prob = [prob[0] + home_boost, prob[1] - home_boost]
    
    # Clip probabilities to valid range
    prob = np.clip(prob, 0.01, 0.99)
    prob = prob / prob.sum()  # Renormalize
    
    prediction = 1 if prob[1] > 0.5 else 0
    predicted_winner = team_a if prediction == 1 else team_b
    confidence = prob[1] if prediction == 1 else prob[0]
    
    return {
        'predicted_winner': predicted_winner,
        'confidence': confidence,
        f'{team_a}_prob': prob[1],
        f'{team_b}_prob': prob[0]
    }


def validate_v2(model, teams_df, results_df, feature_cols, diff_cols):
    """Compare model v2 predictions to actual 2024 playoff results"""
    
    print("\n" + "="*80)
    print("2024 PLAYOFF VALIDATION: MODEL v2 (IMPROVED) vs ACTUAL RESULTS")
    print("="*80)
    
    round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
    
    results = []
    
    for _, game in results_df.iterrows():
        away = game['away_team']
        home = game['home_team']
        actual_winner = game['winner']
        round_name = round_names.get(game['game_type'], game['game_type'])
        location = game['location']
        
        # Determine home team for home field advantage
        if location == 'Home':
            home_team = home
        else:
            home_team = None  # Neutral site (Super Bowl)
        
        pred = predict_matchup_v2(model, teams_df, away, home, 2024, feature_cols, diff_cols, home_team)
        
        if pred is None:
            print(f"Warning: Could not predict {away} vs {home}")
            continue
        
        predicted_winner = pred['predicted_winner']
        confidence = pred['confidence']
        correct = predicted_winner == actual_winner
        
        results.append({
            'round': round_name,
            'matchup': f"{away} @ {home}",
            'home_team': home_team if home_team else 'Neutral',
            'predicted': predicted_winner,
            'actual': actual_winner,
            'confidence': confidence,
            'correct': correct,
            'away_score': game['away_score'],
            'home_score': game['home_score']
        })
    
    results_df_display = pd.DataFrame(results)
    
    # Display results by round
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        round_games = results_df_display[results_df_display['round'] == round_name]
        if len(round_games) == 0:
            continue
            
        print(f"\n{round_name.upper()} ROUND")
        print("-" * 85)
        print(f"{'Matchup':<20} {'Home':<8} {'Predicted':<10} {'Actual':<10} {'Conf':<8} {'Score':<10} {'Result':<8}")
        print("-" * 85)
        
        for _, game in round_games.iterrows():
            result_str = "✓" if game['correct'] else "✗"
            score = f"{game['away_score']}-{game['home_score']}"
            home_str = game['home_team'][:7] if game['home_team'] != 'Neutral' else 'Neutral'
            print(f"{game['matchup']:<20} {home_str:<8} {game['predicted']:<10} {game['actual']:<10} "
                  f"{game['confidence']:.1%}    {score:<10} {result_str}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - MODEL v2")
    print("="*80)
    
    total_correct = results_df_display['correct'].sum()
    total_games = len(results_df_display)
    overall_accuracy = total_correct / total_games
    
    print(f"\nOverall Accuracy: {total_correct}/{total_games} ({overall_accuracy:.1%})")
    
    print("\nAccuracy by Round:")
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        round_games = results_df_display[results_df_display['round'] == round_name]
        if len(round_games) > 0:
            round_acc = round_games['correct'].mean()
            round_correct = round_games['correct'].sum()
            round_total = len(round_games)
            print(f"  {round_name:<15} {round_correct}/{round_total} ({round_acc:.1%})")
    
    # Confidence analysis
    print("\nConfidence Analysis:")
    high_conf = results_df_display[results_df_display['confidence'] >= 0.6]
    low_conf = results_df_display[results_df_display['confidence'] < 0.6]
    
    if len(high_conf) > 0:
        print(f"  High confidence (≥60%): {high_conf['correct'].sum()}/{len(high_conf)} ({high_conf['correct'].mean():.1%})")
    if len(low_conf) > 0:
        print(f"  Low confidence (<60%):  {low_conf['correct'].sum()}/{len(low_conf)} ({low_conf['correct'].mean():.1%})")
    
    return results_df_display


def compare_models(v1_results, v2_results):
    """Compare v1 and v2 model performance"""
    print("\n" + "="*80)
    print("MODEL COMPARISON: v1 (Original) vs v2 (Improved)")
    print("="*80)
    
    v1_acc = v1_results['correct'].mean()
    v2_acc = v2_results['correct'].mean()
    
    print(f"\nOverall Accuracy:")
    print(f"  Model v1: {v1_acc:.1%}")
    print(f"  Model v2: {v2_acc:.1%}")
    print(f"  Improvement: {(v2_acc - v1_acc)*100:+.1f} percentage points")
    
    print("\nGame-by-Game Comparison:")
    print("-" * 70)
    
    for i in range(len(v1_results)):
        v1 = v1_results.iloc[i]
        v2 = v2_results.iloc[i]
        
        v1_status = "✓" if v1['correct'] else "✗"
        v2_status = "✓" if v2['correct'] else "✗"
        
        changed = ""
        if v1['correct'] != v2['correct']:
            if v2['correct']:
                changed = " ← FIXED"
            else:
                changed = " ← BROKE"
        
        print(f"{v1['matchup']:<20} v1: {v1['predicted']:<4} {v1_status}  v2: {v2['predicted']:<4} {v2_status}{changed}")


def analyze_feature_importance(model, feature_cols):
    """Analyze which features matter most in v2"""
    diff_cols = [f'{col}_diff' for col in feature_cols]
    importance = pd.DataFrame({
        'feature': diff_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (Model v2)")
    print("="*80)
    
    print("\nTop 15 Features:")
    for i, row in importance.head(15).iterrows():
        bar = "█" * int(row['importance'] * 100)
        print(f"  {row['feature']:<35} {row['importance']:.4f} {bar}")
    
    return importance


def main():
    # Load data
    teams_df, results_df = load_data()
    
    # Engineer new features
    teams_df = engineer_features(teams_df)
    
    # Show PHI 2024's new features
    phi_2024 = teams_df[(teams_df['team'] == 'PHI') & (teams_df['season'] == 2024)]
    kc_2024 = teams_df[(teams_df['team'] == 'KC') & (teams_df['season'] == 2024)]
    
    print("\n" + "="*80)
    print("NEW ENGINEERED FEATURES - PHI vs KC (2024)")
    print("="*80)
    
    new_feats = ['balanced_elite_offense', 'championship_zone', 'elite_championship_zone', 
                 'offensive_balance', 'pass_rush_dominance', 'point_diff_per_game', 'hot_streak']
    
    print(f"\n{'Feature':<30} {'PHI':<10} {'KC':<10} {'Advantage':<10}")
    print("-" * 60)
    for feat in new_feats:
        if feat in phi_2024.columns:
            phi_val = phi_2024[feat].values[0]
            kc_val = kc_2024[feat].values[0]
            adv = "PHI" if phi_val > kc_val else ("KC" if kc_val > phi_val else "TIE")
            print(f"{feat:<30} {phi_val:<10.3f} {kc_val:<10.3f} {adv:<10}")
    
    # Prepare features
    feature_cols = prepare_features_v2(teams_df)
    print(f"\nUsing {len(feature_cols)} features (including {len(feature_cols) - 15} new engineered features)")
    
    # Train Model v2 with recency weighting
    model_v2, diff_cols = train_model_v2(teams_df, feature_cols, train_seasons_end=2023, recency_weight=True)
    
    # Validate
    v2_results = validate_v2(model_v2, teams_df, results_df, feature_cols, diff_cols)
    
    # Feature importance
    analyze_feature_importance(model_v2, feature_cols)
    
    # Save results
    v2_results.to_csv('../validation_results_2024_v2.csv', index=False)
    print(f"\nValidation results saved to ../validation_results_2024_v2.csv")
    
    return model_v2, teams_df, feature_cols, diff_cols, v2_results


if __name__ == "__main__":
    model_v2, teams_df, feature_cols, diff_cols, v2_results = main()
