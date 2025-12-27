"""
NFL Playoff Model Validation
Compares model predictions against actual 2024 playoff results
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load regular season and playoff results data"""
    teams_df = pd.read_csv('../all_seasons_data.csv')
    results_df = pd.read_csv('../playoff_results_2024.csv')
    
    print(f"Loaded {len(teams_df)} team-season records")
    print(f"Loaded {len(results_df)} playoff games from 2024")
    
    return teams_df, results_df


def prepare_features(df):
    """Prepare feature columns"""
    feature_cols = [
        'passing_epa', 'rushing_epa', 'total_offensive_epa',
        'defensive_epa', 'defensive_pass_epa', 'defensive_rush_epa',
        'win_pct', 'point_differential', 'net_epa',
        'last_5_win_pct', 'momentum_residual',
        'pass_rush_rating', 'sack_rate', 'pass_block_rating',
        'playoff_seed'
    ]
    return [c for c in feature_cols if c in df.columns]


def create_matchup_features(team_a_stats, team_b_stats, feature_cols):
    """Create differential features for a matchup"""
    differential = {}
    for col in feature_cols:
        if 'defensive' in col.lower() or 'allowed' in col.lower():
            differential[f'{col}_diff'] = team_b_stats[col] - team_a_stats[col]
        else:
            differential[f'{col}_diff'] = team_a_stats[col] - team_b_stats[col]
    return differential


def train_model(teams_df, feature_cols, train_seasons_end=2023):
    """
    Train model on historical data.
    For validating 2024 playoffs, train on data up to 2023.
    """
    # Generate training matchups from historical data
    matchups = []
    
    for season in teams_df[teams_df['season'] <= train_seasons_end]['season'].unique():
        season_teams = teams_df[teams_df['season'] == season]
        
        for conf in ['AFC', 'NFC']:
            conf_teams = season_teams[season_teams['conference'] == conf]
            teams_list = conf_teams.to_dict('records')
            
            for i, team_a in enumerate(teams_list):
                for team_b in teams_list[i+1:]:
                    diff = create_matchup_features(team_a, team_b, feature_cols)
                    
                    # Determine winner by playoff finish
                    finish_order = {'0.8-1.0': 5, '0.6-0.8': 4, '0.4-0.6': 3, 
                                   '0.2-0.4': 2, '0.0-0.2': 1}
                    
                    a_finish = finish_order.get(team_a.get('finish_bin', '0.0-0.2'), 0)
                    b_finish = finish_order.get(team_b.get('finish_bin', '0.0-0.2'), 0)
                    
                    if a_finish != b_finish:
                        winner = 1 if a_finish > b_finish else 0
                    else:
                        winner = 1 if team_a['playoff_seed'] < team_b['playoff_seed'] else 0
                    
                    matchup = {'team_a_wins': winner, **diff}
                    matchups.append(matchup)
    
    matchup_df = pd.DataFrame(matchups)
    diff_cols = [f'{col}_diff' for col in feature_cols]
    
    X = matchup_df[diff_cols].fillna(0)
    y = matchup_df['team_a_wins']
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    print(f"\nModel trained on {len(matchup_df)} matchups from seasons up to {train_seasons_end}")
    
    return model, diff_cols


def predict_matchup(model, teams_df, team_a, team_b, season, feature_cols, diff_cols):
    """Predict a single matchup"""
    team_a_data = teams_df[(teams_df['team'] == team_a) & (teams_df['season'] == season)]
    team_b_data = teams_df[(teams_df['team'] == team_b) & (teams_df['season'] == season)]
    
    if len(team_a_data) == 0 or len(team_b_data) == 0:
        return None
    
    team_a_stats = team_a_data.iloc[0].to_dict()
    team_b_stats = team_b_data.iloc[0].to_dict()
    
    diff = create_matchup_features(team_a_stats, team_b_stats, feature_cols)
    X = pd.DataFrame([diff])[diff_cols].fillna(0)
    
    prob = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]
    
    predicted_winner = team_a if prediction == 1 else team_b
    confidence = prob[1] if prediction == 1 else prob[0]
    
    return {
        'predicted_winner': predicted_winner,
        'confidence': confidence,
        f'{team_a}_prob': prob[1],
        f'{team_b}_prob': prob[0]
    }


def validate_against_2024_playoffs(model, teams_df, results_df, feature_cols, diff_cols):
    """Compare model predictions to actual 2024 playoff results"""
    
    print("\n" + "="*80)
    print("2024 PLAYOFF VALIDATION: MODEL PREDICTIONS vs ACTUAL RESULTS")
    print("="*80)
    
    # Map round codes to readable names
    round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
    
    results = []
    
    for _, game in results_df.iterrows():
        away = game['away_team']
        home = game['home_team']
        actual_winner = game['winner']
        round_name = round_names.get(game['game_type'], game['game_type'])
        
        # Get prediction (away team is team_a, home is team_b)
        pred = predict_matchup(model, teams_df, away, home, 2024, feature_cols, diff_cols)
        
        if pred is None:
            print(f"Warning: Could not predict {away} vs {home}")
            continue
        
        predicted_winner = pred['predicted_winner']
        confidence = pred['confidence']
        correct = predicted_winner == actual_winner
        
        results.append({
            'round': round_name,
            'matchup': f"{away} @ {home}",
            'predicted': predicted_winner,
            'actual': actual_winner,
            'confidence': confidence,
            'correct': correct,
            'away_score': game['away_score'],
            'home_score': game['home_score']
        })
    
    # Display results
    results_df_display = pd.DataFrame(results)
    
    # By round
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        round_games = results_df_display[results_df_display['round'] == round_name]
        if len(round_games) == 0:
            continue
            
        print(f"\n{round_name.upper()} ROUND")
        print("-" * 80)
        print(f"{'Matchup':<20} {'Predicted':<10} {'Actual':<10} {'Conf':<8} {'Score':<12} {'Result':<8}")
        print("-" * 80)
        
        for _, game in round_games.iterrows():
            result_str = "✓" if game['correct'] else "✗"
            score = f"{game['away_score']}-{game['home_score']}"
            print(f"{game['matchup']:<20} {game['predicted']:<10} {game['actual']:<10} "
                  f"{game['confidence']:.1%}    {score:<12} {result_str}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_correct = results_df_display['correct'].sum()
    total_games = len(results_df_display)
    overall_accuracy = total_correct / total_games
    
    print(f"\nOverall Accuracy: {total_correct}/{total_games} ({overall_accuracy:.1%})")
    
    # By round accuracy
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
        print(f"  High confidence (≥60%): {high_conf['correct'].sum()}/{len(high_conf)} "
              f"({high_conf['correct'].mean():.1%})")
    if len(low_conf) > 0:
        print(f"  Low confidence (<60%):  {low_conf['correct'].sum()}/{len(low_conf)} "
              f"({low_conf['correct'].mean():.1%})")
    
    # Upsets analysis
    print("\nUpset Analysis (away team won):")
    upsets = results_df_display[results_df_display['actual'] == results_df_display['matchup'].str.split(' @ ').str[0]]
    for _, game in upsets.iterrows():
        pred_status = "Predicted" if game['correct'] else "Missed"
        print(f"  {game['matchup']}: {game['actual']} won - {pred_status}")
    
    return results_df_display


def main():
    # Load data
    teams_df, results_df = load_data()
    
    # Prepare features
    feature_cols = prepare_features(teams_df)
    print(f"Using {len(feature_cols)} features")
    
    # Train model on data up to 2023 (to simulate predicting 2024 playoffs)
    model, diff_cols = train_model(teams_df, feature_cols, train_seasons_end=2023)
    
    # Validate against actual 2024 playoff results
    validation_results = validate_against_2024_playoffs(
        model, teams_df, results_df, feature_cols, diff_cols
    )
    
    # Save validation results
    validation_results.to_csv('../validation_results_2024.csv', index=False)
    print(f"\nValidation results saved to ../validation_results_2024.csv")
    
    return validation_results


if __name__ == "__main__":
    results = main()
