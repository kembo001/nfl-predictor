"""
NFL Playoff Model v5 - Trained on Actual Playoff Games
Key improvements:
1. Train on REAL playoff games (not synthetic pairwise matchups)
2. Compact, non-redundant feature set
3. Learn home field advantage from data
4. Season-aware cross-validation (temporal splits)
5. Optimize for log loss (calibrated probabilities)

Features (Bundle A - compact, strong):
- delta_net_epa (offensive_epa - defensive_epa for each team, then diff)
- delta_point_diff_per_game (if available, else skip)
- seed_diff (lower = better, learned from data)
- is_home (binary indicator, learned from data)
- delta_pressure_rate (if available)
- delta_pass_block_rating (if available)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import warnings
warnings.filterwarnings('ignore')


def load_playoff_games():
    """Load actual playoff game results with EPA stats"""
    df = pd.read_csv('../nfl_playoff_results_2000_2024_with_epa.csv')
    print(f"Loaded {len(df)} actual playoff games (2000-2024)")
    print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"Games per type: {df['game_type'].value_counts().to_dict()}")
    return df


def load_team_season_data():
    """Load full team season data for additional features"""
    df = pd.read_csv('../all_seasons_data.csv')
    return df


def compute_net_epa(off_epa, def_epa):
    """
    Compute net EPA. 
    Offensive EPA: higher is better
    Defensive EPA (allowed): lower is better
    Net EPA = Offensive - Defensive (higher is better)
    """
    if pd.isna(off_epa) or pd.isna(def_epa):
        return np.nan
    return off_epa - def_epa


def prepare_game_features(games_df, team_df):
    """
    Prepare features for each playoff game.
    Creates delta features (away - home perspective, then we predict P(away wins))
    """
    
    features = []
    
    for _, game in games_df.iterrows():
        away = game['away_team']
        home = game['home_team']
        season = game['season']
        
        # Get team season data for additional features (seed, point diff, etc.)
        away_season = team_df[(team_df['team'] == away) & (team_df['season'] == season)]
        home_season = team_df[(team_df['team'] == home) & (team_df['season'] == season)]
        
        # Compute net EPA from playoff results data
        away_net_epa = compute_net_epa(game['away_offensive_epa'], game['away_defensive_epa'])
        home_net_epa = compute_net_epa(game['home_offensive_epa'], game['home_defensive_epa'])
        
        # Delta features (positive = away team advantage)
        feature_row = {
            'season': season,
            'game_type': game['game_type'],
            'away_team': away,
            'home_team': home,
            'winner': game['winner'],
            'away_score': game['away_score'],
            'home_score': game['home_score'],
            
            # Target: did away team win?
            'away_wins': 1 if game['winner'] == away else 0,
            
            # Location: is this a home game for home_team? (0 for neutral/Super Bowl)
            'is_home_game': 1 if game['location'] == 'Home' else 0,
            
            # Delta Net EPA (away - home, higher = away advantage)
            'delta_net_epa': away_net_epa - home_net_epa if not pd.isna(away_net_epa) and not pd.isna(home_net_epa) else np.nan,
            
            # Delta Passing EPA
            'delta_passing_epa': game['away_passing_epa'] - game['home_passing_epa'] if not pd.isna(game['away_passing_epa']) and not pd.isna(game['home_passing_epa']) else np.nan,
            
            # Delta Defensive Pass EPA (lower is better, so home - away for away advantage)
            'delta_def_pass_epa': game['home_defensive_pass_epa'] - game['away_defensive_pass_epa'] if not pd.isna(game['away_defensive_pass_epa']) and not pd.isna(game['home_defensive_pass_epa']) else np.nan,
            
            # Delta Rushing EPA
            'delta_rushing_epa': game['away_rushing_epa'] - game['home_rushing_epa'] if not pd.isna(game['away_rushing_epa']) and not pd.isna(game['home_rushing_epa']) else np.nan,
        }
        
        # Add features from team season data if available
        if len(away_season) > 0 and len(home_season) > 0:
            away_data = away_season.iloc[0]
            home_data = home_season.iloc[0]
            
            # Seed diff (lower is better, so home - away for away advantage)
            feature_row['delta_seed'] = home_data['playoff_seed'] - away_data['playoff_seed']
            
            # Point differential
            feature_row['delta_point_diff'] = away_data['point_differential'] - home_data['point_differential']
            
            # Point diff per game (normalized)
            away_games = away_data['wins'] + away_data['losses']
            home_games = home_data['wins'] + home_data['losses']
            if away_games > 0 and home_games > 0:
                feature_row['delta_point_diff_pg'] = (away_data['point_differential'] / away_games) - (home_data['point_differential'] / home_games)
            
            # Pressure rate (if available)
            if 'pressure_rate' in away_data and 'pressure_rate' in home_data:
                feature_row['delta_pressure_rate'] = away_data['pressure_rate'] - home_data['pressure_rate']
            
            # Pass block rating (if available)
            if 'pass_block_rating' in away_data and 'pass_block_rating' in home_data:
                feature_row['delta_pass_block'] = away_data['pass_block_rating'] - home_data['pass_block_rating']
        
        features.append(feature_row)
    
    return pd.DataFrame(features)


def train_v5_model(features_df, feature_cols, train_seasons):
    """
    Train v5 model on specified seasons.
    Returns model, scaler, and training metrics.
    """
    # Filter to training seasons
    train_df = features_df[features_df['season'].isin(train_seasons)].copy()
    
    # Drop rows with missing features
    train_df = train_df.dropna(subset=feature_cols)
    
    if len(train_df) < 20:
        print(f"Warning: Only {len(train_df)} training samples after dropping NAs")
        return None
    
    X = train_df[feature_cols].values
    y = train_df['away_wins'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Logistic Regression with moderate regularization
    model = LogisticRegression(C=0.5, random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'train_samples': len(train_df),
        'train_seasons': train_seasons
    }


def predict_game(model_dict, game_features):
    """Predict probability that away team wins"""
    if model_dict is None:
        return None
    
    X = np.array([[game_features[col] for col in model_dict['feature_cols']]])
    
    # Check for NaN
    if np.any(np.isnan(X)):
        return None
    
    X_scaled = model_dict['scaler'].transform(X)
    prob = model_dict['model'].predict_proba(X_scaled)[0]
    
    return {
        'away_win_prob': prob[1],
        'home_win_prob': prob[0],
        'predicted_winner': 'away' if prob[1] > 0.5 else 'home'
    }


def temporal_cross_validation(features_df, feature_cols):
    """
    Rolling temporal CV: train on seasons up to N, test on season N+1
    This simulates real prediction scenario.
    """
    seasons = sorted(features_df['season'].unique())
    
    results = []
    all_preds = []
    all_actuals = []
    
    # Start from season with enough training data
    for test_season in seasons[5:]:  # Need at least 5 seasons for training
        train_seasons = [s for s in seasons if s < test_season]
        
        # Train model
        model_dict = train_v5_model(features_df, feature_cols, train_seasons)
        
        if model_dict is None:
            continue
        
        # Test on test_season
        test_df = features_df[features_df['season'] == test_season].dropna(subset=feature_cols)
        
        season_correct = 0
        season_total = 0
        season_preds = []
        season_actuals = []
        
        for _, game in test_df.iterrows():
            pred = predict_game(model_dict, game)
            if pred is None:
                continue
            
            actual_away_win = game['away_wins']
            predicted_away_win = 1 if pred['away_win_prob'] > 0.5 else 0
            
            correct = (predicted_away_win == actual_away_win)
            season_correct += int(correct)
            season_total += 1
            
            season_preds.append(pred['away_win_prob'])
            season_actuals.append(actual_away_win)
            all_preds.append(pred['away_win_prob'])
            all_actuals.append(actual_away_win)
        
        if season_total > 0:
            results.append({
                'season': test_season,
                'accuracy': season_correct / season_total,
                'correct': season_correct,
                'total': season_total,
                'log_loss': log_loss(season_actuals, season_preds) if len(set(season_actuals)) > 1 else np.nan,
                'brier': brier_score_loss(season_actuals, season_preds)
            })
    
    return pd.DataFrame(results), all_preds, all_actuals


def main():
    # Load data
    games_df = load_playoff_games()
    team_df = load_team_season_data()
    
    # Prepare features
    print("\nPreparing game features...")
    features_df = prepare_game_features(games_df, team_df)
    
    print(f"Total games with features: {len(features_df)}")
    print(f"Games with complete EPA data: {features_df['delta_net_epa'].notna().sum()}")
    
    # Define feature sets to test
    
    # Bundle A: Compact (recommended)
    bundle_a = ['delta_net_epa', 'delta_seed', 'delta_point_diff_pg', 'is_home_game']
    
    # Bundle B: With pass focus
    bundle_b = ['delta_net_epa', 'delta_seed', 'delta_point_diff_pg', 'is_home_game', 
                'delta_passing_epa', 'delta_def_pass_epa']
    
    # Bundle C: Kitchen sink (but non-redundant)
    bundle_c = ['delta_net_epa', 'delta_seed', 'delta_point_diff_pg', 'is_home_game',
                'delta_passing_epa', 'delta_def_pass_epa', 'delta_rushing_epa']
    
    print("\n" + "="*80)
    print("TEMPORAL CROSS-VALIDATION RESULTS")
    print("="*80)
    
    for name, feature_cols in [('Bundle A (Compact)', bundle_a), 
                                ('Bundle B (Pass Focus)', bundle_b),
                                ('Bundle C (Extended)', bundle_c)]:
        
        # Check which features are available
        available = [f for f in feature_cols if f in features_df.columns]
        missing = [f for f in feature_cols if f not in features_df.columns]
        
        if missing:
            print(f"\n{name}: Missing features {missing}, skipping")
            continue
        
        # Drop rows with NaN in these features
        valid_df = features_df.dropna(subset=available)
        
        print(f"\n{name}")
        print(f"Features: {available}")
        print(f"Valid games: {len(valid_df)}")
        print("-" * 60)
        
        # Temporal CV
        cv_results, all_preds, all_actuals = temporal_cross_validation(features_df, available)
        
        if len(cv_results) == 0:
            print("No valid CV results")
            continue
        
        # Summary stats
        total_correct = cv_results['correct'].sum()
        total_games = cv_results['total'].sum()
        
        print(f"Overall Accuracy: {total_correct}/{total_games} ({total_correct/total_games:.1%})")
        print(f"Mean Log Loss: {cv_results['log_loss'].mean():.4f}")
        print(f"Mean Brier Score: {cv_results['brier'].mean():.4f}")
        
        print(f"\nPer-Season Results:")
        for _, row in cv_results.iterrows():
            print(f"  {int(row['season'])}: {row['correct']:.0f}/{row['total']:.0f} ({row['accuracy']:.1%}) | LogLoss: {row['log_loss']:.3f}")
    
    # Train final model on all data up to 2023, validate on 2024
    print("\n" + "="*80)
    print("FINAL MODEL: VALIDATE ON 2024 PLAYOFFS")
    print("="*80)
    
    # Use Bundle A for final model
    final_features = ['delta_net_epa', 'delta_seed', 'delta_point_diff_pg', 'is_home_game']
    
    # Check availability
    final_features = [f for f in final_features if f in features_df.columns]
    
    train_seasons = list(range(2000, 2024))
    model_dict = train_v5_model(features_df, final_features, train_seasons)
    
    if model_dict:
        print(f"\nTrained on {model_dict['train_samples']} games from 2000-2023")
        print(f"Features: {final_features}")
        
        # Show coefficients
        print(f"\nLearned Coefficients:")
        for feat, coef in zip(final_features, model_dict['model'].coef_[0]):
            direction = "→ Away team" if coef > 0 else "→ Home team"
            print(f"  {feat:<25} {coef:+.4f} {direction}")
        print(f"  {'Intercept':<25} {model_dict['model'].intercept_[0]:+.4f}")
        
        # Validate on 2024
        print(f"\n2024 PLAYOFF PREDICTIONS:")
        print("-" * 90)
        
        test_2024 = features_df[features_df['season'] == 2024].copy()
        
        round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
        
        results = []
        
        for _, game in test_2024.iterrows():
            pred = predict_game(model_dict, game)
            
            if pred is None:
                print(f"  {game['away_team']} @ {game['home_team']}: Missing data")
                continue
            
            actual_winner = game['winner']
            pred_winner = game['away_team'] if pred['predicted_winner'] == 'away' else game['home_team']
            correct = (pred_winner == actual_winner)
            
            # Get probabilities for display
            away_prob = pred['away_win_prob']
            home_prob = pred['home_win_prob']
            
            results.append({
                'round': round_names.get(game['game_type'], game['game_type']),
                'matchup': f"{game['away_team']} @ {game['home_team']}",
                'predicted': pred_winner,
                'actual': actual_winner,
                'confidence': max(away_prob, home_prob),
                'away_prob': away_prob,
                'home_prob': home_prob,
                'correct': correct,
                'score': f"{game['away_score']}-{game['home_score']}"
            })
        
        results_df = pd.DataFrame(results)
        
        # Display by round
        for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
            round_games = results_df[results_df['round'] == round_name]
            if len(round_games) == 0:
                continue
            
            print(f"\n{round_name.upper()}")
            print(f"{'Matchup':<16} {'Away%':<8} {'Home%':<8} {'Pred':<6} {'Actual':<6} {'Score':<10} {'✓/✗'}")
            print("-" * 75)
            
            for _, g in round_games.iterrows():
                result = "✓" if g['correct'] else "✗"
                print(f"{g['matchup']:<16} {g['away_prob']:.1%}    {g['home_prob']:.1%}    "
                      f"{g['predicted']:<6} {g['actual']:<6} {g['score']:<10} {result}")
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY - MODEL v5")
        print(f"{'='*80}")
        
        total_correct = results_df['correct'].sum()
        total_games = len(results_df)
        
        print(f"\nOverall Accuracy: {total_correct}/{total_games} ({total_correct/total_games:.1%})")
        
        print("\nBy Round:")
        for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
            rg = results_df[results_df['round'] == round_name]
            if len(rg) > 0:
                rc = rg['correct'].sum()
                print(f"  {round_name:<15} {rc}/{len(rg)} ({rg['correct'].mean():.1%})")
        
        # Calibration check
        print("\nCalibration Check:")
        high_conf = results_df[results_df['confidence'] >= 0.65]
        med_conf = results_df[(results_df['confidence'] >= 0.55) & (results_df['confidence'] < 0.65)]
        low_conf = results_df[results_df['confidence'] < 0.55]
        
        if len(high_conf) > 0:
            print(f"  High conf (≥65%):  {high_conf['correct'].sum()}/{len(high_conf)} ({high_conf['correct'].mean():.1%})")
        if len(med_conf) > 0:
            print(f"  Med conf (55-65%): {med_conf['correct'].sum()}/{len(med_conf)} ({med_conf['correct'].mean():.1%})")
        if len(low_conf) > 0:
            print(f"  Low conf (<55%):   {low_conf['correct'].sum()}/{len(low_conf)} ({low_conf['correct'].mean():.1%})")
        
        # Log loss and Brier
        if len(results_df) > 0:
            # For log loss, we need to align predictions with actual outcomes
            actuals = results_df['correct'].astype(int)  # 1 if we predicted correctly
            # Actually, we need P(actual winner)
            actual_probs = results_df.apply(
                lambda r: r['away_prob'] if r['actual'] == r['matchup'].split(' @ ')[0] else r['home_prob'], 
                axis=1
            )
            
            print(f"\nProbabilistic Metrics:")
            print(f"  Log Loss: {log_loss([1]*len(actual_probs), actual_probs):.4f}")
            print(f"  Brier Score: {brier_score_loss([1]*len(actual_probs), actual_probs):.4f}")
        
        # Save results
        results_df.to_csv('../validation_results_2024_v5.csv', index=False)
        print(f"\nResults saved to ../validation_results_2024_v5.csv")
        
        # Return for further analysis
        return model_dict, features_df, results_df


if __name__ == "__main__":
    model_dict, features_df, results = main()
