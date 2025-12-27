"""
NFL Playoff Matchup Predictor
Predicts head-to-head playoff game outcomes using team differentials
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class NFLPlayoffMatchupPredictor:
    """
    Predicts playoff game outcomes based on team stat differentials.
    For a matchup, it calculates the difference between Team A and Team B stats,
    then predicts which team wins.
    """
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.scaler = None
        
    def load_data(self, filepath='../all_seasons_data.csv'):
        """Load the playoff teams data"""
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} team records")
        return self.df
    
    def create_matchup_features(self, team_a_stats, team_b_stats):
        """
        Create features for a matchup by calculating stat differentials.
        Positive values favor Team A, negative favor Team B.
        """
        differential = {}
        for col in self.feature_cols:
            # For defensive stats, lower is better, so we flip the sign
            if 'defensive' in col.lower() or 'allowed' in col.lower():
                # Team A having LOWER defensive EPA is better (they allow fewer points)
                differential[f'{col}_diff'] = team_b_stats[col] - team_a_stats[col]
            else:
                # For offensive stats and wins, higher is better for Team A
                differential[f'{col}_diff'] = team_a_stats[col] - team_b_stats[col]
        
        return differential
    
    def generate_historical_matchups(self, df):
        """
        Generate synthetic matchups from historical data.
        Since we don't have actual game-by-game playoff results,
        we create matchups between teams in the same season/conference.
        
        The target is whether the higher-seeded team won (based on actual results).
        """
        matchups = []
        
        # Group by season
        for season in df['season'].unique():
            season_teams = df[df['season'] == season]
            
            # For each conference
            for conf in ['AFC', 'NFC']:
                conf_teams = season_teams[season_teams['conference'] == conf]
                
                if len(conf_teams) < 2:
                    continue
                
                # Create matchups between teams
                teams_list = conf_teams.to_dict('records')
                
                for i, team_a in enumerate(teams_list):
                    for team_b in teams_list[i+1:]:
                        # Calculate differential features
                        diff_features = self.create_matchup_features(team_a, team_b)
                        
                        # Determine winner based on playoff finish
                        # Teams with better finish_bin went further in playoffs
                        finish_order = {'0.8-1.0': 5, '0.6-0.8': 4, '0.4-0.6': 3, 
                                       '0.2-0.4': 2, '0.0-0.2': 1}
                        
                        team_a_finish = finish_order.get(team_a.get('finish_bin', '0.0-0.2'), 0)
                        team_b_finish = finish_order.get(team_b.get('finish_bin', '0.0-0.2'), 0)
                        
                        # Team A wins if they went further, or if equal, by seed
                        if team_a_finish != team_b_finish:
                            winner = 1 if team_a_finish > team_b_finish else 0
                        else:
                            # Use seed as tiebreaker (lower seed = better)
                            winner = 1 if team_a['playoff_seed'] < team_b['playoff_seed'] else 0
                        
                        matchup = {
                            'season': season,
                            'team_a': team_a['team'],
                            'team_b': team_b['team'],
                            'team_a_seed': team_a['playoff_seed'],
                            'team_b_seed': team_b['playoff_seed'],
                            'team_a_wins': winner,
                            **diff_features
                        }
                        matchups.append(matchup)
        
        return pd.DataFrame(matchups)
    
    def train(self, df=None):
        """Train the matchup prediction model"""
        if df is None:
            df = self.df
        
        # Define features to use for matchup comparison
        self.feature_cols = [
            'passing_epa', 'rushing_epa', 'total_offensive_epa',
            'defensive_epa', 'defensive_pass_epa', 'defensive_rush_epa',
            'win_pct', 'point_differential', 'net_epa',
            'last_5_win_pct', 'momentum_residual',
            'pass_rush_rating', 'sack_rate', 'pass_block_rating'
        ]
        
        # Filter to available columns
        self.feature_cols = [c for c in self.feature_cols if c in df.columns]
        
        print(f"Using {len(self.feature_cols)} features for matchup prediction")
        
        # Generate matchup data
        matchup_df = self.generate_historical_matchups(df)
        print(f"Generated {len(matchup_df)} historical matchups")
        
        # Prepare features
        diff_cols = [f'{col}_diff' for col in self.feature_cols]
        X = matchup_df[diff_cols].fillna(0)
        y = matchup_df['team_a_wins']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        test_acc = accuracy_score(y_test, self.model.predict(X_test))
        
        print(f"\nModel Performance:")
        print(f"  Training Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy: {test_acc:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"  Cross-Val Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Feature importance
        print("\nTop Features for Predicting Matchup Winner:")
        importance = pd.DataFrame({
            'feature': diff_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.model
    
    def predict_matchup(self, team_a_name, team_b_name, season=2024):
        """Predict the outcome of a specific matchup"""
        # Get team stats
        team_a_data = self.df[(self.df['team'] == team_a_name) & 
                              (self.df['season'] == season)]
        team_b_data = self.df[(self.df['team'] == team_b_name) & 
                              (self.df['season'] == season)]
        
        if len(team_a_data) == 0:
            print(f"Team {team_a_name} not found in {season} data")
            return None
        if len(team_b_data) == 0:
            print(f"Team {team_b_name} not found in {season} data")
            return None
        
        team_a_stats = team_a_data.iloc[0].to_dict()
        team_b_stats = team_b_data.iloc[0].to_dict()
        
        # Create matchup features
        diff = self.create_matchup_features(team_a_stats, team_b_stats)
        diff_cols = [f'{col}_diff' for col in self.feature_cols]
        X_matchup = pd.DataFrame([diff])[diff_cols].fillna(0)
        
        # Predict
        prob = self.model.predict_proba(X_matchup)[0]
        prediction = self.model.predict(X_matchup)[0]
        
        winner = team_a_name if prediction == 1 else team_b_name
        confidence = prob[1] if prediction == 1 else prob[0]
        
        return {
            'team_a': team_a_name,
            'team_b': team_b_name,
            'predicted_winner': winner,
            'confidence': confidence,
            'team_a_win_prob': prob[1],
            'team_b_win_prob': prob[0]
        }
    
    def predict_playoff_bracket(self, season=2024):
        """
        Predict full playoff bracket for given season.
        Returns predictions for each round.
        """
        teams = self.df[self.df['season'] == season].sort_values('playoff_seed')
        
        if len(teams) == 0:
            print(f"No data for season {season}")
            return None
        
        print(f"\n{'='*60}")
        print(f"2025 NFL PLAYOFF PREDICTIONS (Based on {season} Regular Season)")
        print(f"{'='*60}")
        
        results = {'AFC': {}, 'NFC': {}}
        
        for conf in ['AFC', 'NFC']:
            conf_teams = teams[teams['conference'] == conf].sort_values('playoff_seed')
            team_list = conf_teams['team'].tolist()
            seeds = conf_teams['playoff_seed'].tolist()
            
            print(f"\n{conf} BRACKET:")
            print("-" * 40)
            
            # Display seeding
            print("Seeds:")
            for team, seed in zip(team_list, seeds):
                print(f"  {int(seed)}. {team}")
            
            # Wild Card Round (seeds 2-7 play, 1 has bye)
            # Matchups: 2 vs 7, 3 vs 6, 4 vs 5
            print(f"\nWild Card Round:")
            
            wc_winners = [team_list[0]]  # 1 seed gets bye
            matchups = [(1, 6), (2, 5), (3, 4)]  # indices: 2v7, 3v6, 4v5
            
            for high, low in matchups:
                if high < len(team_list) and low < len(team_list):
                    result = self.predict_matchup(team_list[high], team_list[low], season)
                    if result:
                        print(f"  ({int(seeds[high])}) {team_list[high]} vs ({int(seeds[low])}) {team_list[low]}")
                        print(f"      → {result['predicted_winner']} wins ({result['confidence']:.1%} confidence)")
                        wc_winners.append(result['predicted_winner'])
            
            results[conf]['wild_card'] = wc_winners
            
            # Divisional Round
            print(f"\nDivisional Round:")
            # 1 seed plays lowest remaining, higher seeds play each other
            div_winners = []
            
            if len(wc_winners) >= 4:
                # Reseed: 1 plays lowest, next two play each other
                # Simplified: 1 vs WC3 winner, WC1 vs WC2 winner
                div_matchups = [(wc_winners[0], wc_winners[3]), 
                               (wc_winners[1], wc_winners[2])]
                
                for team_a, team_b in div_matchups:
                    result = self.predict_matchup(team_a, team_b, season)
                    if result:
                        print(f"  {team_a} vs {team_b}")
                        print(f"      → {result['predicted_winner']} wins ({result['confidence']:.1%} confidence)")
                        div_winners.append(result['predicted_winner'])
            
            results[conf]['divisional'] = div_winners
            
            # Conference Championship
            print(f"\n{conf} Championship:")
            if len(div_winners) >= 2:
                result = self.predict_matchup(div_winners[0], div_winners[1], season)
                if result:
                    print(f"  {div_winners[0]} vs {div_winners[1]}")
                    print(f"      → {result['predicted_winner']} wins ({result['confidence']:.1%} confidence)")
                    results[conf]['champion'] = result['predicted_winner']
        
        # Super Bowl
        print(f"\n{'='*60}")
        print("SUPER BOWL PREDICTION")
        print("="*60)
        
        if 'champion' in results['AFC'] and 'champion' in results['NFC']:
            afc_champ = results['AFC']['champion']
            nfc_champ = results['NFC']['champion']
            
            sb_result = self.predict_matchup(afc_champ, nfc_champ, season)
            if sb_result:
                print(f"\n  {afc_champ} (AFC) vs {nfc_champ} (NFC)")
                print(f"\n  PREDICTED SUPER BOWL CHAMPION: {sb_result['predicted_winner']}")
                print(f"  Confidence: {sb_result['confidence']:.1%}")
                print(f"\n  Win Probabilities:")
                print(f"    {afc_champ}: {sb_result['team_a_win_prob']:.1%}")
                print(f"    {nfc_champ}: {sb_result['team_b_win_prob']:.1%}")
        
        return results


def main():
    # Initialize predictor
    predictor = NFLPlayoffMatchupPredictor()
    
    # Load data
    predictor.load_data('../all_seasons_data.csv')
    
    # Train model
    predictor.train()
    
    # Get available seasons
    seasons = predictor.df['season'].unique()
    latest_season = max(seasons)
    print(f"\nLatest season in data: {latest_season}")
    
    # Predict playoff bracket
    predictor.predict_playoff_bracket(season=latest_season)
    
    # Example: Predict a specific matchup
    print("\n" + "="*60)
    print("CUSTOM MATCHUP PREDICTIONS")
    print("="*60)
    
    # Get some teams from the latest season
    latest_teams = predictor.df[predictor.df['season'] == latest_season]['team'].tolist()
    
    if len(latest_teams) >= 2:
        print(f"\nAvailable teams in {latest_season}: {', '.join(latest_teams)}")
        
        # Predict a sample matchup
        result = predictor.predict_matchup(latest_teams[0], latest_teams[1], latest_season)
        if result:
            print(f"\nSample Matchup: {result['team_a']} vs {result['team_b']}")
            print(f"  Predicted Winner: {result['predicted_winner']}")
            print(f"  {result['team_a']} Win Prob: {result['team_a_win_prob']:.1%}")
            print(f"  {result['team_b']} Win Prob: {result['team_b_win_prob']:.1%}")
    
    return predictor


if __name__ == "__main__":
    predictor = main()
