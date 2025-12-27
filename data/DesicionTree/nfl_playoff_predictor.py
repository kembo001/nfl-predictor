"""
NFL Playoff Predictor using Decision Tree and Random Forest
Uses historical playoff team data from 2000-2024 to predict outcomes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

def load_data(filepath='../all_seasons_data.csv'):
    """Load NFL playoff team data"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} playoff team records from {df['season'].min()}-{df['season'].max()}")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_features(df):
    """
    Select and prepare features for the model.
    Returns X (features) and y (target)
    """
    # Define feature columns - these are the stats that predict playoff success
    feature_cols = [
        # Offensive metrics
        'passing_epa',
        'rushing_epa', 
        'total_offensive_epa',
        
        # Defensive metrics
        'defensive_epa',
        'defensive_pass_epa',
        'defensive_rush_epa',
        
        # Overall team performance
        'win_pct',
        'point_differential',
        'net_epa',
        
        # Momentum/hot streak metrics
        'last_5_win_pct',
        'momentum_residual',
        
        # Pass rush and protection
        'pass_rush_rating',
        'sack_rate',
        'pass_block_rating',
        
        # Playoff seeding
        'playoff_seed',
    ]
    
    # Filter to only columns that exist in the dataframe
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"\nUsing {len(available_features)} features: {available_features}")
    
    X = df[available_features].copy()
    y = df['champion'].copy()  # 1 = Super Bowl winner, 0 = not
    
    # Handle any missing values
    X = X.fillna(X.median())
    
    return X, y, available_features


# ============================================================
# 2. TRAIN MODELS
# ============================================================

def train_decision_tree(X_train, y_train, max_depth=5):
    """Train a Decision Tree classifier"""
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced classes (few champions vs many non-champions)
    )
    dt_model.fit(X_train, y_train)
    return dt_model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """Train a Random Forest classifier"""
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  # Use all CPU cores
    )
    rf_model.fit(X_train, y_train)
    return rf_model


# ============================================================
# 3. EVALUATE MODELS
# ============================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]  # Probability of being champion
    
    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Non-Champion', 'Champion']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    return predictions, proba


def cross_validate_model(model, X, y, cv=5, model_name="Model"):
    """Perform cross-validation"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\n{model_name} Cross-Validation (k={cv}):")
    print(f"  Mean Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    return scores


def plot_feature_importance(model, feature_names, model_name="Model"):
    """Plot feature importance"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name} - Feature Importance")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'../feature_importance_{model_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()
    
    print(f"\n{model_name} Feature Importance Ranking:")
    for i, idx in enumerate(indices[:10]):
        print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")


def visualize_decision_tree(model, feature_names):
    """Visualize the decision tree"""
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=feature_names,
              class_names=['Non-Champion', 'Champion'],
              filled=True,
              rounded=True,
              fontsize=8)
    plt.title("Decision Tree for NFL Playoff Champion Prediction")
    plt.tight_layout()
    plt.savefig('../decision_tree_visualization.png', dpi=150)
    plt.show()


# ============================================================
# 4. PREDICT 2025 PLAYOFFS
# ============================================================

def predict_2025_playoffs(model, df, feature_cols, year=2024):
    """
    Predict championship probability for current playoff teams.
    Uses most recent season data (2024 regular season for 2025 playoffs)
    """
    # Get the most recent season's playoff teams
    current_teams = df[df['season'] == year].copy()
    
    if len(current_teams) == 0:
        print(f"No data found for season {year}")
        return None
    
    print(f"\n{'='*50}")
    print(f"2025 Playoff Predictions (based on {year} regular season)")
    print(f"{'='*50}")
    
    X_current = current_teams[feature_cols].fillna(current_teams[feature_cols].median())
    proba = model.predict_proba(X_current)[:, 1]
    
    # Add predictions to dataframe
    current_teams['champion_probability'] = proba
    
    # Sort by probability
    results = current_teams[['team', 'conference', 'playoff_seed', 'win_pct', 
                             'point_differential', 'champion_probability']].sort_values(
                             'champion_probability', ascending=False)
    
    print("\nTeams Ranked by Championship Probability:")
    print("-" * 70)
    for i, row in results.iterrows():
        print(f"{row['team']:4s} ({row['conference']}) Seed {int(row['playoff_seed']):2d} | "
              f"Win%: {row['win_pct']:.3f} | PD: {row['point_differential']:+4.0f} | "
              f"Champ Prob: {row['champion_probability']:.1%}")
    
    return results


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

def main():
    # Load data
    df = load_data('../all_seasons_data.csv')
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    print(f"\nTarget distribution:")
    print(f"  Non-Champions: {(y==0).sum()} ({(y==0).mean():.1%})")
    print(f"  Champions: {(y==1).sum()} ({(y==1).mean():.1%})")
    
    # Split data - use seasons before 2023 for training, 2023-2024 for testing
    # This simulates predicting future playoffs
    train_mask = df['season'] < 2023
    test_mask = df['season'] >= 2023
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"\nTraining set: {len(X_train)} samples (seasons 2000-2022)")
    print(f"Test set: {len(X_test)} samples (seasons 2023-2024)")
    
    # Train models
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    dt_model = train_decision_tree(X_train, y_train, max_depth=5)
    rf_model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=10)
    
    # Evaluate models
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    dt_pred, dt_proba = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    rf_pred, rf_proba = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Cross-validation on full dataset
    cross_validate_model(DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42), 
                        X, y, cv=5, model_name="Decision Tree")
    cross_validate_model(RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
                        X, y, cv=5, model_name="Random Forest")
    
    # Feature importance
    plot_feature_importance(rf_model, feature_cols, "Random Forest")
    
    # Visualize decision tree
    visualize_decision_tree(dt_model, feature_cols)
    
    # Predict 2025 playoffs (using 2024 season data)
    # Train on all historical data for final predictions
    rf_final = train_random_forest(X, y, n_estimators=200, max_depth=10)
    predict_2025_playoffs(rf_final, df, feature_cols, year=2024)
    
    return dt_model, rf_model, feature_cols


if __name__ == "__main__":
    dt_model, rf_model, features = main()
