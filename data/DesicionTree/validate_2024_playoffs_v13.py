"""
NFL Playoff Model v13 - Stabilized Spread + Source-Aware λ

KEY FIXES FROM v12:
1. Tune spread alpha via held-out validation (not hardcoded)
2. Clamp spread slope to sane band (-0.12 to -0.04 per point)
3. Disable Platt calibration for spread games entirely
4. Cap baseline Platt params (a: 0.8-1.2, b: -0.2 to 0.2)
5. Separate λ by source (λ_spread small, λ_base larger)

ARCHITECTURE:
- Offset GLM with ridge regularization
- Source-aware λ: spread games get minimal adjustment
- No calibration on spread (market is already calibrated)
- Conservative calibration on baseline games
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DATA LOADING
# ============================================================

def load_all_data():
    games_df = pd.read_csv('../nfl_playoff_results_2000_2024_with_epa.csv')
    team_df = pd.read_csv('../df_with_upsets_merged.csv')
    spread_df = pd.read_csv('../csv/nfl_biggest_playoff_upsets_2000_2024.csv')
    
    print(f"Loaded {len(games_df)} playoff games")
    print(f"Loaded {len(team_df)} team-season records")
    print(f"Loaded {len(spread_df)} games with spread data")
    
    return games_df, team_df, spread_df


def merge_spread_data_safe(games_df, spread_df):
    games_df = games_df.copy()
    spread_lookup = {}
    
    for _, row in spread_df.iterrows():
        season, game_type = row['season'], row['game_type']
        underdog = row['underdog']
        magnitude = abs(row['spread_magnitude'])
        teams = {row['winner'], row['loser']}
        
        matching = games_df[
            (games_df['season'] == season) & 
            (games_df['game_type'] == game_type) &
            (games_df['home_team'].isin(teams)) &
            (games_df['away_team'].isin(teams))
        ]
        
        if len(matching) > 0:
            game = matching.iloc[0]
            home = game['home_team']
            home_spread = magnitude if home == underdog else -magnitude
            key = (season, game_type, game['away_team'], home)
            spread_lookup[key] = home_spread
    
    games_df['home_spread'] = games_df.apply(
        lambda r: spread_lookup.get((r['season'], r['game_type'], r['away_team'], r['home_team']), np.nan),
        axis=1
    )
    
    matched = games_df['home_spread'].notna().sum()
    print(f"Matched spread data for {matched}/{len(games_df)} games")
    
    return games_df


# ============================================================
# SPREAD MODEL (with tuned alpha + slope clamping)
# ============================================================

def fit_spread_model_scaled(features_df, train_seasons, alpha=0.5, clamp_slope=True):
    """Spread model with slope clamping to prevent overconfidence."""
    train_df = features_df[
        (features_df['season'].isin(train_seasons)) & 
        (features_df['home_spread'].notna())
    ].copy()
    
    if len(train_df) < 20:
        return None
    
    spread_td = train_df['home_spread'].values / 7.0
    X = sm.add_constant(spread_td)
    y = train_df['home_wins'].values
    
    try:
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        model = glm.fit_regularized(alpha=alpha, L1_wt=0.0)
        
        intercept = model.params[0]
        slope = model.params[1]
        slope_per_point = slope / 7.0
        
        # Clamp slope to realistic range
        if clamp_slope:
            slope_per_point_clamped = np.clip(slope_per_point, -0.12, -0.04)
            slope = slope_per_point_clamped * 7.0
            if abs(slope_per_point - slope_per_point_clamped) > 0.001:
                print(f"  Slope clamped: {slope_per_point:.4f} -> {slope_per_point_clamped:.4f}")
        
        return {'intercept': intercept, 'slope': slope, 'scale': 7.0, 
                'slope_per_point': slope / 7.0}
    except Exception as e:
        print(f"Spread model fit failed: {e}")
        return None


def tune_spread_alpha(features_df, train_seasons, val_seasons):
    """Tune spread model alpha via held-out log loss."""
    print("\nTuning spread alpha...")
    
    best_alpha, best_ll, best_model = 0.5, float('inf'), None
    
    for alpha in [0.1, 0.25, 0.5, 1.0, 2.0]:
        smod = fit_spread_model_scaled(features_df, train_seasons, alpha=alpha, clamp_slope=False)
        if smod is None:
            continue
        
        val_df = features_df[
            (features_df['season'].isin(val_seasons)) & 
            (features_df['home_spread'].notna())
        ].copy()
        
        if len(val_df) < 5:
            continue
        
        probs = val_df['home_spread'].apply(
            lambda s: expit(smod['intercept'] + smod['slope'] * (s / 7.0))
        ).values
        y = val_df['home_wins'].values
        
        # Compute log loss
        ll = -np.mean(np.log(np.clip(np.where(y == 1, probs, 1 - probs), 0.01, 0.99)))
        
        print(f"  α={alpha}: val_ll={ll:.4f}, slope/pt={smod['slope']/7:.4f}")
        
        if ll < best_ll:
            best_ll, best_alpha = ll, alpha
    
    print(f"  Best α={best_alpha}")
    
    # Refit with best alpha + clamping
    return fit_spread_model_scaled(features_df, train_seasons, alpha=best_alpha, clamp_slope=True)


def get_spread_offset_logit(home_spread, spread_model):
    if spread_model is None or pd.isna(home_spread):
        return np.nan
    spread_td = home_spread / spread_model['scale']
    offset = spread_model['intercept'] + spread_model['slope'] * spread_td
    return np.clip(offset, -4, 4)  # Tighter clip for stability


def print_spread_model(spread_model):
    """Print spread model diagnostics."""
    if spread_model is None:
        print("  No spread model")
        return
    
    print(f"\nSpread Model:")
    print(f"  logit(P) = {spread_model['intercept']:.3f} + {spread_model['slope_per_point']:.4f} * pts")
    print(f"  Implied probabilities:")
    for pts in [-10, -7, -3, 0, 3, 7, 10]:
        offset = spread_model['intercept'] + spread_model['slope_per_point'] * pts
        prob = expit(np.clip(offset, -4, 4))
        print(f"    {pts:+3d} pts: {prob:.1%}")


# ============================================================
# BASELINE MODELS
# ============================================================

def compute_historical_baselines(games_df, max_season):
    train_games = games_df[games_df['season'] <= max_season]
    home_games = train_games[train_games['location'] == 'Home']
    if len(home_games) == 0:
        return {'home_win_rate': 0.55}
    home_wins = (home_games['winner'] == home_games['home_team']).sum()
    return {'home_win_rate': home_wins / len(home_games)}


def baseline_probability(home_seed, away_seed, is_neutral, baselines):
    seed_diff = away_seed - home_seed
    if is_neutral:
        base_prob = 0.50 + (seed_diff * 0.03)
    else:
        base_prob = baselines['home_win_rate'] + (seed_diff * 0.02)
    return np.clip(base_prob, 0.20, 0.85)


def fit_expected_win_pct_model(team_df, max_season):
    train_data = team_df[team_df['season'] <= max_season].dropna(subset=['win_pct', 'net_epa'])
    if len(train_data) < 20:
        return {'intercept': 0.5, 'slope': 2.0}
    X = sm.add_constant(train_data['net_epa'].values.reshape(-1, 1))
    model = sm.OLS(train_data['win_pct'].values, X).fit()
    return {'intercept': model.params[0], 'slope': model.params[1]}


# ============================================================
# Z-SCORES
# ============================================================

def compute_season_zscores(team_df):
    team_df = team_df.copy()
    
    higher_is_better = ['passing_epa', 'rushing_epa', 'total_offensive_epa', 'net_epa',
                        'point_differential', 'win_pct', 'pass_rush_rating', 
                        'pressure_rate', 'sack_rate', 'pass_block_rating', 'protection_rate']
    lower_is_better = ['defensive_epa', 'defensive_pass_epa', 'defensive_rush_epa', 'sacks_allowed_rate']
    
    for stat in higher_is_better:
        if stat in team_df.columns:
            team_df[f'z_{stat}'] = team_df.groupby('season')[stat].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9))
    
    for stat in lower_is_better:
        if stat in team_df.columns:
            team_df[f'z_{stat}'] = team_df.groupby('season')[stat].transform(
                lambda x: -(x - x.mean()) / (x.std() + 1e-9))
    
    return team_df


# ============================================================
# MATCHUP FEATURES
# ============================================================

def compute_matchup_features(away_data, home_data):
    features = {}
    
    def get_z(data, stat, default=0):
        z_col = f'z_{stat}'
        return data[z_col] if z_col in data.index and pd.notna(data[z_col]) else default
    
    # EPA z-scores
    home_pass_off = get_z(home_data, 'passing_epa')
    home_rush_off = get_z(home_data, 'rushing_epa')
    away_pass_off = get_z(away_data, 'passing_epa')
    away_rush_off = get_z(away_data, 'rushing_epa')
    
    home_pass_def = get_z(home_data, 'defensive_pass_epa')
    home_rush_def = get_z(home_data, 'defensive_rush_epa')
    away_pass_def = get_z(away_data, 'defensive_pass_epa')
    away_rush_def = get_z(away_data, 'defensive_rush_epa')
    
    # Edges
    home_pass_edge = home_pass_off - away_pass_def
    home_rush_edge = home_rush_off - away_rush_def
    away_pass_edge = away_pass_off - home_pass_def
    away_rush_edge = away_rush_off - home_rush_def
    
    features['delta_pass_edge'] = home_pass_edge - away_pass_edge
    features['delta_rush_edge'] = home_rush_edge - away_rush_edge
    
    # OL/DL
    home_ol = get_z(home_data, 'protection_rate')
    away_ol = get_z(away_data, 'protection_rate')
    home_dl = get_z(home_data, 'pressure_rate')
    away_dl = get_z(away_data, 'pressure_rate')
    
    # Exposures
    home_ol_exposure = np.maximum(0, away_dl - home_ol)
    away_ol_exposure = np.maximum(0, home_dl - away_ol)
    
    features['home_ol_exposure'] = home_ol_exposure
    features['away_ol_exposure'] = away_ol_exposure
    features['delta_ol_exposure'] = away_ol_exposure - home_ol_exposure
    features['total_ol_exposure'] = home_ol_exposure + away_ol_exposure
    
    home_pass_d_exposure = np.maximum(0, away_pass_off - home_pass_def)
    away_pass_d_exposure = np.maximum(0, home_pass_off - away_pass_def)
    
    features['home_pass_d_exposure'] = home_pass_d_exposure
    features['away_pass_d_exposure'] = away_pass_d_exposure
    features['delta_pass_d_exposure'] = away_pass_d_exposure - home_pass_d_exposure
    features['total_pass_d_exposure'] = home_pass_d_exposure + away_pass_d_exposure
    
    features['delta_rush_d_exposure'] = (
        np.maximum(0, home_rush_off - away_rush_def) - 
        np.maximum(0, away_rush_off - home_rush_def)
    )
    
    # Raw for reporting
    features['home_ol_z'] = home_ol
    features['away_ol_z'] = away_ol
    features['home_dl_z'] = home_dl
    features['away_dl_z'] = away_dl
    
    return features


# ============================================================
# FEATURE ENGINEERING v13
# ============================================================

def prepare_game_features_v13(games_df, team_df, epa_model, baselines, spread_model):
    team_df = compute_season_zscores(team_df)
    features = []
    
    for _, game in games_df.iterrows():
        season = game['season']
        is_neutral = game['location'] == 'Neutral'
        
        orig_away, orig_home = game['away_team'], game['home_team']
        
        orig_away_data = team_df[(team_df['team'] == orig_away) & (team_df['season'] == season)]
        orig_home_data = team_df[(team_df['team'] == orig_home) & (team_df['season'] == season)]
        
        if len(orig_away_data) == 0 or len(orig_home_data) == 0:
            continue
        
        orig_away_data, orig_home_data = orig_away_data.iloc[0], orig_home_data.iloc[0]
        
        if pd.isna(game.get('away_offensive_epa')) or pd.isna(game.get('home_offensive_epa')):
            continue
        
        orig_away_seed = int(orig_away_data['playoff_seed'])
        orig_home_seed = int(orig_home_data['playoff_seed'])
        
        # Neutral handling
        if is_neutral and orig_away_seed < orig_home_seed:
            away, home = orig_home, orig_away
            away_data, home_data = orig_home_data, orig_away_data
            away_seed, home_seed = orig_home_seed, orig_away_seed
            home_spread = -game.get('home_spread', np.nan) if pd.notna(game.get('home_spread')) else np.nan
        else:
            away, home = orig_away, orig_home
            away_data, home_data = orig_away_data, orig_home_data
            away_seed, home_seed = orig_away_seed, orig_home_seed
            home_spread = game.get('home_spread', np.nan)
        
        # Core stats
        away_games = away_data['wins'] + away_data['losses']
        home_games = home_data['wins'] + home_data['losses']
        away_pd_pg = away_data['point_differential'] / max(away_games, 1)
        home_pd_pg = home_data['point_differential'] / max(home_games, 1)
        away_net_epa = away_data.get('net_epa', 0)
        home_net_epa = home_data.get('net_epa', 0)
        
        # Matchup
        matchup_features = compute_matchup_features(away_data, home_data)
        
        # Vulnerability
        away_exp = epa_model['intercept'] + epa_model['slope'] * away_net_epa
        home_exp = epa_model['intercept'] + epa_model['slope'] * home_net_epa
        
        # Underseeded
        season_teams = team_df[team_df['season'] == season].copy()
        season_teams['pd_pg'] = season_teams['point_differential'] / (season_teams['wins'] + season_teams['losses']).clip(lower=1)
        season_teams['qrank'] = season_teams['pd_pg'].rank(ascending=False)
        
        away_qr = season_teams[season_teams['team'] == away]['qrank'].values
        home_qr = season_teams[season_teams['team'] == home]['qrank'].values
        away_quality_rank = int(away_qr[0]) if len(away_qr) else len(season_teams) // 2
        home_quality_rank = int(home_qr[0]) if len(home_qr) else len(season_teams) // 2
        
        # Momentum
        away_mom = away_data.get('momentum_residual', 0) or 0
        home_mom = home_data.get('momentum_residual', 0) or 0
        
        # Offsets
        is_neutral_int = 1 if is_neutral else 0
        baseline_prob = baseline_probability(home_seed, away_seed, is_neutral, baselines)
        baseline_logit_val = logit(baseline_prob)
        
        spread_offset = get_spread_offset_logit(home_spread, spread_model)
        
        if pd.notna(spread_offset):
            offset_logit = spread_offset
            offset_source = 'spread'
            spread_prob = expit(spread_offset)
        else:
            offset_logit = baseline_logit_val
            offset_source = 'baseline'
            spread_prob = np.nan
        
        actual_winner = game['winner']
        home_wins = 1 if actual_winner == home else 0
        
        feature_row = {
            'season': season, 'game_type': game['game_type'],
            'away_team': away, 'home_team': home,
            'orig_away_team': orig_away, 'orig_home_team': orig_home,
            'winner': actual_winner,
            'away_score': game['away_score'], 'home_score': game['home_score'],
            'home_wins': home_wins, 'is_neutral': is_neutral_int,
            'away_seed': away_seed, 'home_seed': home_seed,
            'seed_diff': away_seed - home_seed,
            
            'delta_net_epa': home_net_epa - away_net_epa,
            'delta_pd_pg': home_pd_pg - away_pd_pg,
            'delta_vulnerability': (away_data['win_pct'] - away_exp) - (home_data['win_pct'] - home_exp),
            'delta_underseeded': (away_seed - away_quality_rank) - (home_seed - home_quality_rank),
            'delta_momentum': home_mom - away_mom,
            
            'baseline_prob': baseline_prob, 'baseline_logit': baseline_logit_val,
            'home_spread': home_spread, 'spread_offset': spread_offset,
            'spread_prob': spread_prob, 'offset_logit': offset_logit,
            'offset_source': offset_source,
        }
        
        feature_row.update(matchup_features)
        features.append(feature_row)
    
    return pd.DataFrame(features)


# ============================================================
# RIDGE OFFSET GLM
# ============================================================

def train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=1.0):
    train_df = features_df[features_df['season'].isin(train_seasons)].dropna(subset=feature_cols + ['offset_logit'])
    
    if len(train_df) < 30:
        return None
    
    X_raw = train_df[feature_cols].values
    mu, sd = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-9
    X_scaled = (X_raw - mu) / sd
    
    X = sm.add_constant(X_scaled)
    y = train_df['home_wins'].values
    offset = train_df['offset_logit'].values
    
    try:
        result = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit_regularized(alpha=alpha, L1_wt=0.0)
    except:
        try:
            result = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit()
        except:
            return None
    
    return {'model': result, 'feature_cols': feature_cols, 'mu': mu, 'sd': sd, 
            'train_samples': len(train_df), 'alpha': alpha}


# ============================================================
# CALIBRATION (spread disabled, baseline capped)
# ============================================================

def calibrate_baseline_only(model_dict, features_df, calib_seasons, shrinkage=0.3):
    """
    Calibrate ONLY baseline-offset games.
    Spread games get identity (no calibration) since market is already calibrated.
    """
    calib_df = features_df[
        (features_df['season'].isin(calib_seasons)) & 
        (features_df['offset_source'] == 'baseline')
    ].dropna(subset=model_dict['feature_cols'] + ['offset_logit'])
    
    platt_params = {
        'spread': {'a': 1.0, 'b': 0.0},  # Identity for spread
        'baseline': {'a': 1.0, 'b': 0.0}
    }
    
    if len(calib_df) < 5:
        print("  Insufficient baseline games for calibration")
        return platt_params
    
    preds = []
    for _, row in calib_df.iterrows():
        pred = predict_raw(model_dict, row)
        if pred is not None:
            preds.append({'prob': pred['raw_prob'], 'actual': row['home_wins']})
    
    if len(preds) < 5:
        return platt_params
    
    pred_df = pd.DataFrame(preds)
    clipped = pred_df['prob'].clip(0.05, 0.95)
    X = sm.add_constant(logit(clipped).values)
    
    try:
        platt = sm.GLM(pred_df['actual'].values, X, family=sm.families.Binomial()).fit()
        a_raw, b_raw = platt.params[1], platt.params[0]
        
        # Skip if unstable (a_raw negative or extreme)
        if a_raw < 0 or abs(a_raw) > 10 or abs(b_raw) > 5:
            print(f"  Baseline Platt unstable (a={a_raw:.2f}, b={b_raw:.2f}), using identity")
            return platt_params
        
        # Apply shrinkage
        a = 1.0 + shrinkage * (a_raw - 1.0)
        b = 0.0 + shrinkage * b_raw
        
        # Hard cap for safety
        a = np.clip(a, 0.8, 1.2)
        b = np.clip(b, -0.2, 0.2)
        
        platt_params['baseline'] = {'a': a, 'b': b}
        print(f"  Baseline Platt (n={len(preds)}): a={a:.3f}, b={b:.3f}")
        
    except Exception as e:
        print(f"  Baseline Platt failed: {e}")
    
    return platt_params


def predict_raw(model_dict, game_features):
    if model_dict is None:
        return None
    
    try:
        X_raw = np.array([[game_features[col] for col in model_dict['feature_cols']]])
    except KeyError:
        return None
    
    if np.any(np.isnan(X_raw)):
        offset = game_features['offset_logit']
        return {'raw_prob': expit(offset), 'adjustment': 0, 'offset_logit': offset}
    
    X_scaled = (X_raw - model_dict['mu']) / model_dict['sd']
    X = sm.add_constant(X_scaled, has_constant='add')
    offset = np.array([game_features['offset_logit']])
    
    raw_prob = np.clip(model_dict['model'].predict(X, offset=offset)[0], 0.01, 0.99)
    adjustment = logit(raw_prob) - game_features['offset_logit']
    
    return {'raw_prob': raw_prob, 'adjustment': adjustment, 'offset_logit': game_features['offset_logit']}


# ============================================================
# SOURCE-AWARE λ PREDICTION
# ============================================================

def predict_with_source_lambda(model_dict, game_features, platt_params, lambda_spread=0.2, lambda_base=0.7):
    """
    Predict with source-aware λ blending:
    - Spread games: minimal adjustment (trust the market)
    - Baseline games: larger adjustment (model adds value)
    """
    pred = predict_raw(model_dict, game_features)
    if pred is None:
        return None
    
    offset_logit = pred['offset_logit']
    adjustment = pred['adjustment']
    src = game_features['offset_source']
    
    # Source-specific λ
    lam = lambda_spread if src == 'spread' else lambda_base
    
    # Blend
    blended_logit = offset_logit + lam * adjustment
    blended_prob = expit(np.clip(blended_logit, -4, 4))
    
    # Calibrate (only baseline gets non-identity)
    platt = platt_params.get(src, {'a': 1.0, 'b': 0.0})
    calibrated_logit = platt['b'] + platt['a'] * logit(np.clip(blended_prob, 0.05, 0.95))
    calibrated_logit = np.clip(calibrated_logit, -4, 4)
    final_prob = expit(calibrated_logit)
    
    return {
        'home_prob': final_prob,
        'raw_prob': pred['raw_prob'],
        'blended_prob': blended_prob,
        'adjustment': adjustment,
        'offset_logit': offset_logit,
        'lambda_used': lam,
        'predicted_winner': 'home' if final_prob > 0.5 else 'away'
    }


def tune_source_lambdas(model_dict, features_df, tune_seasons, platt_params):
    """Tune λ separately for spread and baseline games."""
    tune_df = features_df[features_df['season'].isin(tune_seasons)].dropna(
        subset=model_dict['feature_cols'] + ['offset_logit']
    )
    
    best_params = {'spread': 0.2, 'base': 0.7}
    
    for src, src_name in [('spread', 'spread'), ('baseline', 'base')]:
        src_df = tune_df[tune_df['offset_source'] == src]
        
        if len(src_df) < 10:
            continue
        
        best_lam, best_imp = 0.5, -float('inf')
        
        for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            model_lls, offset_lls = [], []
            
            for _, row in src_df.iterrows():
                pred = predict_raw(model_dict, row)
                if pred is None:
                    continue
                
                blended_logit = pred['offset_logit'] + lam * pred['adjustment']
                blended_prob = expit(np.clip(blended_logit, -4, 4))
                
                actual = row['home_wins']
                model_prob = blended_prob if actual == 1 else 1 - blended_prob
                offset_prob = expit(row['offset_logit'])
                offset_prob = offset_prob if actual == 1 else 1 - offset_prob
                
                model_lls.append(-np.log(np.clip(model_prob, 0.01, 0.99)))
                offset_lls.append(-np.log(np.clip(offset_prob, 0.01, 0.99)))
            
            if model_lls:
                improvement = np.mean(offset_lls) - np.mean(model_lls)
                if improvement > best_imp:
                    best_imp, best_lam = improvement, lam
        
        best_params[src_name] = best_lam
        print(f"  λ_{src_name} = {best_lam} (Δll = {best_imp:+.4f})")
    
    return best_params


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(results_df):
    metrics = {}
    metrics['accuracy'] = results_df['correct'].mean()
    metrics['n_games'] = len(results_df)
    
    model_probs = results_df.apply(
        lambda r: r['home_prob'] if r['actual_home_wins'] == 1 else 1 - r['home_prob'], axis=1)
    metrics['log_loss'] = -np.mean(np.log(model_probs.clip(0.01, 0.99)))
    metrics['brier'] = np.mean((results_df['home_prob'] - results_df['actual_home_wins'])**2)
    
    offset_probs = results_df.apply(
        lambda r: expit(r['offset_logit']) if r['actual_home_wins'] == 1 else 1 - expit(r['offset_logit']), axis=1)
    metrics['offset_log_loss'] = -np.mean(np.log(offset_probs.clip(0.01, 0.99)))
    
    base_probs = results_df.apply(
        lambda r: r['baseline_prob'] if r['actual_home_wins'] == 1 else 1 - r['baseline_prob'], axis=1)
    metrics['baseline_log_loss'] = -np.mean(np.log(base_probs.clip(0.01, 0.99)))
    
    spread_df = results_df.dropna(subset=['spread_prob'])
    metrics['spread_games'] = len(spread_df)
    if len(spread_df) > 0:
        sp = spread_df.apply(lambda r: r['spread_prob'] if r['actual_home_wins'] == 1 else 1 - r['spread_prob'], axis=1)
        metrics['spread_log_loss'] = -np.mean(np.log(sp.clip(0.01, 0.99)))
    
    upset_games = results_df[results_df['seed_diff'] > 0]
    if len(upset_games) > 0:
        actual = upset_games[upset_games['actual_home_wins'] == 0]
        predicted = upset_games[upset_games['predicted_winner'] == 'away']
        metrics['actual_upsets'] = len(actual)
        metrics['predicted_upsets'] = len(predicted)
        if len(actual) > 0:
            metrics['upset_recall'] = len(actual[actual.index.isin(predicted.index)]) / len(actual)
    
    return metrics


def print_matchup_report(results_df, features_df):
    print(f"\n{'='*100}")
    print("MATCHUP REPORT - 2024 UPSETS")
    print(f"{'='*100}")
    
    upsets = results_df[(results_df['seed_diff'] > 0) & (results_df['actual_home_wins'] == 0)]
    
    for _, g in upsets.iterrows():
        matchup = g['matchup']
        feat = features_df[
            (features_df['season'] == 2024) & 
            (features_df['orig_away_team'] == matchup.split(' @ ')[0]) &
            (features_df['orig_home_team'] == matchup.split(' @ ')[1])
        ]
        if len(feat) == 0:
            continue
        f = feat.iloc[0]
        
        called = "✓ CALLED" if g['predicted'] == g['actual'] else "✗ MISSED"
        
        print(f"\n{matchup}: {g['actual']} won [{called}]")
        print(f"  Seeds: {int(g['away_seed'])}v{int(g['home_seed'])}, Score: {g['score']}")
        print(f"  Model: {g['home_prob']:.0%}, Offset: {expit(g['offset_logit']):.0%}, λ={g['lambda_used']:.1f}")
        
        print(f"  Edges: Pass={f.get('delta_pass_edge', 0):+.2f}, Rush={f.get('delta_rush_edge', 0):+.2f}")
        print(f"  OL Exposure: Home={f.get('home_ol_exposure', 0):.2f}, Away={f.get('away_ol_exposure', 0):.2f}")


def validate_on_2024(model_dict, features_df, platt_params, lambda_params):
    print("\n" + "="*115)
    print(f"2024 PLAYOFF VALIDATION - MODEL v13 (λ_spread={lambda_params['spread']:.1f}, λ_base={lambda_params['base']:.1f})")
    print("="*115)
    
    test_df = features_df[features_df['season'] == 2024].copy()
    round_names = {'WC': 'Wild Card', 'DIV': 'Divisional', 'CON': 'Conference', 'SB': 'Super Bowl'}
    
    results = []
    
    for _, gf in test_df.iterrows():
        pred = predict_with_source_lambda(
            model_dict, gf, platt_params,
            lambda_spread=lambda_params['spread'],
            lambda_base=lambda_params['base']
        )
        if pred is None:
            continue
        
        actual_winner = gf['winner']
        pred_team = gf['home_team'] if pred['predicted_winner'] == 'home' else gf['away_team']
        
        results.append({
            'round': round_names.get(gf['game_type'], gf['game_type']),
            'matchup': f"{gf['orig_away_team']} @ {gf['orig_home_team']}",
            'away_seed': gf['away_seed'], 'home_seed': gf['home_seed'],
            'seed_diff': gf['seed_diff'],
            'baseline_prob': gf['baseline_prob'],
            'offset_src': gf['offset_source'],
            'offset_logit': gf['offset_logit'],
            'adjustment': pred['adjustment'],
            'blended_prob': pred['blended_prob'],
            'home_prob': pred['home_prob'],
            'spread_prob': gf.get('spread_prob', np.nan),
            'lambda_used': pred['lambda_used'],
            'predicted': pred_team,
            'predicted_winner': pred['predicted_winner'],
            'actual': actual_winner,
            'correct': pred_team == actual_winner,
            'actual_home_wins': 1 if actual_winner == gf['home_team'] else 0,
            'score': f"{gf['away_score']}-{gf['home_score']}",
        })
    
    results_df = pd.DataFrame(results)
    
    # Display
    for round_name in ['Wild Card', 'Divisional', 'Conference', 'Super Bowl']:
        rg = results_df[results_df['round'] == round_name]
        if len(rg) == 0:
            continue
        
        print(f"\n{round_name.upper()}")
        print(f"{'Matchup':<16} {'Seeds':<7} {'Base':<6} {'Src':<5} {'λ':<4} {'Adj':>6} {'Blend':>6} {'Final':>6} {'Sprd':>6} {'Pred':<5} {'Act':<5} {'✓'}")
        print("-" * 110)
        
        for _, g in rg.iterrows():
            result = "✓" if g['correct'] else "✗"
            seeds = f"{int(g['away_seed'])}v{int(g['home_seed'])}"
            spread_str = f"{g['spread_prob']:.0%}" if pd.notna(g['spread_prob']) else "N/A"
            src = "sprd" if g['offset_src'] == 'spread' else "base"
            print(f"{g['matchup']:<16} {seeds:<7} {g['baseline_prob']:.0%}   {src:<5} {g['lambda_used']:.1f}  {g['adjustment']:>+5.2f} "
                  f"{g['blended_prob']:>5.0%} {g['home_prob']:>5.0%} {spread_str:>6} "
                  f"{g['predicted']:<5} {g['actual']:<5} {result}")
    
    # Metrics
    print(f"\n{'='*115}")
    print("METRICS - MODEL v13")
    print(f"{'='*115}")
    
    metrics = evaluate_predictions(results_df)
    
    print(f"\n1. ACCURACY: {results_df['correct'].sum()}/{len(results_df)} ({metrics['accuracy']:.1%})")
    
    print(f"\n2. LOG LOSS (lower = better)")
    print(f"   Model v13:    {metrics['log_loss']:.4f}")
    print(f"   Offset-only:  {metrics['offset_log_loss']:.4f}")
    print(f"   Baseline:     {metrics['baseline_log_loss']:.4f}")
    if 'spread_log_loss' in metrics:
        print(f"   Spread:       {metrics['spread_log_loss']:.4f} (n={metrics['spread_games']})")
    
    vs_base = metrics['baseline_log_loss'] - metrics['log_loss']
    vs_off = metrics['offset_log_loss'] - metrics['log_loss']
    print(f"\n   vs Baseline:    {vs_base:+.4f} {'✓' if vs_base > 0 else '✗'}")
    print(f"   vs Offset-only: {vs_off:+.4f} {'✓' if vs_off > 0 else '✗'}")
    
    print(f"\n3. UPSETS: {metrics.get('actual_upsets', 0)} actual, {metrics.get('predicted_upsets', 0)} predicted, "
          f"recall={metrics.get('upset_recall', 0):.0%}")
    
    return results_df, metrics


# ============================================================
# MAIN
# ============================================================

def main():
    games_df, team_df, spread_df = load_all_data()
    games_df = merge_spread_data_safe(games_df, spread_df)
    
    train_seasons = list(range(2000, 2023))
    calib_seasons = [2021, 2022, 2023]
    tune_seasons = list(range(2015, 2024))
    spread_val_seasons = [2020, 2021, 2022, 2023]
    
    baselines = compute_historical_baselines(games_df, max_season=2023)
    print(f"\nBaseline home win rate: {baselines['home_win_rate']:.1%}")
    
    epa_model = fit_expected_win_pct_model(team_df, max_season=2023)
    
    # Initial features for spread tuning
    temp_features = prepare_game_features_v13(games_df, team_df, epa_model, baselines, None)
    
    # Tune spread alpha
    spread_model = tune_spread_alpha(temp_features, train_seasons, spread_val_seasons)
    print_spread_model(spread_model)
    
    # Rebuild features
    print("\nPreparing v13 features...")
    features_df = prepare_game_features_v13(games_df, team_df, epa_model, baselines, spread_model)
    print(f"Total games: {len(features_df)}")
    
    # Features
    feature_cols = [
        'delta_net_epa', 'delta_pd_pg', 'seed_diff',
        'delta_vulnerability', 'delta_underseeded', 'delta_momentum',
        'delta_pass_edge', 'delta_rush_edge',
        'delta_ol_exposure', 'delta_pass_d_exposure',
        'home_ol_exposure', 'away_ol_exposure', 'total_pass_d_exposure',
    ]
    
    available = [f for f in feature_cols if f in features_df.columns]
    print(f"\nFeatures ({len(available)}): {available}")
    feature_cols = available
    
    # Train
    print("\nTraining ridge model...")
    best_alpha, best_ll = 1.0, float('inf')
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        model = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=alpha)
        if model is None:
            continue
        
        test_df = features_df[features_df['season'] == 2023].dropna(subset=feature_cols + ['offset_logit'])
        preds = [predict_raw(model, row) for _, row in test_df.iterrows()]
        valid = [(p['raw_prob'], a) for p, a in zip(preds, test_df['home_wins'].values) if p is not None]
        
        if valid:
            ll = -np.mean([np.log(p if a == 1 else 1-p) for p, a in valid])
            print(f"  α={alpha}: ll={ll:.4f}")
            if ll < best_ll:
                best_ll, best_alpha = ll, alpha
    
    model_dict = train_ridge_offset_model(features_df, feature_cols, train_seasons, alpha=best_alpha)
    print(f"\nBest α={best_alpha}, trained on {model_dict['train_samples']} games")
    
    # Coefficients
    print("\nCoefficients:")
    for name, coef in zip(['const'] + feature_cols, model_dict['model'].params):
        print(f"  {name:<25} {coef:>+8.4f}")
    
    # Calibration (baseline only)
    print("\nCalibrating (baseline only)...")
    platt_params = calibrate_baseline_only(model_dict, features_df, calib_seasons)
    
    # Tune λ by source
    print("\nTuning λ by source...")
    lambda_params = tune_source_lambdas(model_dict, features_df, tune_seasons, platt_params)
    
    # Validate
    results_df, metrics = validate_on_2024(model_dict, features_df, platt_params, lambda_params)
    
    # Matchup report
    print_matchup_report(results_df, features_df)
    
    # Historical validation
    print("\n" + "="*115)
    print("HISTORICAL ROLLING VALIDATION (2015-2024)")
    print("="*115)
    
    all_results = []
    
    for test_year in range(2015, 2025):
        train_yrs = list(range(2000, test_year - 2))
        calib_yrs = [test_year - 2, test_year - 1]
        tune_yrs = list(range(2010, test_year))
        spread_val_yrs = list(range(test_year - 4, test_year))
        
        w_baselines = compute_historical_baselines(games_df, max_season=test_year - 1)
        w_epa = fit_expected_win_pct_model(team_df, max_season=test_year - 1)
        
        temp = prepare_game_features_v13(games_df, team_df, w_epa, w_baselines, None)
        w_spread = tune_spread_alpha(temp, train_yrs, spread_val_yrs)
        
        w_features = prepare_game_features_v13(games_df, team_df, w_epa, w_baselines, w_spread)
        
        model = train_ridge_offset_model(w_features, feature_cols, train_yrs, alpha=best_alpha)
        if model is None:
            continue
        
        platt = calibrate_baseline_only(model, w_features, calib_yrs)
        lam_params = tune_source_lambdas(model, w_features, tune_yrs, platt)
        
        test_df = w_features[w_features['season'] == test_year].dropna(subset=feature_cols + ['offset_logit'])
        
        results = []
        for _, gf in test_df.iterrows():
            pred = predict_with_source_lambda(model, gf, platt, lam_params['spread'], lam_params['base'])
            if pred is None:
                continue
            
            actual = 1 if gf['winner'] == gf['home_team'] else 0
            results.append({
                'correct': (pred['predicted_winner'] == 'home') == (actual == 1),
                'home_prob': pred['home_prob'],
                'offset_logit': gf['offset_logit'],
                'baseline_prob': gf['baseline_prob'],
                'actual': actual
            })
        
        if results:
            rdf = pd.DataFrame(results)
            model_ll = -np.mean([np.log(r['home_prob'] if r['actual'] else 1-r['home_prob']) for _, r in rdf.iterrows()])
            offset_ll = -np.mean([np.log(expit(r['offset_logit']) if r['actual'] else 1-expit(r['offset_logit'])) for _, r in rdf.iterrows()])
            base_ll = -np.mean([np.log(r['baseline_prob'] if r['actual'] else 1-r['baseline_prob']) for _, r in rdf.iterrows()])
            
            all_results.append({
                'season': test_year, 'games': len(rdf), 'correct': rdf['correct'].sum(),
                'acc': rdf['correct'].mean(), 'model_ll': model_ll, 'offset_ll': offset_ll, 'base_ll': base_ll,
                'lam_spread': lam_params['spread'], 'lam_base': lam_params['base']
            })
    
    hist_df = pd.DataFrame(all_results)
    
    print(f"\n{'Season':<8} {'N':<5} {'Acc':<7} {'Model':<8} {'Offset':<8} {'Base':<8} {'vs Base':<9} {'vs Off':<9} {'λs':<4} {'λb':<4}")
    print("-" * 90)
    for _, r in hist_df.iterrows():
        vs_base = r['base_ll'] - r['model_ll']
        vs_off = r['offset_ll'] - r['model_ll']
        print(f"{int(r['season']):<8} {int(r['games']):<5} {r['acc']:.1%}   {r['model_ll']:.4f}   {r['offset_ll']:.4f}   "
              f"{r['base_ll']:.4f}   {vs_base:+.4f} {'✓' if vs_base > 0 else ''}   {vs_off:+.4f} {'✓' if vs_off > 0 else ''}   "
              f"{r['lam_spread']:.1f}  {r['lam_base']:.1f}")
    
    print("-" * 90)
    avg = lambda col: hist_df[col].mean()
    print(f"{'AVG':<8} {int(hist_df['games'].sum()):<5} {hist_df['correct'].sum()/hist_df['games'].sum():.1%}   "
          f"{avg('model_ll'):.4f}   {avg('offset_ll'):.4f}   {avg('base_ll'):.4f}   "
          f"{avg('base_ll')-avg('model_ll'):+.4f}      {avg('offset_ll')-avg('model_ll'):+.4f}")
    
    print(f"\nModel beats baseline: {(hist_df['base_ll'] > hist_df['model_ll']).sum()}/{len(hist_df)}")
    print(f"Model beats offset:   {(hist_df['offset_ll'] > hist_df['model_ll']).sum()}/{len(hist_df)}")
    
    results_df.to_csv('../validation_results_2024_v13.csv', index=False)
    print(f"\nSaved to ../validation_results_2024_v13.csv")
    
    return model_dict, features_df, results_df, metrics


if __name__ == "__main__":
    model_dict, features_df, results, metrics = main()
