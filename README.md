# NFL Playoff Predictor

A machine learning ensemble system for predicting NFL playoff outcomes using 25 years of historical data (2000-2024). The model combines XGBoost, Ridge regression, and market spread analysis to generate game-by-game predictions and bracket simulations.

## Features

- **Game Outcome Predictions** - Win probabilities for individual playoff matchups
- **Bracket Simulation** - Monte Carlo simulations for championship probability projections
- **Upset Detection** - Flags games where model predictions diverge from baseline expectations
- **Market Integration** - Incorporates betting spreads when available for enhanced accuracy
- **Historical Validation** - Tested against 25 seasons of playoff data

## Model Architecture

The prediction system uses a calibrated ensemble approach:

| Component | Purpose |
|-----------|---------|
| XGBoost Classifier | Gradient boosting for baseline predictions |
| Ridge Regression | Regularized logistic regression |
| Platt Scaling | Calibration to correct overconfidence |
| Spread-to-Probability GLM | Converts market spreads to win probabilities |

### Key Features Used

- **EPA Metrics** - Expected Points Added for offensive/defensive pass/rush plays
- **Matchup Differentials** - Pass edge, rush edge, offensive line exposure
- **Seed-Based Baselines** - Historical home-field advantage (~55% win rate)
- **Market Spreads** - ~3% probability adjustment per point

## Project Structure

```
nfl-predictor/
├── data/
│   ├── DesicionTree/                    # Main prediction scripts
│   │   ├── nfl_2025.py                  # Current season predictions (v16.2)
│   │   ├── predict_2025_playoffs_*.py   # Version-specific predictors
│   │   ├── validate_2024_playoffs_*.py  # Model validation scripts
│   │   └── nfl_playoff_predictor.py     # Core prediction framework
│   ├── csv/
│   │   ├── 2025/                        # Current season data
│   │   └── 2021-2024/                   # Historical yearly data
│   ├── nfl_playoff_results_2000_2024_with_epa.csv
│   ├── df_with_upsets_merged.csv
│   └── *.ipynb                          # Analysis notebooks
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd nfl-predictor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost statsmodels scipy matplotlib seaborn
```

## Usage

### Generate 2025 Playoff Predictions

```bash
cd data/DesicionTree
python nfl_2025.py
```

This outputs:
- Console display of round-by-round probabilities
- `wildcard_predictions_2025_v16_2.csv` - Wild card round predictions
- `playoff_sim_2025_v16_2.csv` - Full bracket simulation results

### Output Example

```
Wild Card Round:
  Chargers @ Texans: Texans 58.2% (Predicted: Texans)
  Steelers @ Ravens: Ravens 67.4% (Predicted: Ravens)
  ...

Championship Probabilities:
  Chiefs: 24.3%
  Lions: 18.7%
  ...
```

## Data Sources

| Dataset | Description |
|---------|-------------|
| `nfl_playoff_results_2000_2024_with_epa.csv` | Historical playoff games with EPA stats |
| `df_with_upsets_merged.csv` | Team-season records with 40+ metrics |
| `team_stats_2025_playoffs.csv` | Current season playoff team statistics |
| `wild_card.csv` | Wild card matchups with market spreads |

## Model Versions

The project tracks iterative model improvements:

- **v15** - Initial ensemble logistic regression
- **v16** - Added XGBoost with Platt scaling calibration
- **v16.1** - Improved regularization and safety guards
- **v16.2** - Tossup shrinking and refined ensemble weighting (current)

## How It Works

1. **Data Loading** - Historical games, team stats, and spread data
2. **Feature Engineering** - Per-season z-scores, matchup differentials
3. **Model Training** - Fit XGBoost, Ridge, and GLM on historical data
4. **Ensemble Prediction** - Weighted blend of models with calibration
5. **Bracket Simulation** - Monte Carlo simulations (10,000+ iterations)

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- statsmodels
- scipy
- matplotlib
- seaborn

## License

MIT License
