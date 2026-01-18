#  ATP Tennis Match Predictor

A complete machine learning pipeline that predicts ATP tennis match outcomes using historical data, player ratings, and advanced statistical features. This project demonstrates end-to-end ML engineering from web scraping to model training.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-orange.svg)](https://github.com/microsoft/LightGBM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This system predicts the win probability for ATP tennis matches by combining:
- **Elo Rating System** - Surface-specific player strength ratings (Hard, Clay, Grass, Carpet)
- **Recent Form** - Win rate over last 90 days
- **Surface Performance** - Historical win percentage on each surface type
- **Head-to-Head Records** - Historical matchup results between players
- **Player Statistics** - Serve, return, rally, and tactical metrics from Tennis Abstract
- **Match Context** - Tournament level, round, best-of format

**Model Performance:**
- **Training Log Loss**: 0.459
- **Test Log Loss**: 0.628
- **Training Accuracy**: 79.3%
- **Test Accuracy**: 63.8%

---

##  Pipeline Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Data Ingestion │  →   │  Data Processing │  →   │ Feature Eng.    │
│                 │      │                  │      │                 │
│ • Web Scraping  │      │ • Name Normalization  │ • Elo Ratings   │
│ • Tennis Abstract│     │ • Surface Standardize │ • Form Metrics  │
│ • Player Stats  │      │ • Data Validation     │ • H2H History   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                             │
                         ┌──────────────────┐              │
                         │  Model Training  │  ←───────────┘
                         │                  │
                         │ • LightGBM       │
                         │ • Time-Series CV │
                         │ • Calibration    │
                         └──────────────────┘
```

---

##  Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tennis-predictor

# Install dependencies
pip install pandas numpy scikit-learn lightgbm selenium beautifulsoup4 pyyaml tqdm joblib
```

**Note:** You'll also need ChromeDriver installed for web scraping. Download from [chromedriver.chromium.org](https://chromedriver.chromium.org/)

### Running the Pipeline

```bash
# 1. Scrape player statistics from Tennis Abstract
python src/ingest/webscraper.py

# 2. Process and clean the data
python src/clean/data_processor.py

# 3. Filter to players with complete statistics
python src/clean/filter_players.py

# 4. Calculate Elo ratings
python src/ratings/elo_system.py

# 5. Build feature matrix
python src/features/feature_engineerng

# 6. Train the model
python src/model/trainer.py
```

**Note:** The project includes a Makefile for automation, but you can also run each script directly as shown above.

---

## Project Structure

```
tennis-predictor/
├── configs/                    # Configuration files
│   ├── data.yaml              # Data sources and paths
│   ├── features.yaml          # Feature engineering settings
│   └── model.yaml             # Model hyperparameters
│
├── data/                      # Data storage
│   ├── raw/                   # Original data files
│   ├── processed/             # Cleaned and normalized data
│   │   ├── matches.csv        # All processed matches
│   │   ├── matches_filtered.csv  # Matches with complete stats
│   │   ├── players.csv        # Player information
│   │   └── players_filtered.csv  # 68 players with detailed stats
│   └── features/              # Generated features
│       ├── elo_ratings.csv    # Final Elo ratings
│       ├── match_ratings.csv  # Match-level Elo snapshots
│       └── feature_matrix.csv # Complete ML-ready dataset
│
├── src/                       # Source code
│   ├── ingest/
│   │   └── webscraper.py     # Tennis Abstract data scraper
│   ├── clean/
│   │   ├── data_processor.py # Data cleaning and normalization
│   │   └── filter_players.py # Filter to players with stats
│   ├── ratings/
│   │   └── elo_system.py     # Elo rating calculator
│   ├── features/
│   │   └── feature_engineerng # Feature engineering pipeline
│   └── model/
│       └── trainer.py        # Model training with calibration
│
├── models/                    # Trained models
│   ├── lgbm_model.txt        # LightGBM model
│   ├── features.json         # Feature list
│   └── metrics.json          # Performance metrics
│
├── Makefile                  # Build automation
├── pyproject.toml           # Poetry dependencies
└── README.md                # This file
```

---

## Components

### 1. Data Ingestion (`src/ingest/`)

**Web Scraper** (`webscraper.py`)
- Scrapes Tennis Abstract for detailed player statistics
- Collects serve, return, rally, and tactical metrics for 68 top players
- Uses Selenium + BeautifulSoup for dynamic content extraction
- Outputs: `serve_leaders.csv`, `return_leaders.csv`, `rally_leaders.csv`, `tactics_leaders.csv`

**Data Sources:**
- Tennis Abstract player statistics (last 52 weeks)
- ATP match data (primary dataset: `atp_tennis.csv`)

### 2. Data Processing (`src/clean/`)

**Data Processor** (`data_processor.py`)
- Normalizes player names: `"Federer R."` → `"R. Federer"`
- Standardizes surface names: `hard`, `clay`, `grass`, `carpet`
- Merges match data with player statistics
- Handles missing values and validates data integrity
- Outputs: `matches.csv`, `players.csv`

**Player Filter** (`filter_players.py`)
- Filters matches to only include 68 players with complete statistics
- Ensures both players in each match have detailed stats
- Outputs: `matches_filtered.csv`, `players_filtered.csv`

### 3. Elo Rating System (`src/ratings/`)

**Elo Calculator** (`elo_system.py`)
- Maintains separate Elo ratings for:
  - Overall performance
  - Surface-specific ratings (Hard, Clay, Grass, Carpet)
- Initial rating: 1500
- K-factor: 32
- Stores pre-match ratings for feature engineering
- Outputs: `elo_ratings.csv`, `match_ratings.csv`

**Elo Formula:**
```python
Expected_Win_Prob = 1 / (1 + 10^((Rating_B - Rating_A) / 400))
New_Rating = Old_Rating + K * (Actual_Result - Expected_Prob)
```

### 4. Feature Engineering (`src/features/`)

**Feature Builder** (`feature_engineerng`)

Creates comprehensive feature matrix with:

**Elo Features:**
- `p1_elo_overall`, `p2_elo_overall` - Overall player strength
- `p1_elo_surface`, `p2_elo_surface` - Surface-specific ratings
- `elo_diff`, `elo_surface_diff` - Rating differences

**Form Features:**
- `p1_form_90d`, `p2_form_90d` - Win rate in last 90 days
- `p1_matches_90d`, `p2_matches_90d` - Match count (activity level)

**Surface Features:**
- `p1_surface_win_pct`, `p2_surface_win_pct` - Historical surface performance
- `p1_surface_matches`, `p2_surface_matches` - Experience on surface

**Head-to-Head:**
- `h2h_p1_win_pct` - Historical win rate in matchup
- `h2h_matches` - Number of previous encounters

**Ranking:**
- `p1_rank`, `p2_rank` - ATP rankings
- `rank_diff` - Ranking differential

**Player Stats** (for 68 players with data):
- Serve metrics: ace rate, serve win %, etc.
- Return metrics: break point conversion, etc.
- Rally metrics: rally length preferences
- Tactical metrics: net approach frequency

**Match Context:**
- `surface` - Hard, Clay, Grass, Carpet
- `level` - Grand Slam, Masters, ATP 250/500
- `round` - Round in tournament
- `best_of` - Match format (3 or 5 sets)

**Target:**
- `p1_won` - Binary outcome (1 if player 1 won, 0 otherwise)

### 5. Model Training (`src/model/`)

**Trainer** (`trainer.py`)

**Algorithm:** LightGBM (Gradient Boosting Decision Trees)

**Training Process:**
1. Time-series split (train on pre-2020, test on 2020+)
2. One-hot encoding of categorical features
3. LightGBM training with early stopping
4. Platt scaling calibration on holdout set
5. Model serialization with metrics tracking

**Hyperparameters:**
- `num_leaves`: 31
- `learning_rate`: 0.05
- `max_depth`: 6
- `feature_fraction`: 0.8
- `bagging_fraction`: 0.8
- Early stopping: 50 rounds

**Evaluation Metrics:**
- Log Loss (primary metric)
- Brier Score
- Accuracy
- AUC-ROC (when calibrated)

**Output Files:**
- `lgbm_model.txt` - Trained model
- `metrics.json` - Performance metrics
- `features.json` - Feature importance

---

## Model Performance

### Current Results

```json
{
  "train": {
    "log_loss": 0.459,
    "brier_score": 0.148,
    "accuracy": 0.793
  },
  "test": {
    "log_loss": 0.628,
    "brier_score": 0.220,
    "accuracy": 0.638
  }
}
```

### Interpretation

- **Log Loss < 0.693**: Better than random guessing (0.693)
- **Test Accuracy 63.8%**: Reasonable for tennis prediction (inherent randomness)
- **Training vs Test Gap**: Indicates some overfitting; potential improvements:
  - More regularization
  - Feature selection
  - Additional training data
  - Time-series cross-validation instead of single split

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11+ |
| **ML Framework** | LightGBM, scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Web Scraping** | Selenium, BeautifulSoup4 |
| **Configuration** | PyYAML |
| **Utilities** | tqdm, joblib |

---

## Future Enhancements

### Immediate Improvements
- [ ] Implement proper time-series cross-validation (rolling window)
- [ ] Add feature importance analysis and selection
- [ ] Calibration curve visualization
- [ ] Handle missing statistics more robustly

### Model Enhancements
- [ ] Ensemble models (XGBoost + LightGBM)
- [ ] Neural network baseline (multilayer perceptron)
- [ ] Incorporate betting odds as features
- [ ] Add player fatigue modeling (days since last match)

### Data Expansion
- [ ] Scrape historical ATP data (pre-2020)
- [ ] Add live rankings data
- [ ] Include injury status
- [ ] Weather conditions (for outdoor matches)

### Future Deployment (Not Yet Implemented)
- [ ] FastAPI REST API for predictions
- [ ] Docker containerization
- [ ] Model versioning with MLflow
- [ ] Web interface for predictions

---

## 📝 Configuration

Configuration files in `configs/` allow easy customization:

**data.yaml** - Data sources and scraping settings
**features.yaml** - Feature engineering parameters
**model.yaml** - Model hyperparameters and CV settings

Example configuration:
```yaml
# configs/model.yaml
lightgbm:
  num_leaves: 63
  max_depth: 8
  learning_rate: 0.05
  n_estimators: 3000
  
calibration: platt
```

---


## 🙏 Acknowledgments

- **Data Sources:**
  - [Tennis Abstract](http://www.tennisabstract.com/) - Player statistics
  - Jeff Sackmann's [ATP Tennis Database](https://github.com/JeffSackmann/tennis_atp)
  
- **Inspiration:**
  - FiveThirtyEight's Elo tennis ratings
  - Academic research on tennis match prediction

---

