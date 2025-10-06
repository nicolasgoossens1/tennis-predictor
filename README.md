# Tennis Match Predictor

A pre-match win probability model for ATP singles tennis matches (2000–present) using gradient-boosted trees with time-aware validation and post-hoc calibration.

## Overview

This project builds a machine learning model to predict the probability that Player A beats Player B in a tennis match, given contextual information like surface, tournament level, and recent form. The model uses LightGBM with careful feature engineering and time-series cross-validation to ensure robust, leak-free predictions.

## Success Criteria (MVP)

- **Log loss < 0.69** (better than coin flip)
- **Brier score ≤ 0.24**
- **Expected Calibration Error (ECE) < 0.03**
- Clear model card with data window, features, parameters, and metrics
- Leak-safe features (built only from data available before match date)

## Tech Stack

- **Python 3.11+**
- **Data**: DuckDB + Parquet, pandas/Polars, Pandera for schema validation
- **Modeling**: LightGBM, scikit-learn for calibration/metrics
- **Config**: YAML with OmegaConf/Hydra
- **API**: FastAPI with Pydantic schemas
- **Experiment tracking**: MLflow
- **Packaging**: Poetry
- **Testing**: pytest

## Quick Start

```bash
# Install dependencies
make setup

# Run the full training pipeline (with placeholder data)
make pipeline

# Start the API server
make serve

# Run tests
make test
```

## Project Structure

```
tennis-predictor/
├─ data/                      # raw/, processed/, features/ (gitignored)
├─ configs/                   # YAML configuration files
│  ├─ data.yaml              # data paths and source configuration
│  ├─ features.yaml          # feature engineering parameters
│  └─ model.yaml             # model hyperparameters and CV config
├─ notebooks/                 # EDA and analysis notebooks
├─ src/
│  ├─ ingest/                # data loading and normalization
│  ├─ clean/                 # data cleaning and canonicalization
│  ├─ ratings/               # Elo and serve/return rating systems
│  ├─ features/              # feature engineering transformers
│  ├─ model/                 # training, calibration, and prediction
│  ├─ eval/                  # evaluation metrics and model cards
│  └─ app/                   # FastAPI server and schemas
├─ tests/                    # unit tests for all modules
├─ mlruns/                   # MLflow experiment tracking
└─ models/                   # saved model artifacts
```

## Data Model

The system uses five canonical tables with strict Pandera schemas:

- **players**: player_id, name, hand, birth_year
- **matches**: match_id, date, tournament info, player IDs, winner, odds
- **stats**: match-level statistics (serve/return points, aces, etc.)
- **ratings**: time-series of Elo and serve/return ratings per player
- **features**: final feature matrix for model training

## Features

The model uses compact, symmetric features:

- **Player strength**: Overall & surface-specific Elo ratings, 26w/52w deltas
- **Serve & return**: Rolling hold%/break% with opponent adjustment
- **Context**: Surface, indoor flag, tournament level, round depth, best-of format
- **Form & fatigue**: Last-10 win%, days since last match, rest flags
- **Head-to-head**: Recent matchup history (capped at 5 matches)
- **Demographics**: Age, handedness, experience buckets

## Model Training

- **Algorithm**: LightGBM with binary log-loss objective
- **Validation**: Rolling time-series CV (train ≤2018 → val 2019, etc.)
- **Calibration**: Platt scaling or isotonic regression on out-of-fold predictions
- **Metrics**: Log loss (primary), Brier score, AUC, Expected Calibration Error

## API Usage

The FastAPI server provides two endpoints:

```bash
# Health check
GET /health

# Match prediction
POST /predict
{
  "player_a": "Roger Federer",
  "player_b": "Rafael Nadal", 
  "surface": "hard",
  "indoor": false,
  "best_of": 3,
  "level": "Grand Slam",
  "round": "F",
  "date": "2023-09-10"
}
```

Response:
```json
{
  "prob_a_wins": 0.45,
  "model_version": "v1.0.0",
  "explanations": {...}
}
```

## Development

```bash
# Set up development environment
make dev-setup

# Run linting and type checks
make lint
make type-check

# Format code
make format

# Run all CI checks
make ci
```

## Configuration

All settings are configurable via YAML files in the `configs/` directory:

- `data.yaml`: Data paths and source configuration
- `features.yaml`: Feature engineering parameters and windows
- `model.yaml`: Model hyperparameters and cross-validation setup

## Testing

The test suite includes:

- Player name canonicalization
- Elo rating updates with toy datasets
- Feature symmetry validation
- Leak prevention (date wall enforcement)
- Calibration monotonicity checks
- API schema validation

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run only fast tests
make quick-test
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `make ci` to check all tests pass
5. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact nicolasgoossens7@gmail.com .