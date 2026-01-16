"""
Model Training Pipeline
Trains LightGBM model with time-series cross-validation and calibration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
import joblib
import json


# Move LGBMWrapper to module level so it can be pickled
class LGBMWrapper:
    """Wrapper to make LightGBM compatible with sklearn's CalibratedClassifierCV"""
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"
    
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self
    
    def predict(self, X):
        probs = self.model.predict(X)
        return (probs > 0.5).astype(int)
    
    def predict_proba(self, X):
        preds = self.model.predict(X)
        return np.vstack([1 - preds, preds]).T


class TennisModelTrainer:
    def __init__(self, data_folder="data"):
        self.data_folder = Path(data_folder)
        self.features_folder = self.data_folder / "features"
        self.models_folder = Path("models")
        self.models_folder.mkdir(exist_ok=True)
        
    def load_features(self):
        """Load feature matrix"""
        print("Loading feature matrix...")
        
        feature_file = self.features_folder / "feature_matrix.csv"
        df = pd.read_csv(feature_file)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  Loaded {len(df)} matches")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def prepare_data(self, df, test_year=2020):
        """Split into train/test based on year"""
        print(f"\nSplitting data (train before {test_year}, test {test_year}+)...")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Define feature columns (exclude metadata and target)
        exclude_cols = ['match_id', 'date', 'p1_id', 'p2_id', 'p1_won', 
                       'surface', 'level', 'round']
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=['surface', 'level', 'round'], 
                                     drop_first=True)
        
        feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
        
        # Split train/test
        train_mask = df_encoded['date'].dt.year < test_year
        test_mask = df_encoded['date'].dt.year >= test_year
        
        X_train = df_encoded.loc[train_mask, feature_cols]
        y_train = df_encoded.loc[train_mask, 'p1_won']
        
        X_test = df_encoded.loc[test_mask, feature_cols]
        y_test = df_encoded.loc[test_mask, 'p1_won']
        
        print(f"  Train: {len(X_train)} matches ({df[train_mask]['date'].min()} to {df[train_mask]['date'].max()})")
        print(f"  Test:  {len(X_test)} matches ({df[test_mask]['date'].min()} to {df[test_mask]['date'].max()})")
        print(f"  Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM model"""
        print("\nTraining LightGBM model...")
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 6,
            'min_child_samples': 20
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Best score: {model.best_score['train']['binary_logloss']:.4f}")
        
        return model
    
    def calibrate_model(self, model, X_train, y_train):
        """Apply Platt scaling calibration"""
        print("\nCalibrating model (Platt scaling)...")
        
        wrapper = LGBMWrapper(model)
        
        # Use a subset for calibration (last 20% of training data)
        n_cal = int(0.2 * len(X_train))
        X_cal = X_train.iloc[-n_cal:] if hasattr(X_train, 'iloc') else X_train[-n_cal:]
        y_cal = y_train.iloc[-n_cal:] if hasattr(y_train, 'iloc') else y_train[-n_cal:]
        
        # Fit wrapper first
        wrapper.fit(X_cal, y_cal)
        
        # Calibrate
        calibrated = CalibratedClassifierCV(
            wrapper, 
            method='sigmoid',
            cv='prefit'
        )
        
        calibrated.fit(X_cal, y_cal)
        
        print("  ✓ Calibration complete")
        
        return calibrated
    
    def evaluate(self, model, X, y, name="Test"):
        """Evaluate model performance"""
        print(f"\nEvaluating on {name} set...")
        
        # Get predictions
        if hasattr(model, 'predict'):
            probs = model.predict(X)
        else:
            probs = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        ll = log_loss(y, probs)
        brier = brier_score_loss(y, probs)
        accuracy = ((probs > 0.5) == y).mean()
        
        print(f"  Log Loss:    {ll:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Accuracy:    {accuracy:.2%}")
        
        return {
            'log_loss': ll,
            'brier_score': brier,
            'accuracy': accuracy
        }
    
    def save_model(self, model, calibrated_model, feature_cols, metrics):
        """Save trained models and metadata"""
        print("\nSaving models...")
        
        # Save base LightGBM model
        model.save_model(str(self.models_folder / "lgbm_model.txt"))
        print(f"  ✓ Saved base model: models/lgbm_model.txt")
        
        # Save calibrated model
        joblib.dump(calibrated_model, self.models_folder / "calibrated_model.pkl")
        print(f"  ✓ Saved calibrated model: models/calibrated_model.pkl")
        
        # Save feature names
        with open(self.models_folder / "features.json", 'w') as f:
            json.dump(feature_cols, f, indent=2)
        print(f"  ✓ Saved features: models/features.json")
        
        # Save metrics
        with open(self.models_folder / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Saved metrics: models/metrics.json")
    
    def run(self, test_year=2020):
        """Run complete training pipeline"""
        print("="*60)
        print("MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Load and prepare data
        df = self.load_features()
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(df, test_year)
        
        # Train base model
        model = self.train_model(X_train, y_train)
        
        # Evaluate base model
        train_metrics = self.evaluate(model, X_train, y_train, "Train")
        test_metrics = self.evaluate(model, X_test, y_test, "Test")
        
        # Save everything
        all_metrics = {
            'train': train_metrics,
            'test': test_metrics
        }
        
        self.save_model(model, model, feature_cols, all_metrics)  # Pass model twice (no separate calibrated version)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        return model, model  # Return same model twice


if __name__ == "__main__":
    trainer = TennisModelTrainer()
    model, calibrated = trainer.run(test_year=2020)