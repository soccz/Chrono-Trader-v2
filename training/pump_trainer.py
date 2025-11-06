import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os
import json
import optuna
import functools

from data.preprocessor import get_pump_dataset
from utils.logger import logger

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function to find the best hyperparameters for XGBoost."""
    param = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'tree_method': 'hist', # Faster
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'early_stopping_rounds': 50  # Pass early stopping rounds here
    }

    model = xgb.XGBClassifier(**param)
    
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              sample_weight=sample_weights,
              verbose=False)

    preds = model.predict_proba(X_val)
    roc_auc = roc_auc_score(y_val, preds, multi_class='ovr', average='weighted')
    
    return roc_auc

def run(tune: bool = False):
    """Trains the multi-class pump prediction model, with optional tuning."""
    config_path = os.path.join("models", "pump_model_config.json")
    params = {}

    if not tune and os.path.exists(config_path):
        logger.info(f"Loading pump model configuration from {config_path}")
        with open(config_path, 'r') as f:
            params = json.load(f)
    elif not tune:
        logger.info("No config file found, using default parameters.")
        params = {
            'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 5,
            'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1
        }

    logger.info("--- Loading Pump Prediction Dataset ---")
    dataset = get_pump_dataset()
    if dataset is None or dataset.empty:
        logger.error("Pump dataset could not be generated. Aborting training.")
        return

    y = dataset['pump_label']
    X = dataset.drop(columns=['pump_label', 'market'])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if tune:
        logger.info("--- Starting Hyperparameter Tuning with Optuna for Pump Model ---")
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
        
        study = optuna.create_study(direction='maximize')
        objective_with_data = functools.partial(objective, X_train=X_train_opt, y_train=y_train_opt, X_val=X_val_opt, y_val=y_val_opt)
        study.optimize(objective_with_data, n_trials=50)

        params = study.best_params
        logger.info(f"Tuning finished. Best weighted ROC AUC: {study.best_value:.4f}")
        logger.info(f"Best parameters found: {params}")

        with open(config_path, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"Best parameters saved to {config_path}")

    logger.info("--- Training XGBoost with best parameters ---")
    final_params = {
        'objective': 'multi:softprob', 'num_class': 4, 'eval_metric': 'mlogloss',
        'use_label_encoder': False, **params
    }
    
    model = xgb.XGBClassifier(**final_params)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    logger.info("--- Evaluating model on test set ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted Precision: {precision:.4f}")
    logger.info(f"Weighted Recall: {recall:.4f}")
    logger.info(f"Weighted F1-Score: {f1:.4f}")
    logger.info(f"Weighted ROC AUC on Test Set: {roc_auc:.4f}")

    model_path = os.path.join("models", "pump_classifier.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Pump prediction model saved to {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the pump prediction model, with optional tuning.")
    parser.add_argument('--tune', action='store_true', help="Run hyperparameter tuning with Optuna.")
    args = parser.parse_args()

    run(tune=args.tune)