import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import optuna
import matplotlib.pyplot as plt
import warnings
import json
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
optuna.logging.set_verbosity(optuna.logging.WARNING)

W_XGB = 0.4
W_LGB = 0.4
W_LR = 0.2
CLIP_LOW = 0.05
CLIP_HIGH = 0.95

def load_and_prep_data():
    df = pd.read_csv(f'{DATA_DIR}/training_dataset.csv')
    feature_cols = [c for c in df.columns if c.endswith('_Diff')]
    
    X = df[feature_cols].copy()
    y = df['Target']
    seasons = df['Season']
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)
    
    return df, X, y, seasons, feature_cols

def tune_xgb(X, y, groups):
    print("\n--- Tuning XGBoost with Optuna ---")
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'seed': 42
        }
        
        gkf = GroupKFold(n_splits=3)
        losses = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            dtrain, dval = xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_val, label=y_val)
            model = xgb.train(params, dtrain, num_boost_round=300, evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False)
            preds = model.predict(dval)
            losses.append(log_loss(y_val, preds))
        return np.mean(losses)
        
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=15)
    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['seed'] = 42
    print("Best XGB Params:", best_params)
    return best_params

def tune_lgb(X, y, groups):
    print("\n--- Tuning LightGBM with Optuna ---")
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'verbose': -1,
            'seed': 42
        }
        
        gkf = GroupKFold(n_splits=3)
        losses = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(params, train_data, num_boost_round=300, valid_sets=[val_data], 
                              callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = model.predict(X_val)
            losses.append(log_loss(y_val, preds))
        return np.mean(losses)
        
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=15)
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'binary_logloss'
    best_params['verbose'] = -1
    best_params['seed'] = 42
    print("Best LGB Params:", best_params)
    return best_params

def train_eval_ensemble(X, y, groups, xgb_p, lgb_p):
    print("\n============================================================")
    print("Evaluating Ensemble with Season-Grouped CV (Leave Seasons Out)")
    print("============================================================")
    
    gkf = GroupKFold(n_splits=5)
    
    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_lr = np.zeros(len(X))
    
    xgb_losses, lgb_losses, lr_losses = [], [], []
    
    scaler = StandardScaler()
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # XGB
        dtrain, dval = xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_val, label=y_val)
        xgb_m = xgb.train(xgb_p, dtrain, num_boost_round=500, evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        xgb_p_f = xgb_m.predict(dval)
        oof_xgb[val_idx] = xgb_p_f
        xgb_losses.append(log_loss(y_val, xgb_p_f))
        
        # LGB
        l_train, l_val = lgb.Dataset(X_train, label=y_train), lgb.Dataset(X_val, label=y_val)
        lgb_m = lgb.train(lgb_p, l_train, num_boost_round=500, valid_sets=[l_val], callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_p_f = lgb_m.predict(X_val)
        oof_lgb[val_idx] = lgb_p_f
        lgb_losses.append(log_loss(y_val, lgb_p_f))
        
        # LR
        X_t_s = scaler.fit_transform(X_train)
        X_v_s = scaler.transform(X_val)
        lr_m = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', random_state=42)
        lr_m.fit(X_t_s, y_train)
        lr_p_f = lr_m.predict_proba(X_v_s)[:, 1]
        oof_lr[val_idx] = lr_p_f
        lr_losses.append(log_loss(y_val, lr_p_f))
        
        ens = W_XGB * xgb_p_f + W_LGB * lgb_p_f + W_LR * lr_p_f
        print(f"Fold {fold+1} [Seasons left out: {len(groups.iloc[val_idx].unique())}] - XGB: {xgb_losses[-1]:.4f} | LGB: {lgb_losses[-1]:.4f} | LR: {lr_losses[-1]:.4f} | Ens: {log_loss(y_val, ens):.4f}")

    ens_oof = W_XGB * oof_xgb + W_LGB * oof_lgb + W_LR * oof_lr
    ens_clipped = np.clip(ens_oof, CLIP_LOW, CLIP_HIGH)
    
    print("\n--- Summary ---")
    print(f"XGB Avg Loss: {np.mean(xgb_losses):.4f}")
    print(f"LGB Avg Loss: {np.mean(lgb_losses):.4f}")
    print(f"LR  Avg Loss: {np.mean(lr_losses):.4f}")
    print(f"ENSEMBLE OOF: {log_loss(y, ens_oof):.4f}")
    print(f"CLIPPED OOF:  {log_loss(y, ens_clipped):.4f}")
    
    # Save parameters for predictions
    with open('best_params.json', 'w') as f:
        json.dump({'xgb': xgb_p, 'lgb': lgb_p}, f)

def main():
    df, X, y, seasons, feature_cols = load_and_prep_data()
    best_xgb = tune_xgb(X, y, seasons)
    best_lgb = tune_lgb(X, y, seasons)
    train_eval_ensemble(X, y, seasons, best_xgb, best_lgb)

if __name__ == '__main__':
    main()
