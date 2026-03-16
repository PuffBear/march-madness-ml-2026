import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
import json
warnings.filterwarnings('ignore')

DATA_DIR = 'data'

W_XGB = 0.4
W_LGB = 0.4
W_LR = 0.2
CLIP_LOW = 0.05
CLIP_HIGH = 0.95

def load_and_train():
    print("Loading data and training final ensemble on ALL data...")
    df = pd.read_csv(f'{DATA_DIR}/training_dataset.csv')
    
    feature_cols = [c for c in df.columns if c.endswith('_Diff')]
    X = df[feature_cols].copy()
    y = df['Target']
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    col_means = X.mean()
    X.fillna(col_means, inplace=True)
    
    with open('best_params.json', 'r') as f:
        params = json.load(f)
        xgb_params = params['xgb']
        lgb_params = params['lgb']
        
    xgb_params['seed'] = 43
    lgb_params['seed'] = 43
    
    # --- XGBoost ---
    dtrain = xgb.DMatrix(X, label=y)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=150)
    
    # --- LightGBM ---
    lgb_train = lgb.Dataset(X, label=y)
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=150)
    
    # --- Logistic Regression ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr_model = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', random_state=42)
    lr_model.fit(X_scaled, y)
    
    return xgb_model, lgb_model, lr_model, scaler, feature_cols, col_means

def run_predictions(xgb_model, lgb_model, lr_model, scaler, feature_cols, col_means):
    print("Generating predictions for Stage 2 Submission...")
    sub = pd.read_csv(f'{DATA_DIR}/SampleSubmissionStage2.csv')
    features = pd.read_csv(f'{DATA_DIR}/team_season_features.csv')
    
    sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
    sub['Team1'] = sub['ID'].apply(lambda x: int(x.split('_')[1]))
    sub['Team2'] = sub['ID'].apply(lambda x: int(x.split('_')[2]))
    
    # Merge Features T1
    df = sub.merge(features, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
    df.drop('TeamID', axis=1, inplace=True)
    rename_t1 = {col: f"{col}_T1" for col in features.columns if col not in ['Season', 'TeamID']}
    df.rename(columns=rename_t1, inplace=True)
    
    # Merge Features T2
    df = df.merge(features, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')
    df.drop('TeamID', axis=1, inplace=True)
    rename_t2 = {col: f"{col}_T2" for col in features.columns if col not in ['Season', 'TeamID']}
    df.rename(columns=rename_t2, inplace=True)
    
    # Compute differences
    for col in feature_cols:
        base_col = col.replace('_Diff', '')
        df[col] = df[f"{base_col}_T1"] - df[f"{base_col}_T2"]
    
    # --- Forward predictions ---
    X_fwd = df[feature_cols].copy()
    X_fwd.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_fwd.fillna(col_means, inplace=True)
    
    xgb_fwd = xgb_model.predict(xgb.DMatrix(X_fwd))
    lgb_fwd = lgb_model.predict(X_fwd)
    lr_fwd = lr_model.predict_proba(scaler.transform(X_fwd))[:, 1]
    
    ens_fwd = W_XGB * xgb_fwd + W_LGB * lgb_fwd + W_LR * lr_fwd
    
    # --- Inverse predictions (for symmetry) ---
    X_inv = -1 * X_fwd
    
    xgb_inv = xgb_model.predict(xgb.DMatrix(X_inv))
    lgb_inv = lgb_model.predict(X_inv)
    lr_inv = lr_model.predict_proba(scaler.transform(X_inv))[:, 1]
    
    ens_inv = W_XGB * xgb_inv + W_LGB * lgb_inv + W_LR * lr_inv
    
    # --- Symmetry-enforced blend ---
    final_preds = (ens_fwd + (1 - ens_inv)) / 2
    final_preds = np.clip(final_preds, CLIP_LOW, CLIP_HIGH)
    
    sub['Pred'] = final_preds
    sub[['ID', 'Pred']].to_csv(f'{DATA_DIR}/final_submission.csv', index=False)
    print(f"Saved {DATA_DIR}/final_submission.csv ({len(sub)} rows)")

def main():
    xgb_model, lgb_model, lr_model, scaler, feature_cols, col_means = load_and_train()
    run_predictions(xgb_model, lgb_model, lr_model, scaler, feature_cols, col_means)

if __name__ == '__main__':
    main()
