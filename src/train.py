"""
Train multiple ML models for FIFA World Cup match prediction.
Automatically selects and saves the best model.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_clean
from features import compute_team_stats, FEATURE_COLUMNS
from elo import calculate_elo_ratings

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "matches.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42
        ),
    }
    
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="mlogloss",
            use_label_encoder=False, verbosity=0
        )
    
    if HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            random_seed=42, verbose=0
        )
    
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, verbose=-1
        )
    
    return models


def train():
    print("=" * 60)
    print("FIFA WORLD CUP PREDICTOR - TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading and preprocessing data...")
    if not os.path.exists(DATA_PATH):
        print("  Data not found, generating synthetic dataset...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "data", "generate_data.py")],
                      cwd=os.path.join(os.path.dirname(__file__), ".."))
    
    df = load_and_clean(DATA_PATH)
    print(f"  Loaded {len(df)} matches, {df['home_team'].nunique()} unique teams")
    
    # Calculate Elo ratings
    print("\n[2/5] Calculating Elo ratings...")
    elo_ratings, elo_history = calculate_elo_ratings(df)
    print(f"  Elo calculated for {len(elo_ratings)} teams")
    top5 = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5: {', '.join(f'{t}({e:.0f})' for t, e in top5)}")
    
    # Feature engineering
    print("\n[3/5] Engineering features...")
    feature_df = compute_team_stats(df, elo_ratings)
    X = feature_df[FEATURE_COLUMNS].values
    y = df["result"].values  # 0=away win, 1=draw, 2=home win
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  Class distribution: Away={np.sum(y==0)}, Draw={np.sum(y==1)}, Home={np.sum(y==2)}")
    
    # Train/test split (temporal — use last 20% as test)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train models
    print("\n[4/5] Training and evaluating models...")
    models = build_models()
    results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            ll = log_loss(y_test, y_proba)
            
            results[name] = {"model": model, "accuracy": acc, "f1": f1, "log_loss": ll}
            print(f"  {name:25s} | Acc: {acc:.3f} | F1: {f1:.3f} | LogLoss: {ll:.3f}")
        except Exception as e:
            print(f"  {name:25s} | ERROR: {e}")
    
    if not results:
        raise RuntimeError("No models trained successfully!")
    
    # Select best model (by accuracy)
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\n  ✓ Best model: {best_name} (accuracy={results[best_name]['accuracy']:.3f})")
    
    # Save artifacts
    print("\n[5/5] Saving artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(elo_ratings, os.path.join(MODEL_DIR, "elo_ratings.pkl"))
    joblib.dump(elo_history, os.path.join(MODEL_DIR, "elo_history.pkl"))
    joblib.dump({"name": best_name, "results": {k: {m: v for m, v in vv.items() if m != "model"} for k, vv in results.items()}},
                os.path.join(MODEL_DIR, "model_comparison.pkl"))
    
    # Save team stats snapshot for prediction at inference time
    # Compute final team stats from full dataset
    final_feature_df = compute_team_stats(df, elo_ratings)
    
    # Build a team stats lookup (last known stats per team)
    team_stats = {}
    for team in set(list(df["home_team"].unique()) + list(df["away_team"].unique())):
        home_rows = df[df["home_team"] == team]
        away_rows = df[df["away_team"] == team]
        
        all_rows = pd.concat([home_rows, away_rows]).sort_values("match_id")
        if len(all_rows) == 0:
            continue
        
        last_idx = all_rows.index[-1]
        if last_idx in final_feature_df.index:
            feat_row = final_feature_df.loc[last_idx]
            if all_rows.iloc[-1]["home_team"] == team:
                team_stats[team] = {
                    "win_rate": feat_row["h_win_rate"],
                    "avg_goals": feat_row["h_avg_goals"],
                    "avg_conceded": feat_row["h_avg_conceded"],
                    "form": feat_row["h_form"],
                    "wc_exp": feat_row["h_wc_exp"],
                    "matches": feat_row["h_matches"],
                    "elo": elo_ratings.get(team, 1500),
                }
            else:
                team_stats[team] = {
                    "win_rate": feat_row["a_win_rate"],
                    "avg_goals": feat_row["a_avg_goals"],
                    "avg_conceded": feat_row["a_avg_conceded"],
                    "form": feat_row["a_form"],
                    "wc_exp": feat_row["a_wc_exp"],
                    "matches": feat_row["a_matches"],
                    "elo": elo_ratings.get(team, 1500),
                }
    
    joblib.dump(team_stats, os.path.join(MODEL_DIR, "team_stats.pkl"))
    joblib.dump(df, os.path.join(MODEL_DIR, "matches_df.pkl"))
    
    print(f"  ✓ Saved to {MODEL_DIR}/")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    return best_model, elo_ratings, team_stats


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    train()
