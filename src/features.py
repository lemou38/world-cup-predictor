"""
Advanced feature engineering for FIFA World Cup Predictor.
"""
import pandas as pd
import numpy as np
from collections import defaultdict


def compute_team_stats(df: pd.DataFrame, elo_ratings: dict) -> pd.DataFrame:
    """
    Compute advanced features for each match:
    - Recent form (last 5 matches)
    - Win rate
    - Avg goals scored / conceded
    - World Cup experience
    - Home advantage
    - Elo difference
    - Ranking difference (approximated by Elo)
    """
    df = df.copy()
    
    # Running stats per team
    team_wins = defaultdict(list)
    team_goals_scored = defaultdict(list)
    team_goals_conceded = defaultdict(list)
    team_wc_matches = defaultdict(int)
    
    # Features to compute
    features = []
    
    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        year = row["year"]
        stage = str(row.get("stage", "Friendly"))
        
        # --- Home team stats ---
        h_matches = len(team_wins[home])
        h_recent_wins = team_wins[home][-5:] if h_matches > 0 else []
        h_recent_goals = team_goals_scored[home][-5:] if h_matches > 0 else []
        h_recent_conceded = team_goals_conceded[home][-5:] if h_matches > 0 else []
        
        h_win_rate = np.mean(h_recent_wins) if h_recent_wins else 0.33
        h_avg_goals = np.mean(h_recent_goals) if h_recent_goals else 1.2
        h_avg_conceded = np.mean(h_recent_conceded) if h_recent_conceded else 1.2
        h_form = sum(h_recent_wins[-3:]) if len(h_recent_wins) >= 3 else (sum(h_recent_wins) if h_recent_wins else 1)
        h_wc_exp = min(team_wc_matches[home], 50) / 50.0
        
        # --- Away team stats ---
        a_matches = len(team_wins[away])
        a_recent_wins = team_wins[away][-5:] if a_matches > 0 else []
        a_recent_goals = team_goals_scored[away][-5:] if a_matches > 0 else []
        a_recent_conceded = team_goals_conceded[away][-5:] if a_matches > 0 else []
        
        a_win_rate = np.mean(a_recent_wins) if a_recent_wins else 0.33
        a_avg_goals = np.mean(a_recent_goals) if a_recent_goals else 1.2
        a_avg_conceded = np.mean(a_recent_conceded) if a_recent_conceded else 1.2
        a_form = sum(a_recent_wins[-3:]) if len(a_recent_wins) >= 3 else (sum(a_recent_wins) if a_recent_wins else 1)
        a_wc_exp = min(team_wc_matches[away], 50) / 50.0
        
        # Elo ratings
        h_elo = elo_ratings.get(home, 1500)
        a_elo = elo_ratings.get(away, 1500)
        elo_diff = h_elo - a_elo
        
        # Home advantage
        home_advantage = 0 if row.get("neutral", False) else 1
        
        # Is World Cup?
        is_wc = 1 if stage not in ["Friendly", "Qualifier"] else 0
        
        feat = {
            # Home features
            "h_win_rate": h_win_rate,
            "h_avg_goals": h_avg_goals,
            "h_avg_conceded": h_avg_conceded,
            "h_form": h_form,
            "h_wc_exp": h_wc_exp,
            "h_elo": h_elo,
            "h_matches": min(h_matches, 200) / 200.0,
            # Away features
            "a_win_rate": a_win_rate,
            "a_avg_goals": a_avg_goals,
            "a_avg_conceded": a_avg_conceded,
            "a_form": a_form,
            "a_wc_exp": a_wc_exp,
            "a_elo": a_elo,
            "a_matches": min(a_matches, 200) / 200.0,
            # Differential features
            "elo_diff": elo_diff,
            "win_rate_diff": h_win_rate - a_win_rate,
            "form_diff": h_form - a_form,
            "goals_diff": h_avg_goals - a_avg_goals,
            "conceded_diff": h_avg_conceded - a_avg_conceded,
            "wc_exp_diff": h_wc_exp - a_wc_exp,
            # Context
            "home_advantage": home_advantage,
            "is_wc": is_wc,
            "year_normalized": (year - 1930) / 96.0,
        }
        features.append(feat)
        
        # Update running stats AFTER computing features (no leakage)
        h_won = 1 if row["home_goals"] > row["away_goals"] else (0.5 if row["home_goals"] == row["away_goals"] else 0)
        a_won = 1 - h_won
        
        team_wins[home].append(h_won)
        team_goals_scored[home].append(row["home_goals"])
        team_goals_conceded[home].append(row["away_goals"])
        
        team_wins[away].append(a_won)
        team_goals_scored[away].append(row["away_goals"])
        team_goals_conceded[away].append(row["home_goals"])
        
        if stage not in ["Friendly", "Qualifier"]:
            team_wc_matches[home] += 1
            team_wc_matches[away] += 1
    
    feature_df = pd.DataFrame(features, index=df.index)
    return feature_df


FEATURE_COLUMNS = [
    "h_win_rate", "h_avg_goals", "h_avg_conceded", "h_form", "h_wc_exp",
    "h_elo", "h_matches", "a_win_rate", "a_avg_goals", "a_avg_conceded",
    "a_form", "a_wc_exp", "a_elo", "a_matches", "elo_diff", "win_rate_diff",
    "form_diff", "goals_diff", "conceded_diff", "wc_exp_diff",
    "home_advantage", "is_wc", "year_normalized",
]
