"""
Dynamic Elo Rating System for international football teams.
"""
import pandas as pd
import numpy as np


BASE_ELO = 1500
K_FACTOR_BASE = 32
K_FACTORS = {
    "World Cup": 60,
    "Final": 60,
    "Semi-finals": 55,
    "Quarter-finals": 50,
    "Round of 16": 45,
    "Group Stage": 40,
    "Qualifier": 30,
    "Friendly": 20,
}


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for team A given ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, stage: str = "Friendly") -> tuple:
    """Update Elo ratings after a match."""
    k = K_FACTORS.get(stage, 30)
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a
    
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - exp_b)
    
    return new_a, new_b


def calculate_elo_ratings(df: pd.DataFrame) -> dict:
    """
    Calculate Elo ratings for all teams across the entire dataset.
    Returns a dict of {team: final_elo}.
    Also returns a history dict for charting.
    """
    ratings = {}
    history = {}  # team -> list of (year, elo)
    
    df_sorted = df.sort_values(["year", "match_id"]).reset_index(drop=True)
    
    for _, row in df_sorted.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        
        # Initialize teams
        if home not in ratings:
            ratings[home] = BASE_ELO
            history[home] = []
        if away not in ratings:
            ratings[away] = BASE_ELO
            history[away] = []
        
        r_home = ratings[home]
        r_away = ratings[away]
        
        # Determine match outcome
        if row["home_goals"] > row["away_goals"]:
            score = 1.0
        elif row["home_goals"] == row["away_goals"]:
            score = 0.5
        else:
            score = 0.0
        
        stage = str(row.get("stage", "Friendly"))
        new_home, new_away = update_elo(r_home, r_away, score, stage)
        
        ratings[home] = new_home
        ratings[away] = new_away
        
        history[home].append({"year": row["year"], "elo": new_home, "match_id": row["match_id"]})
        history[away].append({"year": row["year"], "elo": new_away, "match_id": row["match_id"]})
    
    return ratings, history


def get_elo_dataframe(history: dict) -> pd.DataFrame:
    """Convert history dict to a flat DataFrame for charting."""
    rows = []
    for team, records in history.items():
        for r in records:
            rows.append({"team": team, "year": r["year"], "elo": r["elo"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    df = pd.read_csv("../data/matches.csv")
    ratings, history = calculate_elo_ratings(df)
    top = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 Teams by Elo:")
    for team, elo in top:
        print(f"  {team}: {elo:.0f}")
