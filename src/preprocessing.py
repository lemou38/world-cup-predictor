"""
Data preprocessing for FIFA World Cup Predictor.
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


# Team name normalization map
TEAM_NAME_MAP = {
    "West Germany": "Germany",
    "FR Germany": "Germany",
    "German DR": "Germany",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
    "ZR of Congo": "Cameroon",
    "Zaire": "Cameroon",
    "North Ireland": "Ireland",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "Trinidad Tobago": "Trinidad and Tobago",
    "C.d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "USA": "USA",
    "United States": "USA",
}


def normalize_team_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load and clean the match dataset."""
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Normalize team names
    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)
    
    # Fill missing values
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").fillna(0).astype(int)
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").fillna(0).astype(int)
    df["neutral"] = df["neutral"].fillna(False).astype(bool)
    df["stage"] = df["stage"].fillna("Friendly")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(2000).astype(int)
    
    # Create result target: 0 = away win, 1 = draw, 2 = home win
    def get_result(row):
        if row["home_goals"] > row["away_goals"]:
            return 2
        elif row["home_goals"] == row["away_goals"]:
            return 1
        else:
            return 0
    
    df["result"] = df.apply(get_result, axis=1)
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    
    # Sort by year then match_id
    if "match_id" in df.columns:
        df = df.sort_values(["year", "match_id"]).reset_index(drop=True)
    else:
        df = df.sort_values("year").reset_index(drop=True)
        df["match_id"] = range(len(df))
    
    return df


if __name__ == "__main__":
    df = load_and_clean("../data/matches.csv")
    print(f"Loaded {len(df)} matches")
    print(df["result"].value_counts())
    print(df.head())
