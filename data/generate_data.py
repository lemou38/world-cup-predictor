"""
Generate realistic FIFA World Cup match data for training.
This script creates a comprehensive dataset based on historical patterns.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# All World Cup participating nations (historical + current)
TEAMS = {
    # Top tier
    "Brazil": {"strength": 92, "wc_wins": 5, "region": "CONMEBOL"},
    "Germany": {"strength": 90, "wc_wins": 4, "region": "UEFA"},
    "Italy": {"strength": 88, "wc_wins": 4, "region": "UEFA"},
    "Argentina": {"strength": 91, "wc_wins": 3, "region": "CONMEBOL"},
    "France": {"strength": 90, "wc_wins": 2, "region": "UEFA"},
    "Spain": {"strength": 89, "wc_wins": 1, "region": "UEFA"},
    "England": {"strength": 83, "wc_wins": 1, "region": "UEFA"},
    "Uruguay": {"strength": 80, "wc_wins": 2, "region": "CONMEBOL"},
    # Second tier
    "Netherlands": {"strength": 85, "wc_wins": 0, "region": "UEFA"},
    "Portugal": {"strength": 85, "wc_wins": 0, "region": "UEFA"},
    "Belgium": {"strength": 84, "wc_wins": 0, "region": "UEFA"},
    "Croatia": {"strength": 82, "wc_wins": 0, "region": "UEFA"},
    "Denmark": {"strength": 79, "wc_wins": 0, "region": "UEFA"},
    "Sweden": {"strength": 78, "wc_wins": 0, "region": "UEFA"},
    "Switzerland": {"strength": 78, "wc_wins": 0, "region": "UEFA"},
    "Mexico": {"strength": 78, "wc_wins": 0, "region": "CONCACAF"},
    "Colombia": {"strength": 79, "wc_wins": 0, "region": "CONMEBOL"},
    "Chile": {"strength": 77, "wc_wins": 0, "region": "CONMEBOL"},
    "Poland": {"strength": 77, "wc_wins": 0, "region": "UEFA"},
    "USA": {"strength": 76, "wc_wins": 0, "region": "CONCACAF"},
    # Third tier
    "Morocco": {"strength": 78, "wc_wins": 0, "region": "CAF"},
    "Senegal": {"strength": 76, "wc_wins": 0, "region": "CAF"},
    "Nigeria": {"strength": 75, "wc_wins": 0, "region": "CAF"},
    "Cameroon": {"strength": 73, "wc_wins": 0, "region": "CAF"},
    "Ghana": {"strength": 72, "wc_wins": 0, "region": "CAF"},
    "Ivory Coast": {"strength": 73, "wc_wins": 0, "region": "CAF"},
    "South Korea": {"strength": 74, "wc_wins": 0, "region": "AFC"},
    "Japan": {"strength": 74, "wc_wins": 0, "region": "AFC"},
    "Iran": {"strength": 72, "wc_wins": 0, "region": "AFC"},
    "Australia": {"strength": 72, "wc_wins": 0, "region": "AFC"},
    "Saudi Arabia": {"strength": 70, "wc_wins": 0, "region": "AFC"},
    "Qatar": {"strength": 68, "wc_wins": 0, "region": "AFC"},
    "Algeria": {"strength": 74, "wc_wins": 0, "region": "CAF"},
    "Tunisia": {"strength": 70, "wc_wins": 0, "region": "CAF"},
    "Egypt": {"strength": 71, "wc_wins": 0, "region": "CAF"},
    "Serbia": {"strength": 76, "wc_wins": 0, "region": "UEFA"},
    "Czech Republic": {"strength": 75, "wc_wins": 0, "region": "UEFA"},
    "Hungary": {"strength": 73, "wc_wins": 0, "region": "UEFA"},
    "Austria": {"strength": 74, "wc_wins": 0, "region": "UEFA"},
    "Scotland": {"strength": 72, "wc_wins": 0, "region": "UEFA"},
    "Turkey": {"strength": 74, "wc_wins": 0, "region": "UEFA"},
    "Greece": {"strength": 72, "wc_wins": 0, "region": "UEFA"},
    "Romania": {"strength": 71, "wc_wins": 0, "region": "UEFA"},
    "Ecuador": {"strength": 72, "wc_wins": 0, "region": "CONMEBOL"},
    "Peru": {"strength": 72, "wc_wins": 0, "region": "CONMEBOL"},
    "Paraguay": {"strength": 70, "wc_wins": 0, "region": "CONMEBOL"},
    "Bolivia": {"strength": 66, "wc_wins": 0, "region": "CONMEBOL"},
    "Venezuela": {"strength": 65, "wc_wins": 0, "region": "CONMEBOL"},
    "Costa Rica": {"strength": 70, "wc_wins": 0, "region": "CONCACAF"},
    "Panama": {"strength": 65, "wc_wins": 0, "region": "CONCACAF"},
    "Jamaica": {"strength": 64, "wc_wins": 0, "region": "CONCACAF"},
    "New Zealand": {"strength": 63, "wc_wins": 0, "region": "OFC"},
    "China": {"strength": 66, "wc_wins": 0, "region": "AFC"},
    "Indonesia": {"strength": 63, "wc_wins": 0, "region": "AFC"},
    "Ukraine": {"strength": 75, "wc_wins": 0, "region": "UEFA"},
    "Slovakia": {"strength": 73, "wc_wins": 0, "region": "UEFA"},
    "Slovenia": {"strength": 72, "wc_wins": 0, "region": "UEFA"},
    "Ireland": {"strength": 71, "wc_wins": 0, "region": "UEFA"},
    "Norway": {"strength": 75, "wc_wins": 0, "region": "UEFA"},
    "Finland": {"strength": 70, "wc_wins": 0, "region": "UEFA"},
    "Wales": {"strength": 74, "wc_wins": 0, "region": "UEFA"},
    "Russia": {"strength": 72, "wc_wins": 0, "region": "UEFA"},
    "Canada": {"strength": 73, "wc_wins": 0, "region": "CONCACAF"},
    "Honduras": {"strength": 66, "wc_wins": 0, "region": "CONCACAF"},
    "Trinidad and Tobago": {"strength": 64, "wc_wins": 0, "region": "CONCACAF"},
}

team_list = list(TEAMS.keys())


def simulate_match(home_team, away_team, year, neutral=False):
    """Simulate a match result based on team strengths."""
    h_str = TEAMS[home_team]["strength"]
    a_str = TEAMS[away_team]["strength"]
    
    # Home advantage (slightly less at neutral venues)
    home_adv = 3 if not neutral else 1
    
    # Year-based adjustments (teams improve/decline over time)
    year_factor = (year - 1930) / 96.0
    
    h_str = h_str + home_adv + np.random.normal(0, 5)
    a_str = a_str + np.random.normal(0, 5)
    
    diff = (h_str - a_str) / 20.0
    
    # Poisson-based goal simulation
    home_lambda = max(0.3, 1.5 + diff * 0.8 + np.random.normal(0, 0.3))
    away_lambda = max(0.3, 1.2 - diff * 0.6 + np.random.normal(0, 0.3))
    
    home_goals = np.random.poisson(home_lambda)
    away_goals = np.random.poisson(away_lambda)
    
    return int(home_goals), int(away_goals)


def generate_dataset():
    """Generate a comprehensive dataset of international matches."""
    records = []
    
    # World Cup years
    wc_years = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970,
                1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006,
                2010, 2014, 2018, 2022]
    
    match_id = 1
    
    for year in wc_years:
        # Select 32 teams (fewer for earlier tournaments)
        n_teams = 13 if year < 1950 else (16 if year < 1982 else (24 if year < 1998 else 32))
        teams_in_wc = np.random.choice(team_list, min(n_teams, len(team_list)), replace=False)
        
        # Group stage
        groups = [teams_in_wc[i:i+4] for i in range(0, len(teams_in_wc), 4)]
        
        for group_idx, group in enumerate(groups):
            group_letter = chr(65 + group_idx)
            # Round robin within group
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    h_goals, a_goals = simulate_match(group[i], group[j], year, neutral=True)
                    records.append({
                        "match_id": match_id,
                        "year": year,
                        "stage": "Group Stage",
                        "group": group_letter,
                        "home_team": group[i],
                        "away_team": group[j],
                        "home_goals": h_goals,
                        "away_goals": a_goals,
                        "neutral": True,
                    })
                    match_id += 1
        
        # Knockout rounds
        stages = ["Round of 16", "Quarter-finals", "Semi-finals", "Final"]
        ko_teams = list(teams_in_wc[:16]) if len(teams_in_wc) >= 16 else list(teams_in_wc)
        
        for stage in stages:
            if len(ko_teams) < 2:
                break
            next_round = []
            np.random.shuffle(ko_teams)
            for i in range(0, len(ko_teams)-1, 2):
                h_goals, a_goals = simulate_match(ko_teams[i], ko_teams[i+1], year, neutral=True)
                records.append({
                    "match_id": match_id,
                    "year": year,
                    "stage": stage,
                    "group": None,
                    "home_team": ko_teams[i],
                    "away_team": ko_teams[i+1],
                    "home_goals": h_goals,
                    "away_goals": a_goals,
                    "neutral": True,
                })
                match_id += 1
                winner = ko_teams[i] if h_goals >= a_goals else ko_teams[i+1]
                next_round.append(winner)
            ko_teams = next_round
    
    # Also add friendly/qualifier matches for more data
    for _ in range(2000):
        t1, t2 = np.random.choice(team_list, 2, replace=False)
        year = np.random.randint(1990, 2024)
        h_goals, a_goals = simulate_match(t1, t2, year, neutral=False)
        records.append({
            "match_id": match_id,
            "year": year,
            "stage": "Friendly" if np.random.random() > 0.5 else "Qualifier",
            "group": None,
            "home_team": t1,
            "away_team": t2,
            "home_goals": h_goals,
            "away_goals": a_goals,
            "neutral": False,
        })
        match_id += 1
    
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Generating FIFA match dataset...")
    df = generate_dataset()
    df.to_csv("data/matches.csv", index=False)
    print(f"Dataset generated: {len(df)} matches")
    print(df.head())
    print(df["stage"].value_counts())
