"""
FIFA World Cup 2026 Tournament Simulator with Monte Carlo analysis.
"""
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


# 2026 World Cup format: 48 teams, 12 groups of 4
WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Denmark"],
    "B": ["Canada", "Italy", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Ukraine", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Jamaica", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Panama", "Ghana"]
}


class TournamentSimulator:
    def __init__(self, predictor):
        self.predictor = predictor
    
    def simulate_group_stage(self, groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Simulate group stage. Returns dict of group -> [1st, 2nd, 3rd, 4th].
        Top 2 from each group + 8 best 3rd-place teams advance.
        """
        group_standings = {}
        
        for group_letter, teams in groups.items():
            points = defaultdict(int)
            goals_scored = defaultdict(int)
            goals_conceded = defaultdict(int)
            
            # Round robin
            for i in range(len(teams)):
                for j in range(i+1, len(teams)):
                    t1, t2 = teams[i], teams[j]
                    result = self.predictor.predict(t1, t2, neutral=True)
                    
                    # Sample outcome
                    r = np.random.random() * 100
                    if r < result["prob_home_win"]:
                        points[t1] += 3
                        g1 = max(1, int(np.random.poisson(result["expected_home_goals"])))
                        g2 = max(0, g1 - 1 - int(np.random.poisson(0.3)))
                    elif r < result["prob_home_win"] + result["prob_draw"]:
                        points[t1] += 1
                        points[t2] += 1
                        g1 = g2 = max(0, int(np.random.poisson(1.0)))
                    else:
                        points[t2] += 3
                        g2 = max(1, int(np.random.poisson(result["expected_away_goals"])))
                        g1 = max(0, g2 - 1 - int(np.random.poisson(0.3)))
                    
                    goals_scored[t1] += g1
                    goals_conceded[t1] += g2
                    goals_scored[t2] += g2
                    goals_conceded[t2] += g1
            
            # Sort by points, then goal difference, then goals scored
            sorted_teams = sorted(teams, key=lambda t: (
                points[t], goals_scored[t] - goals_conceded[t], goals_scored[t]
            ), reverse=True)
            
            group_standings[group_letter] = sorted_teams
        
        return group_standings
    
    def simulate_knockout(self, bracket: List[str]) -> Tuple[str, List[dict]]:
        """
        Simulate a knockout bracket. Returns (winner, match_results).
        """
        teams = list(bracket)
        round_names = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
        all_results = []
        round_idx = 0
        
        while len(teams) > 1:
            next_round = []
            round_name = round_names[min(round_idx, len(round_names)-1)]
            
            for i in range(0, len(teams) - 1, 2):
                t1, t2 = teams[i], teams[i+1]
                winner = self.predictor.simulate_single_match(t1, t2)
                all_results.append({"round": round_name, "team1": t1, "team2": t2, "winner": winner})
                next_round.append(winner)
            
            # Handle odd number (bye)
            if len(teams) % 2 == 1:
                next_round.append(teams[-1])
            
            teams = next_round
            round_idx += 1
        
        return teams[0], all_results
    
    def run_full_simulation(self, groups: Dict[str, List[str]] = None) -> dict:
        """Run a complete World Cup simulation."""
        if groups is None:
            groups = WC2026_GROUPS
        
        # Group stage
        standings = self.simulate_group_stage(groups)
        
        # Build knockout bracket (top 2 per group + best 3rd place)
        qualifiers = []
        for g in sorted(standings.keys()):
            qualifiers.extend(standings[g][:2])  # Top 2
        
        # Add best 3rd-place (simplified: take 3rd from first 8 groups)
        third_place = [standings[g][2] for g in sorted(standings.keys())[:8]]
        
        # Evaluate 3rd place by Elo
        elo_ratings = self.predictor.elo_ratings
        third_place_sorted = sorted(third_place, key=lambda t: elo_ratings.get(t, 1500), reverse=True)
        qualifiers.extend(third_place_sorted[:8])  # Best 8 of 12 third-place
        
        # Shuffle and pad to 32
        np.random.shuffle(qualifiers)
        while len(qualifiers) < 32:
            qualifiers.append(qualifiers[0])
        qualifiers = qualifiers[:32]
        
        # Knockout
        champion, ko_results = self.simulate_knockout(qualifiers)
        
        return {
            "group_standings": standings,
            "knockout_bracket": ko_results,
            "champion": champion,
        }
    
    def monte_carlo(self, n_simulations: int = 1000, groups: Dict[str, List[str]] = None,
                    progress_callback=None) -> Dict[str, float]:
        """
        Run Monte Carlo simulations and return championship probabilities.
        Returns: {team: probability_percent}
        """
        if groups is None:
            groups = WC2026_GROUPS
        
        champion_counts = defaultdict(int)
        semifinal_counts = defaultdict(int)
        finalist_counts = defaultdict(int)
        
        for i in range(n_simulations):
            if progress_callback and i % 50 == 0:
                progress_callback(i / n_simulations)
            
            result = self.run_full_simulation(groups)
            champion_counts[result["champion"]] += 1
            
            # Track semi-finalists and finalists
            for match in result["knockout_bracket"]:
                if match["round"] == "Semi-finals":
                    semifinal_counts[match["team1"]] += 1
                    semifinal_counts[match["team2"]] += 1
                if match["round"] == "Final":
                    finalist_counts[match["team1"]] += 1
                    finalist_counts[match["team2"]] += 1
        
        if progress_callback:
            progress_callback(1.0)
        
        all_teams = set(list(champion_counts.keys()) + list(semifinal_counts.keys()))
        
        return {
            "championship_probs": {t: round(c / n_simulations * 100, 2) for t, c in champion_counts.items()},
            "semifinal_probs": {t: round(c / (n_simulations * 2) * 100, 2) for t, c in semifinal_counts.items()},
            "finalist_probs": {t: round(c / (n_simulations * 2) * 100, 2) for t, c in finalist_counts.items()},
            "n_simulations": n_simulations,
        }


if __name__ == "__main__":
    from predict import get_predictor
    p = get_predictor()
    sim = TournamentSimulator(p)
    
    print("Running 100 simulations...")
    results = sim.monte_carlo(100)
    
    top = sorted(results["championship_probs"].items(), key=lambda x: x[1], reverse=True)[:8]
    print("\nTop 8 Championship Probabilities:")
    for team, prob in top:
        bar = "█" * int(prob / 2)
        print(f"  {team:20s} {prob:5.1f}% {bar}")
