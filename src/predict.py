"""
Prediction engine for FIFA World Cup match outcomes.
"""
import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_COLUMNS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class MatchPredictor:
    def __init__(self):
        self.model = None
        self.elo_ratings = {}
        self.team_stats = {}
        self._loaded = False
    
    def load(self):
        """Load model and team stats from disk."""
        model_path = os.path.join(MODEL_DIR, "best_model.pkl")
        elo_path = os.path.join(MODEL_DIR, "elo_ratings.pkl")
        stats_path = os.path.join(MODEL_DIR, "team_stats.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please run: python src/train.py"
            )
        
        self.model = joblib.load(model_path)
        self.elo_ratings = joblib.load(elo_path)
        self.team_stats = joblib.load(stats_path)
        self._loaded = True
    
    def _get_team_features(self, team: str) -> dict:
        """Get cached team stats or defaults."""
        if team in self.team_stats:
            return self.team_stats[team]
        return {
            "win_rate": 0.33,
            "avg_goals": 1.2,
            "avg_conceded": 1.2,
            "form": 1.5,
            "wc_exp": 0.3,
            "matches": 0.5,
            "elo": self.elo_ratings.get(team, 1500),
        }
    
    def _build_feature_vector(self, team_a: str, team_b: str, neutral: bool = True) -> np.ndarray:
        """Build a feature vector for a match prediction."""
        h = self._get_team_features(team_a)
        a = self._get_team_features(team_b)
        
        h_elo = self.elo_ratings.get(team_a, h["elo"])
        a_elo = self.elo_ratings.get(team_b, a["elo"])
        
        features = {
            "h_win_rate": h["win_rate"],
            "h_avg_goals": h["avg_goals"],
            "h_avg_conceded": h["avg_conceded"],
            "h_form": h["form"],
            "h_wc_exp": h["wc_exp"],
            "h_elo": h_elo,
            "h_matches": h["matches"],
            "a_win_rate": a["win_rate"],
            "a_avg_goals": a["avg_goals"],
            "a_avg_conceded": a["avg_conceded"],
            "a_form": a["form"],
            "a_wc_exp": a["wc_exp"],
            "a_elo": a_elo,
            "a_matches": a["matches"],
            "elo_diff": h_elo - a_elo,
            "win_rate_diff": h["win_rate"] - a["win_rate"],
            "form_diff": h["form"] - a["form"],
            "goals_diff": h["avg_goals"] - a["avg_goals"],
            "conceded_diff": h["avg_conceded"] - a["avg_conceded"],
            "wc_exp_diff": h["wc_exp"] - a["wc_exp"],
            "home_advantage": 0 if neutral else 1,
            "is_wc": 1,
            "year_normalized": 0.97,  # 2026
        }
        return np.array([features[c] for c in FEATURE_COLUMNS]).reshape(1, -1)
    
    def predict(self, team_a: str, team_b: str, neutral: bool = True) -> dict:
        """
        Predict match outcome between team_a and team_b.
        Returns probabilities and estimated score.
        """
        if not self._loaded:
            self.load()
        
        X = self._build_feature_vector(team_a, team_b, neutral)
        proba = self.model.predict_proba(X)[0]
        
        # Classes: 0=away win, 1=draw, 2=home win
        classes = list(self.model.classes_)
        prob_away = proba[classes.index(0)] if 0 in classes else 0.25
        prob_draw = proba[classes.index(1)] if 1 in classes else 0.25
        prob_home = proba[classes.index(2)] if 2 in classes else 0.50
        
        # Estimate goals using Poisson-like model
        h_stats = self._get_team_features(team_a)
        a_stats = self._get_team_features(team_b)
        
        elo_diff = (self.elo_ratings.get(team_a, 1500) - self.elo_ratings.get(team_b, 1500)) / 400.0
        
        base_home = max(0.5, h_stats["avg_goals"] - a_stats["avg_conceded"] / 2 + elo_diff * 0.5 + 0.3)
        base_away = max(0.5, a_stats["avg_goals"] - h_stats["avg_conceded"] / 2 - elo_diff * 0.5)
        
        # Confidence: how far the top probability is from uniform (0.33)
        max_prob = max(prob_home, prob_draw, prob_away)
        confidence = min(100, int((max_prob - 0.33) / 0.67 * 100) + 30)
        
        return {
            "team_a": team_a,
            "team_b": team_b,
            "prob_home_win": round(prob_home * 100, 1),
            "prob_draw": round(prob_draw * 100, 1),
            "prob_away_win": round(prob_away * 100, 1),
            "expected_home_goals": round(base_home, 1),
            "expected_away_goals": round(base_away, 1),
            "predicted_score": f"{round(base_home)}-{round(base_away)}",
            "confidence": confidence,
            "elo_home": round(self.elo_ratings.get(team_a, 1500)),
            "elo_away": round(self.elo_ratings.get(team_b, 1500)),
        }
    
    def simulate_single_match(self, team_a: str, team_b: str) -> str:
        """
        Simulate a single knockout match, returns the winner.
        In case of draw, uses penalty probabilities weighted by Elo.
        """
        result = self.predict(team_a, team_b, neutral=True)
        r = np.random.random() * 100
        
        if r < result["prob_home_win"]:
            return team_a
        elif r < result["prob_home_win"] + result["prob_draw"]:
            # Penalty shootout — slightly favor higher Elo
            elo_h = self.elo_ratings.get(team_a, 1500)
            elo_a = self.elo_ratings.get(team_b, 1500)
            pen_prob = 0.5 + (elo_h - elo_a) / 2000.0
            return team_a if np.random.random() < pen_prob else team_b
        else:
            return team_b
    
    @property
    def all_teams(self):
        if not self._loaded:
            self.load()
        return sorted(set(list(self.elo_ratings.keys()) + list(self.team_stats.keys())))


# Singleton predictor
_predictor = None

def get_predictor() -> MatchPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MatchPredictor()
        _predictor.load()
    return _predictor


if __name__ == "__main__":
    p = get_predictor()
    result = p.predict("Algeria", "Spain")
    print("\nAlgeria vs Spain:")
    for k, v in result.items():
        print(f"  {k}: {v}")
