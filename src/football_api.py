import requests
import os

API_KEY = os.getenv("API_FOOTBALL_KEY")

BASE_URL = "https://v3.football.api-sports.io"

headers = {
    "x-apisports-key": API_KEY
}


def get_last_matches(team_name, last=10):
    url = f"{BASE_URL}/fixtures?team={team_name}&last={last}"
    r = requests.get(url, headers=headers)
    return r.json()


def get_team_stats(team_id, season=2025):
    url = f"{BASE_URL}/teams/statistics?team={team_id}&season={season}"
    r = requests.get(url, headers=headers)
    return r.json()