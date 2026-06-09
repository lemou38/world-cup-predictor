# ⚽ FIFA World Cup 2026 AI Predictor

A premium, AI-powered match prediction and tournament simulation platform for the FIFA World Cup 2026.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![License](https://img.shields.io/badge/License-MIT-gold)

---

## 🌟 Features

| Feature | Description |
|---|---|
| **AI Match Prediction** | Predict any international match with probability breakdowns |
| **Elo Rating System** | Dynamic Elo ratings recalculated from all historical matches |
| **Monte Carlo Simulator** | Run 100–2000 full World Cup simulations |
| **Live Countdown** | Real-time countdown to WC2026 kick-off (June 11, 2026) |
| **Analytics Dashboard** | Goals distribution, top nations, Elo evolution charts |
| **Premium Dark UI** | Gold & dark theme with animated probability bars |

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourname/world_cup_predictor
cd world_cup_predictor
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Data & Train Model

```bash
# Generate synthetic training data
python data/generate_data.py

# Train all models and select the best
python src/train.py
```

### 3. Launch the App

```bash
streamlit run app/streamlit_app.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
world_cup_predictor/
├── data/
│   ├── generate_data.py    # Synthetic dataset generator
│   └── matches.csv         # Generated match data
├── src/
│   ├── preprocessing.py    # Data cleaning & normalization
│   ├── features.py         # Feature engineering
│   ├── elo.py              # Dynamic Elo rating system
│   ├── train.py            # Multi-model training pipeline
│   ├── predict.py          # Match prediction engine
│   └── simulator.py        # Tournament Monte Carlo simulator
├── models/
│   ├── best_model.pkl      # Serialized best model
│   ├── elo_ratings.pkl     # Final Elo ratings
│   ├── team_stats.pkl      # Team statistics snapshot
│   └── model_comparison.pkl
├── app/
│   └── streamlit_app.py    # Main Streamlit application
├── requirements.txt
└── README.md
```

---

## 🤖 Machine Learning Models

The training pipeline evaluates 6 models and auto-selects the best:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline, fast |
| Random Forest | Robust ensemble |
| Gradient Boosting | Sklearn GBM |
| **XGBoost** | Usually best performer |
| CatBoost | Strong on categorical |
| LightGBM | Fast gradient boosting |

**Evaluation metrics:** Accuracy, F1-score (weighted), Log Loss

---

## 📊 Feature Engineering

The model uses 23 features per match:

- **Elo ratings** (home + away + difference)
- **Recent form** — win rate over last 5 matches
- **Goals per game** — scored and conceded averages
- **World Cup experience** — normalized match count
- **Home advantage** — flag for neutral venue
- **Tournament context** — is it a WC match?
- **Year normalization** — historical era adjustment

---

## 🎲 Tournament Simulator

The World Cup 2026 format:
- **48 teams** in 12 groups of 4
- Top 2 per group + 8 best 3rd-place advance
- Standard knockout rounds through to the Final
- Penalty shootouts use Elo-weighted probabilities

Monte Carlo output shows:
- Championship probability per team
- Semi-final probability
- Sample knockout bracket

---

## 📡 Deployment

### Streamlit Community Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set main file: `app/streamlit_app.py`
4. Deploy!

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python data/generate_data.py && python src/train.py
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t wc-predictor .
docker run -p 8501:8501 wc-predictor
```

### Render / Railway

Set build command:
```
pip install -r requirements.txt && python data/generate_data.py && python src/train.py
```

Start command:
```
streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 📝 Example Prediction

```
Algeria vs Spain
───────────────────────────────
🇩🇿 Algeria Win:   14%  ░░░░░░░░░░░░░░░░░░░░
⚖️  Draw:          18%  ░░░░░░░░░░░░░░░░░░░░░░
🇪🇸 Spain Win:     68%  ████████████████████████████████████████

Predicted Score: 1-2
AI Confidence:   74%
```

---

## 📄 License

MIT License — free for personal and commercial use.

---

*Built with ❤️ and ⚽ — Powered by XGBoost, Elo, and Monte Carlo simulation*
