[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?logo=render&logoColor=white)](https://YOUR-RENDER-URL.onrender.com)

---
title: Predictive Maintenance
emoji: 🏭
sdk: docker
app_port: 7860
pinned: false
---

## 🌐 Live Demo

**👉 [Launch Live Dashboard](https://zifj5cfvx65phnbkvhunki.streamlit.app/)**

# 🏭 Predictive Maintenance System

Predicts **equipment failure risk** and **Remaining Useful Life (RUL)** from sensor data, detects
**anomalies** early, explains *why* with SHAP, and uses a **multi-agent GenAI layer (LangGraph + Groq,
grounded in equipment manuals via RAG)** to write maintenance recommendations. Includes a FastAPI
backend, a live Streamlit dashboard, and Docker deployment.

**Data:** NASA C-MAPSS turbofan engine degradation (FD001): 100 training engines run to failure,
100 test engines stopped early.

## Architecture

```
 Sensor data ─▶ Feature engineering ─▶ Models ─────────────▶ Detector (plain code)
 (C-MAPSS)      rolling mean/std/diff   Isolation Forest      risk, RUL, anomaly flags,
                scaling, engine-level   Autoencoder           rule-based urgency level
                split (no leakage)      XGBoost / RF risk              │
                                        LSTM RUL                       ▼
                                                    ┌──── LangGraph relay ────┐
                                      healthy? ───▶ │ skip LLM (all clear)    │
                                                    │ Explainer (SHAP + facts)│
                                                    │ Recommender (RAG on     │
                                                    │   manuals, cited IDs)   │
                                                    └───────────┬─────────────┘
                              SQLite history ◀── FastAPI ◀──────┘
                                                   ▲
                                          Streamlit dashboard (live replay, alerts, metrics)
```

**Design decisions**
- **Urgency is decided by code, not the LLM.** Policy rules (POL-01, POL-03) are deterministic. The LLM only explains.
- **The LLM is grounded.** It sees only model evidence and retrieved manual sections, must cite procedure IDs,
  and every cited ID is verified against what was retrieved.
- **Healthy machines skip the LLM**, which saves cost and latency.
- **No leakage:** train/validation split by engine, scaler fitted on training engines only, backward-looking features only.

## Results

| Task | Model | Result |
|---|---|---|
| Anomaly detection | Isolation Forest | Validation ROC-AUC 0.989, PR-AUC 0.943 |
| Anomaly detection | Autoencoder | Validation ROC-AUC 0.764, PR-AUC 0.468 (weaker; secondary signal) |
| Failure within 30 cycles | XGBoost | Validation PR-AUC 0.961; test (all rows) PR-AUC 0.788, F1 0.70 |
| Failure within 30 cycles | Random Forest | Validation PR-AUC 0.963; test (all rows) PR-AUC 0.803, F1 0.68 |
| RUL (capped at 125) | **LSTM** | Test (all rows) MAE 7.8, RMSE 12.7; last row MAE 11.2, RMSE 15.4 |
| RUL (capped at 125) | XGBoost regressor | Test (all rows) MAE 11.7, RMSE 16.6 |
| RUL | Always-guess-average baseline | Last-row MAE 34.8 |

Exact numbers are in `artifacts/*_metrics.json` and the dashboard's "Model metrics" tab.
Validation numbers are optimistic because validation data tunes thresholds and early stopping.
Test rows are the honest ones.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-xgb.txt
pip install torch                                      # CPU is enough
cp .env.example .env                                   # then add your GROQ_API_KEY
# put train_FD001.txt, test_FD001.txt, RUL_FD001.txt into data/raw/
python scripts/train_all.py                            # rebuilds every model
uvicorn src.api.main:app --port 8000                   # terminal 1
streamlit run dashboard/app.py                         # terminal 2  → http://localhost:8501
```

Tests: `pip install -r requirements-dev.txt && pytest -q`

## Docker

```bash
docker compose up --build        # API :8000, dashboard :8501, history persisted in a volume
```

Single-container mode (what the public demo uses): `docker build -t pm . && docker run -p 7860:7860 --env-file .env pm`

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status, LLM availability |
| `GET /machines` | Machines and last cycle |
| `POST /predict` | Risk, RUL, anomaly flags, urgency, SHAP drivers (no LLM) |
| `POST /analyze` | Full agent relay: adds explanation and cited actions |
| `GET /history`, `GET /alerts` | Logged predictions (alerts = Level 2+) |
| `GET /metrics` | Saved model metrics |

Interactive docs at `/docs`. Body example: `{"unit": 35, "cycle": 150}` (omit `cycle` for the latest).
`/analyze` is capped by `MAX_AI_PER_HOUR` (default 60).

## Project structure

```
configs/        settings            src/data/       loader
src/features/   feature engineering src/models/     anomaly, failure, RUL, detector
src/explain/    SHAP + names        src/genai/      RAG, explainer chain, LangGraph agents
src/api/        FastAPI + SQLite    dashboard/      Streamlit app
data/manuals/   fictional manuals   scripts/        train_all, freeze_requirements
```

## Limitations (please read)

- **Simulation, not a live plant:** the dashboard replays recorded engines cycle by cycle.
- **Single operating condition (FD001).** Multi-condition sets (FD002/FD004) need extra normalization.
- **Manuals are fictional training documents.** Real use needs the maker's manuals and expert review.
- **The Autoencoder is weak** on this smooth degradation; Isolation Forest is the primary detector.
- **"Drifting sensors" is a proxy:** the count of top SHAP drivers that raise risk, capped at 5.
  A better version would measure deviation from each sensor's healthy level.
- **RUL is capped at 125** (standard practice); early-life estimates are therefore coarse.
- **Estimates, not diagnoses.** Always confirm by physical inspection (policy POL-04).
- Test-set danger rows are few (about 330), so test metrics are noisy.

## Ideas for next steps
Multi-condition datasets, uncertainty intervals for RUL, drift monitoring, authentication, a real
sensor stream (Kafka/MQTT), and CI that runs the tests on every push.

Data: NASA Prognostics Center of Excellence. Manuals in `data/manuals/` are fictional.
