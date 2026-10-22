FROM python:3.12-slim

ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf NUMBA_CACHE_DIR=/tmp/numba PM_DB_PATH=/data/history.db

WORKDIR /app

# curl for the health check, libgomp1 because XGBoost needs it
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1) CPU-only PyTorch first (a much smaller image than the default GPU build)
COPY requirements-torch.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt

# 2) Everything else, at the exact versions from your machine
COPY requirements.txt requirements-xgb.txt ./
RUN pip install -r requirements.txt \
 && pip install --no-deps -r requirements-xgb.txt


# 3) Download the embedding model now, so the container never needs the internet for it
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

# 4) Our code, data and trained models
COPY configs ./configs
COPY src ./src
COPY dashboard ./dashboard
COPY data/processed ./data/processed
COPY data/manuals ./data/manuals
COPY artifacts ./artifacts
COPY start.sh ./start.sh

# Run as a normal user, not root; /data holds the alert history
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh
RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app:app /data
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 7860 8000 8501
CMD ["bash", "start.sh"]