uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &

for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8000/health > /dev/null && break
  sleep 2
done

exec streamlit run dashboard/app.py \
  --server.port="${port:-7860}" --server.address=0.0.0.0 --server.headless=true \
  --server.enableCORS=false --server.enableXsrfProtection=false
