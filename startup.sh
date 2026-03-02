
echo "Starting Wine Quality Classifier...."

PORT = "${PORT:8000}"
echo "Using PORT=${PORT}

exec python -m streamlit run app.py \
    --server.port="${PORT}"
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.gatherUsageStats=false