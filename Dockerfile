# Misma versión de Python con la que se entrenó el modelo (ver MLmodel)
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/customer-churn-model \
    LOG_LEVEL=INFO

WORKDIR /app

# Dependencias primero para aprovechar la caché de capas
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código de la aplicación y de inferencia (no se copian training ni datos)
COPY app/ app/
COPY src/__init__.py src/__init__.py
COPY src/inference/ src/inference/

# Modelo exportado desde el Model Registry (se obtiene con `dvc pull`)
COPY models/customer-churn-model/ models/customer-churn-model/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
