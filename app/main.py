"""
API REST de predicción de churn.

    GET  /health   estado del servicio y versión del modelo cargado
    POST /predict  probabilidad de churn y nivel de riesgo de un cliente
    GET  /docs     documentación Swagger generada por FastAPI
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.inference.predictor import ChurnPredictor

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("churn-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Si el modelo no carga, el servicio arranca igual pero /health lo informa (503)
    try:
        app.state.predictor = ChurnPredictor()
        meta = app.state.predictor.metadata
        logger.info(
            "Servicio iniciado - modelo %s v%s (run_id=%s)",
            meta["model_name"], meta["model_version"], meta["run_id"],
        )
    except Exception:
        app.state.predictor = None
        logger.exception("No se pudo cargar el modelo")
    yield
    logger.info("Servicio detenido")


app = FastAPI(
    title="Customer Churn API",
    description="Estima la probabilidad de abandono de un cliente de telecomunicaciones.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Asigna un request_id a cada request y registra estado y latencia."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(status="unavailable", model_loaded=False).model_dump(),
        )
    meta = predictor.metadata
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=meta["model_name"],
        model_version=meta["model_version"],
        run_id=meta["run_id"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    request_id = request.state.request_id
    try:
        result = predictor.predict(customer.model_dump())
    except Exception:
        logger.exception("request_id=%s error en la inferencia", request_id)
        raise HTTPException(status_code=500, detail="Error interno al generar la predicción")

    logger.info(
        "request_id=%s prediction=%s probability=%.4f risk_level=%s model_version=%s",
        request_id, result["prediction"], result["probability"],
        result["risk_level"], result["model_version"],
    )
    return PredictionResponse(**result, request_id=request_id)
