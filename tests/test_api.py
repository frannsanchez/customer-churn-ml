"""Tests de los endpoints de la API."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_health_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]
    assert body["run_id"]


def test_predict_valido(client, valid_customer):
    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "probability", "risk_level", "model_version", "request_id"}
    assert body["prediction"] in {"CHURN", "NO_CHURN"}
    assert isinstance(body["probability"], float)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["model_version"], str)
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_predict_respeta_request_id_recibido(client, valid_customer):
    response = client.post("/predict", json=valid_customer, headers={"X-Request-ID": "abc-123"})
    assert response.json()["request_id"] == "abc-123"


def test_predict_version_coincide_con_health(client, valid_customer):
    health = client.get("/health").json()
    prediction = client.post("/predict", json=valid_customer).json()
    assert prediction["model_version"] == health["model_version"]


def test_root_redirige_a_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_swagger_disponible(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("Contract", "Weekly"),          # categoría inexistente
        ("tenure", -1),                  # fuera de rango
        ("MonthlyCharges", "caro"),      # tipo incorrecto
        ("SeniorCitizen", 2),            # solo admite 0/1
    ],
)
def test_predict_valores_invalidos_devuelven_422(client, valid_customer, field, value):
    valid_customer[field] = value
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 422


def test_predict_campo_faltante_devuelve_422(client, valid_customer):
    del valid_customer["Contract"]
    assert client.post("/predict", json=valid_customer).status_code == 422


def test_predict_campo_extra_devuelve_422(client, valid_customer):
    valid_customer["customerID"] = "1234-ABCD"
    assert client.post("/predict", json=valid_customer).status_code == 422


def test_predict_body_vacio_devuelve_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_sin_modelo_devuelve_503(monkeypatch, tmp_path, valid_customer):
    """Si el modelo no se puede cargar, el servicio lo informa en vez de caerse."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(app) as no_model_client:
        assert no_model_client.get("/health").status_code == 503
        assert no_model_client.post("/predict", json=valid_customer).status_code == 503
