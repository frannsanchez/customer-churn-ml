import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import CustomerFeatures
from src.inference.predictor import ChurnPredictor


@pytest.fixture(scope="session")
def predictor():
    return ChurnPredictor()


@pytest.fixture(scope="session")
def client():
    # El context manager ejecuta el lifespan (carga del modelo)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer() -> dict:
    return dict(CustomerFeatures.model_config["json_schema_extra"]["example"])
