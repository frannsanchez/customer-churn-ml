"""Tests de la capa de inferencia, independientes de la API."""
import pytest

from src.inference.predictor import ChurnPredictor, get_risk_level


def test_model_loads_with_metadata(predictor):
    assert predictor.pipeline is not None
    assert predictor.model_version
    assert predictor.metadata["run_id"]
    assert predictor.metadata["model_name"] == "customer-churn-model"


def test_model_dir_invalido_falla(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChurnPredictor(model_dir=tmp_path)


def test_valid_prediction_structure(predictor, valid_customer):
    result = predictor.predict(valid_customer)

    assert set(result) == {"prediction", "probability", "risk_level", "model_version"}
    assert result["prediction"] in {"CHURN", "NO_CHURN"}
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["model_version"] == predictor.model_version


def test_prediction_consistente_con_threshold(predictor, valid_customer):
    result = predictor.predict(valid_customer)
    expected = "CHURN" if result["probability"] >= 0.5 else "NO_CHURN"
    assert result["prediction"] == expected


def test_total_charges_faltante_se_imputa(predictor, valid_customer):
    valid_customer["TotalCharges"] = None
    result = predictor.predict(valid_customer)
    assert 0.0 <= result["probability"] <= 1.0


def test_contrato_largo_tiene_menor_riesgo(predictor, valid_customer):
    """Sanity check de negocio: un contrato a 2 años con antigüedad reduce el riesgo."""
    nuevo = predictor.predict(valid_customer)["probability"]
    fiel = predictor.predict(
        {**valid_customer, "Contract": "Two year", "tenure": 60, "TotalCharges": 5000.0}
    )["probability"]
    assert fiel < nuevo


@pytest.mark.parametrize(
    "probability, expected",
    [(0.0, "LOW"), (0.39, "LOW"), (0.40, "MEDIUM"), (0.69, "MEDIUM"), (0.70, "HIGH"), (1.0, "HIGH")],
)
def test_risk_level_thresholds(probability, expected):
    assert get_risk_level(probability) == expected
