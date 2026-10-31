"""
Capa de inferencia: carga el pipeline exportado (preprocessing + estimador) y
traduce la probabilidad de churn a una predicción y un nivel de riesgo.

Es independiente de FastAPI para poder testearla y reutilizarla por separado.
"""
import json
import os
from pathlib import Path

import pandas as pd
import skops.io as sio

DEFAULT_MODEL_DIR = "models/customer-churn-model"
POSITIVE_CLASS = "Yes"
# Mismos tipos declarados como confiables al loguear el modelo en train.py
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]

# Umbral de decisión CHURN / NO_CHURN. El modelo se entrenó con
# class_weight="balanced", por lo que 0.5 ya favorece el recall (≈0.72 en test).
DECISION_THRESHOLD = 0.5

# Umbrales de riesgo definidos sobre el set de test (churn real 26%):
#   LOW    p < 0.40  -> 52% de clientes, churn real  9%
#   MEDIUM 0.40-0.70 -> 26% de clientes, churn real 32%
#   HIGH   p >= 0.70 -> 21% de clientes, churn real 61% (concentra el 50% de los churners)
RISK_MEDIUM_THRESHOLD = 0.40
RISK_HIGH_THRESHOLD = 0.70


def get_risk_level(probability: float) -> str:
    if probability >= RISK_HIGH_THRESHOLD:
        return "HIGH"
    if probability >= RISK_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


class ChurnPredictor:
    def __init__(self, model_dir: str | Path | None = None):
        model_dir = Path(model_dir or os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
        model_path = model_dir / "model.skops"
        # skops solo carga los tipos explícitamente confiables (no ejecuta pickle arbitrario)
        self.pipeline = sio.load(model_path, trusted=SKOPS_TRUSTED_TYPES)
        self.metadata = json.loads((model_dir / "metadata.json").read_text())

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"]

    def predict(self, customer: dict) -> dict:
        X = pd.DataFrame([customer])
        classes = list(self.pipeline.classes_)
        probability = float(self.pipeline.predict_proba(X)[0, classes.index(POSITIVE_CLASS)])
        return {
            "prediction": "CHURN" if probability >= DECISION_THRESHOLD else "NO_CHURN",
            "probability": round(probability, 4),
            "risk_level": get_risk_level(probability),
            "model_version": self.model_version,
        }
