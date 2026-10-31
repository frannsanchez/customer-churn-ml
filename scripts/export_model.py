"""
Exporta una versión del Model Registry (DagsHub/MLflow) a models/ para que la API
la sirva sin depender de MLflow ni de credenciales en tiempo de ejecución.

Junto al modelo se escribe metadata.json con el lineage completo:
versión registrada -> Run de MLflow -> commit de Git -> versión de datos (md5 DVC).

Uso:
    python -m scripts.export_model --version 1
    dvc add models/customer-churn-model && dvc push
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

from src.training.train import REGISTERED_MODEL_NAME

OUTPUT_DIR = Path("models/customer-churn-model")


def export_model(version: str, output_dir: Path = OUTPUT_DIR) -> dict:
    client = MlflowClient()
    model_version = client.get_model_version(REGISTERED_MODEL_NAME, version)
    run = client.get_run(model_version.run_id)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        f"models:/{REGISTERED_MODEL_NAME}/{version}", dst_path=str(output_dir)
    )

    metadata = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(model_version.version),
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "git_commit": run.data.tags.get("mlflow.source.git.commit"),
        "data_md5": run.data.tags.get("data_md5"),
        "metrics": run.data.metrics,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Versión del Model Registry a exportar")
    args = parser.parse_args()

    load_dotenv()
    metadata = export_model(args.version)
    print(json.dumps(metadata, indent=2))
    print(f"\nModelo exportado en {OUTPUT_DIR}. Versionarlo con:")
    print(f"  dvc add {OUTPUT_DIR} && dvc push")


if __name__ == "__main__":
    main()
