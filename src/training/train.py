import argparse
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data.crear_dataset import load_raw_data, get_train_test_indices, get_xy, RANDOM_STATE
from src.features.build_features import build_preprocessor
from src.evaluation.evaluate import compute_metrics, print_metrics, print_confusion_matrix
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/customer_churn_historical.csv"
EXPERIMENT_NAME = "customer-churn"

# Réplica de los 10 experimentos trabajados en el notebook exploratorio.
MODEL_CONFIGS = [
    {
        "name": "exp1_decision_tree_base",
        "categorical_features": ["Partner", "Dependents", "PaperlessBilling"],
        "numeric_features": [],
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
    },
    {
        "name": "exp2_baseline_dummy",
        "categorical_features": ["Partner", "Dependents", "PaperlessBilling"],
        "numeric_features": [],
        "estimator": DummyClassifier(strategy="most_frequent"),
    },
    {
        "name": "exp3_decision_tree_totalcharges",
        "categorical_features": ["Partner", "Dependents", "PaperlessBilling"],
        "numeric_features": ["TotalCharges"],
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
    },
    {
        "name": "exp4_logistic_regression",
        "categorical_features": ["Partner", "Dependents", "PaperlessBilling"],
        "numeric_features": ["TotalCharges"],
        "estimator": LogisticRegression(max_iter=1000),
    },
    {
        "name": "exp5_decision_tree_monthlycharges",
        "categorical_features": ["Partner", "Dependents", "PaperlessBilling"],
        "numeric_features": ["TotalCharges", "MonthlyCharges"],
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
    },
    {
        "name": "exp6_decision_tree_tenure",
        "categorical_features": ["Partner", "Dependents", "PaperlessBilling"],
        "numeric_features": ["TotalCharges", "MonthlyCharges", "tenure"],
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
    },
    {
        "name": "exp7_decision_tree_contract",
        "categorical_features": ["Partner", "Dependents", "PaperlessBilling", "Contract"],
        "numeric_features": ["TotalCharges", "MonthlyCharges"],
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
    },
    {
        "name": "exp8_decision_tree_internetservice",
        "categorical_features": ["Partner", "Dependents", "PaperlessBilling", "InternetService"],
        "numeric_features": ["TotalCharges", "MonthlyCharges"],
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
    },
    {
        "name": "exp9_decision_tree_contract_replace",
        "categorical_features": ["Partner", "Dependents", "Contract"],
        "numeric_features": ["TotalCharges", "MonthlyCharges"],
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
    },
    {
        "name": "exp10_random_forest",
        "categorical_features": ["Partner", "Dependents", "PaperlessBilling"],
        "numeric_features": ["TotalCharges", "MonthlyCharges"],
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
    },
]


def train_and_log(config: dict, df, train_idx, test_idx) -> Pipeline:
    logger.info("Iniciando experimento: %s", config["name"])
    features = config["categorical_features"] + config["numeric_features"]
    X_train, y_train = get_xy(df, train_idx, features)
    X_test, y_test = get_xy(df, test_idx, features)

    preprocessor = build_preprocessor(config["categorical_features"], config["numeric_features"])
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", config["estimator"]),
    ])

    with mlflow.start_run(run_name=config["name"]):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        print_metrics(config["name"], metrics)
        print_confusion_matrix(y_test, y_pred)

        logger.info(
    "Experimento %s - Precision: %.4f - Recall: %.4f - F1: %.4f - ROC-AUC: %.4f",
    config["name"],
    metrics["precision"],
    metrics["recall"],
    metrics["f1"],
    metrics["roc_auc"],
)

        mlflow.log_param("categorical_features", config["categorical_features"])
        mlflow.log_param("numeric_features", config["numeric_features"])
        mlflow.log_params(config["estimator"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="model",skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"])

        logger.info("Experimento finalizado: %s", config["name"])
    return pipeline
def get_config_by_name(name: str) -> dict:
    for config in MODEL_CONFIGS:
        if config["name"] == name:
            return config
    disponibles = ", ".join(c["name"] for c in MODEL_CONFIGS)
    raise ValueError(f"No existe el experimento '{name}'. Disponibles: {disponibles}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Nombre de un experimento puntual (ver MODEL_CONFIGS). Si no se pasa, corre los 10.",
    )
    args = parser.parse_args()

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_data(DATA_PATH)
    train_idx, test_idx = get_train_test_indices(df)

    if args.experiment:
        config = get_config_by_name(args.experiment)
        train_and_log(config, df, train_idx, test_idx)
    else:
        for config in MODEL_CONFIGS:
            train_and_log(config, df, train_idx, test_idx)


if __name__ == "__main__":
    main()