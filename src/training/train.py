import argparse
import yaml
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
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/customer_churn_historical.csv"
EXPERIMENT_NAME = "customer-churn"
REGISTERED_MODEL_NAME = "customer-churn-model"


def get_data_version(data_path: str = DATA_PATH) -> str:
    """Hash md5 del dataset según su archivo .dvc (identifica la versión de datos usada)."""
    with open(f"{data_path}.dvc") as f:
        return yaml.safe_load(f)["outs"][0]["md5"]

# Todas las variables disponibles (customerID se descarta en load_raw_data)
ALL_CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
ALL_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

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
    # A partir de acá: todas las variables. class_weight="balanced" compensa el
    # desbalance (26% churn) para no sacrificar recall: un falso negativo es un
    # cliente que se va sin que se intente retenerlo.
    {
        "name": "exp11_logistic_regression_all",
        "categorical_features": ALL_CATEGORICAL,
        "numeric_features": ALL_NUMERIC,
        "estimator": LogisticRegression(max_iter=1000),
    },
    {
        "name": "exp12_logistic_regression_all_balanced",
        "categorical_features": ALL_CATEGORICAL,
        "numeric_features": ALL_NUMERIC,
        "estimator": LogisticRegression(max_iter=1000, class_weight="balanced"),
    },
    {
        "name": "exp13_decision_tree_all_balanced",
        "categorical_features": ALL_CATEGORICAL,
        "numeric_features": ALL_NUMERIC,
        "estimator": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_STATE
        ),
    },
    {
        "name": "exp14_random_forest_all_balanced",
        "categorical_features": ALL_CATEGORICAL,
        "numeric_features": ALL_NUMERIC,
        "estimator": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    },
    {
        "name": "exp15_random_forest_tuned",
        "categorical_features": ALL_CATEGORICAL,
        "numeric_features": ALL_NUMERIC,
        "estimator": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            min_samples_split=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
        ),
    },
]


def train_and_log(config: dict, df, train_idx, test_idx, register: bool = False) -> Pipeline:
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
        mlflow.set_tags({
            "data_path": DATA_PATH,
            "data_md5": get_data_version(),
        })
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        classifier = pipeline.named_steps["classifier"]
        positive_class_index = list(classifier.classes_).index("Yes")
        y_prob = pipeline.predict_proba(X_test)[:, positive_class_index]
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
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            # Solo el modelo candidato elegido se registra en el Model Registry
            registered_model_name=REGISTERED_MODEL_NAME if register else None,
        )

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
        help="Nombre de un experimento puntual (ver MODEL_CONFIGS). Si no se pasa, corre todos.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help=f"Registra el modelo en el Model Registry como '{REGISTERED_MODEL_NAME}'. Requiere --experiment.",
    )
    args = parser.parse_args()
    if args.register and not args.experiment:
        parser.error("--register requiere indicar --experiment (solo se registra el modelo candidato)")

    # Credenciales y tracking URI de DagsHub se leen del archivo .env (ver .env.example)
    load_dotenv()
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_data(DATA_PATH)
    train_idx, test_idx = get_train_test_indices(df)

    if args.experiment:
        config = get_config_by_name(args.experiment)
        train_and_log(config, df, train_idx, test_idx, register=args.register)
    else:
        for config in MODEL_CONFIGS:
            train_and_log(config, df, train_idx, test_idx)


if __name__ == "__main__":
    main()