from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    """
    Réplica del preprocessing usado en el notebook:
    - Categóricas: imputación por moda + OneHotEncoder (handle_unknown="ignore")
    - Numéricas: imputación por mediana + StandardScaler (necesario para modelos
      lineales; no altera a los modelos de árboles)
    Ambas listas son parametrizables porque cada experimento usa un subconjunto
    distinto de columnas.
    """
    transformers = []

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    return ColumnTransformer(transformers=transformers)