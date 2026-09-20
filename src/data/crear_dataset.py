import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_raw_data(path: str) -> pd.DataFrame:
    """
    - descarta customerID (no debe usarse como predictor)
    - fuerza MonthlyCharges y TotalCharges a numérico
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"], errors="ignore")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def get_train_test_indices(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """
    Genera UNA única partición train/test (por índice)
    """
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET],
    )
    return train_idx, test_idx


def get_xy(df: pd.DataFrame, idx, features: list[str]):
    X = df.loc[idx, features]
    y = df.loc[idx, TARGET]
    return X, y