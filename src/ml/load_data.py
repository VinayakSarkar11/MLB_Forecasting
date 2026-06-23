import pandas as pd

from src.ml.snowflake_client import query_to_df


def load_model1_features() -> pd.DataFrame:
    df = query_to_df("SELECT * FROM vw_model1_features")
    df.columns = [c.lower() for c in df.columns]
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_model2_features() -> pd.DataFrame:
    df = query_to_df("SELECT * FROM vw_model2_pa_features")
    df.columns = [c.lower() for c in df.columns]
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_model2_midab_features() -> pd.DataFrame:
    df = query_to_df("SELECT * FROM vw_model2_midab_features")
    df.columns = [c.lower() for c in df.columns]
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df
