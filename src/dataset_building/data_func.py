import pandas as pd

def read_clean_dataset(dataset_path: str, features: list) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    clean_df = df[features]
    return clean_df
