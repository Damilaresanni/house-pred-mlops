import pandas as pd
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert not df.empty, "Dataset is empty!"
    assert "price" in df.columns, "Missing price column column"
    return df


def validate_data(df:pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["price"])
    
    df = df.drop_duplicates()
    return df

if __name__ == "__main__":
    df = load_data("/home/fidisroxy/development/mlops/house-pred-mlops/data/raw/")
    df = validate_data(df)
    df.to_csv("data/processed/clean.csv", index=False)