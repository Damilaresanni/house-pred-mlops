import pandas as pd
import mlflow
import joblib
import yaml
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from src.data.preprocess import preprocessor


logging.basicConfig(
    level= logging.INFO,
    format="%(asctime)s- %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "/home/fidisroxy/development/mlops/house-pred-mlops/configs/model_config.yaml"
logger.info("Loading Model config")
def train(config_path="/home/fidisroxy/development/mlops/house-pred-mlops/configs/model_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
        params = {
            **cfg["model"],
            **cfg["training"]
        }
        
        df = pd.read_csv(cfg["data"]["processed_path"])
        df = df.dropna(subset=["amount"])
        X = df.drop("amount", axis=1)
        y = df["amount"]
        print(df.head(5))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg["data"]["test_size"],
                                                            random_state=cfg["training"]["random_seed"])
        
        mlflow.set_experiment("house-price-prediction")
        
        logger.info("Creating model Pipeline")
        with mlflow.start_run():
            mlflow.log_params({
                 "model_name": cfg["model"]["name"],
                "n_estimators": cfg["model"]["n_estimators"],
                "max_depth": cfg["model"]["max_depth"],
                "test_size": cfg["data"]["test_size"],
                "random_seed": cfg["training"]["random_seed"]
            })
            
            # mlflow.log_artifact(config_path)
            model = RandomForestRegressor(
                n_estimators=cfg["model"]["n_estimators"],
                max_depth=cfg["model"]["max_depth"]
            )
            model_pipeline = Pipeline(
                steps=[
                 ("preprocessor", preprocessor),
                 ("model", model)]
            ) 
            
            
            
            logger.info("Training Info")
            print("X_train columns:", X_train.columns.tolist())
            model_pipeline.fit(X_train, y_train)

            predictions = model_pipeline.predict(X_test)
            
            
            rmse = root_mean_squared_error(
                y_test,
                predictions
               
            )
            
            
        
            mlflow.log_metric("rmse", rmse)
            
            joblib.dump(model_pipeline, "/home/fidisroxy/development/mlops/house-pred-mlops/models/model.pkl")
            
            mlflow.log_artifact("/home/fidisroxy/development/mlops/house-pred-mlops/models/model.pkl")
            
            logger.info("Saving model...")
            print(f"RMSE: {rmse}")
            return rmse


if __name__ == "__main__":
    train()