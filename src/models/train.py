import pandas as pd
import mlflow
import joblib
import yaml
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, accuracy_score, mean_absolute_error
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


def evaluate(y_test,predictions):
    rmse = root_mean_squared_error(y_test,predictions)
    mae = mean_absolute_error(y_test,predictions)
    r2 = r2_score(y_test,predictions)
    
    
    return {
        "rmse": rmse,
        "mae" : mae,
        "r2" : r2,
    }

def train(config_path="/home/fidisroxy/development/mlops/house-pred-mlops/configs/model_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
        params = {
            **cfg["models"]
        }
        
        df = pd.read_csv(cfg["data"]["processed_path"])
        df = df.dropna(subset=["amount"])
        X = df.drop("amount", axis=1)
        y = df["amount"]
        print(df.head(5))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg["data"]["test_size"],
                                                            random_state=cfg["training"]["random_seed"])
        
        mlflow.set_experiment("house-price-prediction")
        
        
    def get_model(model_name, params):
             
            if model_name == "linear_regression":
                model = LinearRegression()
                return model
                
            if model_name == "random_forest_regressor":
                model = RandomForestRegressor(**params)
                return model
            
            if model_name == "decision_tree_regressor":
                model = DecisionTreeRegressor(**params)
                return model
            
            else:
                return(f"model Type Not Supported: {model_name}")
        
        
    for model_name, params in cfg["models"].items():
        model = get_model(model_name, params)
        print(f"Model: {model_name}")
        print(f"The Model: {model}")
        print(f"Parameters: {params}")
        print("-" * 30)
        logger.info("Creating model Pipeline")
        with mlflow.start_run(run_name="The Trinity"):
            mlflow.log_params({
                 **params
            })
            
            # mlflow.log_artifact(config_path)
            
            
            
        model_pipeline = Pipeline(
                steps=[
                 ("preprocessor", preprocessor),
                 ("model", model)]
            ) 
            
            
            
        logger.info("Training Info")
        print("X_train columns:", X_train.columns.tolist())
        model_pipeline.fit(X_train, y_train)

        predictions = model_pipeline.predict(X_test)
            
            
        results = evaluate(y_test, predictions)
            
            
        mlflow.log_param("model", model_name)
        mlflow.log_metrics(results)
            
        logger.info("Saving model...")
        joblib.dump(model_pipeline, "/home/fidisroxy/development/mlops/house-pred-mlops/models/model.pkl")
            
        mlflow.log_artifact("/home/fidisroxy/development/mlops/house-pred-mlops/models/model.pkl")
            
        logger.info("Model Saved...")
        print(f"RMSE: {results}")
        return results


if __name__ == "__main__":
    train()