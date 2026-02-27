import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models


# ✅ Set DagsHub tracking (IMPORTANT)
import dagshub
import mlflow

dagshub.init(
    repo_owner='bharti8102',
    repo_name='mltest',
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/bharti8102/mltest.mlflow")
mlflow.set_experiment("my-experiment")

print("Tracking URI:", mlflow.get_tracking_uri())
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts", "model_trainer", "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ]
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                parameters=params,
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]

            logging.info(f"Best model found: {best_model_name}")

            # ✅ MLflow Logging
            with mlflow.start_run():
                predicted = best_model.predict(X_test)
                r2 = r2_score(y_test, predicted)

                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("r2_score", r2)

                mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model,
            )

            return r2

        except Exception as e:
            raise CustomException(e, sys)