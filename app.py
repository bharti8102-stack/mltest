from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer

import sys
import dagshub
import mlflow

# 🔥 DagsHub Setup (MUST BE BEFORE PIPELINE RUNS)
dagshub.init(repo_owner='bharti8102', repo_name='mltest', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/bharti8102/mltest.mlflow")
mlflow.set_experiment("my-experiment")

print("Tracking URI:", mlflow.get_tracking_uri())


if __name__ == "__main__":
    try:
        logging.info("🚀 Application started")

        # Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        # Model Training
        model_trainer = ModelTrainer()
        r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)

        print(f"Model training completed with R2 score: {r2}")

        logging.info("✅ Pipeline completed successfully")

    except CustomException as e:
        logging.error(str(e))
        print(str(e))

    except Exception as e:
        logging.error(str(e))
        raise CustomException(e, sys)