from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainerConfig, ModelTrainer



import sys


if __name__ == "__main__":
    try:
        logging.info("🚀 Application started")
  # Data-Ingestion
        data_ingestion = DataIngestion()

        train_path, test_path = data_ingestion.initiate_data_ingestion()
    # Data-Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_path, test_path)
         
    #Model-Trainer
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Model training completed with R2 score: {r2_score}")
        
        logging.info(f"✅ Data ingestion successful")
        logging.info(f"Train data path: {train_path}")
        logging.info(f"Test data path: {test_path}")

        print("Data ingestion completed successfully")

    except CustomException as e:
        logging.error(str(e))
        print(str(e))

    except Exception as e:
        logging.error(str(e))
        raise CustomException(e, sys)

from src.mlproject.utils import save_object

save_object("artifacts/preprocessor.pkl", {"ok": True})
save_object("artifacts/model.pkl", {"ok": True})