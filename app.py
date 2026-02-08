from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformationConfig



import sys


if __name__ == "__main__":
    try:
        logging.info("🚀 Application started")

        data_ingestion = DataIngestion()

        train_path, test_path = data_ingestion.initiate_data_ingestion()
    
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