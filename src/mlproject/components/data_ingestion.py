import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import read_sql_data


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("🚀 Data ingestion started")

            # 1️⃣ Read data from MySQL
            df = read_sql_data()
            logging.info("Data read successfully from database")

            # 2️⃣ Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # 3️⃣ Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved successfully")

            # 4️⃣ Train-test split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            # 5️⃣ Save train & test data
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train and test data saved successfully")
            logging.info("✅ Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except CustomException:
            raise
        except Exception as ex:
            raise CustomException(ex, sys)
