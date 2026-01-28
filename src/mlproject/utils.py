import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

load_dotenv(find_dotenv())

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    try:
        logging.info("Reading SQL Database started")

        engine = create_engine(
            f"mysql+pymysql://{user}:{password}@{host}/{db}"
        )

        query = "SELECT * FROM student"
        df = pd.read_sql(query, engine)

        logging.info(f"Data fetched successfully | Rows={len(df)}")
        return df

    except Exception as ex:
        raise CustomException(ex, sys)
