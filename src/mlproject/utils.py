import os
import sys
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

import pickle
import numpy as np

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
    

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)

        print(">>> save_object called")
        print("Directory:", dir_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(object, file_obj)

        print(">>> Pickle saved successfully")

    except Exception as e:
        print(">>> Pickle save FAILED:", e)
        raise e

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]

            # Hyperparameter tuning
            gs = GridSearchCV(model, param, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Model prediction
            y_pred = gs.predict(X_test)

            # Calculate R2 score
            r2 = r2_score(y_test, y_pred)
            report[list(models.keys())[i]] = r2

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
    def Load_object(file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)
        except Exception as e:
            raise CustomException(e,sys)
        
            