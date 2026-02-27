import os
import sys
import pandas as pd
import pickle
import numpy as np

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging


# Load environment variables
load_dotenv(find_dotenv())

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


# -------------------------------
# READ DATA FROM SQL
# -------------------------------
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


# -------------------------------
# SAVE PICKLE OBJECT
# -------------------------------
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
        raise CustomException(e, sys)


# -------------------------------
# LOAD PICKLE OBJECT
# -------------------------------
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


# -------------------------------
# EVALUATE MODELS
# -------------------------------
def evaluate_models(X_train, y_train, X_test, y_test, models, parameters):
    try:
        report = {}

        for model_name, model in models.items():

            param = parameters.get(model_name, {})

            # If hyperparameters exist → use GridSearch
            if param:
                gs = GridSearchCV(model, param, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            report[model_name] = r2

        return report

    except Exception as e:
        raise CustomException(e, sys)