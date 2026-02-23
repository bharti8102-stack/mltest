import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
 
from sklearn.ensemble import (
     RandomForestClassifier,
     GradientBoostingClassifier,
     AdaBoostRegressor,
 )
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object,evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model_trainer", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={ 
                    "Random Forest":RandomForestClassifier(),
                    "Gradient Boosting":GradientBoostingClassifier(),
                    "AdaBoost":AdaBoostRegressor(),
                    "Linear Regression":LinearRegression(),
                    "Decision Tree":DecisionTreeRegressor(),
                    "XGBRegressor":XGBRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {
                   'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting": {
                    'Learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost": {
                    'Learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,parameters=params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            
            print("This is the best model:",best_model_name)
            model_names = list(params.keys())
            
            actual_model=""
            
            for model in model_names:
                if model in best_model_name:
                    actual_model = actual_model + model
                    break
                best_params = params[actual_model]
                mlflow.set_registry_uri("https://dagshub.com/bharti8102/mltest.mlflow")
                tracking_uri = urlparse(mlflow.get_tracking_uri().scheme)
                
                #mlflow
                 
            with mlflow.start_run():
                
                predicted = best_model.predict(X_test)
                rmse, mae, r2 = evaluate_model(y_test, predicted)
                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
            
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
            else:
                mlflow.sklearn.log_model(best_model, "model")
                
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )
            
            predicted=best_model.predict(X_test)
            r2_score= r2_score(y_test.predicted)
            return r2_score
        
   
            
        except Exception as e:
            raise CustomException(e,sys)
        
 
 