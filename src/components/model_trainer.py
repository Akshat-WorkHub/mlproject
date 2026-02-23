import os,sys
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from dataclasses import dataclass

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

@dataclass
class ModelTrainerConfig:
    trained_model_filepath: str = os.path.join('artifact',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()

    def initiate_model_train(self,train_arr,test_arr):
        try:
            logging.info("Splitting Train Array and Test Array into (X_train,y_train) , (X_test,y_test)")

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(verbose=False),
                "Decision Tree" : DecisionTreeRegressor(),
                "K-Neighbours Regressor" : KNeighborsRegressor(),
                "Linear Regression" : LinearRegression(),
                "XGBRegressor" : XGBRegressor(),
            }

            logging.info("Hyperparmeter tuning is done ")
            params={
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Linear Regression":{},
                "K-Neighbours Regressor":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
            }

            logging.info(f"Models Used in Model Training : {list(models.keys())}")

            model_report, trained_models = evaluate_models(X_train,y_train,X_test,y_test,models,params)

            best_model_name = max(model_report,key=lambda x : model_report[x]["Testing Score"])
            best_model_score = model_report[best_model_name]["Testing Score"]

            best_model = trained_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Training Model gives testing score accuracy more than 70% ")
            
            save_object(
                filepath=self.model_train_config.trained_model_filepath,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_val = r2_score(y_test,predicted)
            
            return r2_val

        except Exception as e:
            try:
                raise CustomException(e,sys)
            except CustomException as ce:
                print(ce)

