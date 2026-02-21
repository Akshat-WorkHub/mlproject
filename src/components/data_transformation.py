import os,sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath: str = os.path.join('artifact',"preprocessor_obj.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()

    def get_preprocessor_obj(self,data):

        logging.info("Getting Pre Processor Object !")
        try:
            numerical_columns = [col for col in data.columns if data[col].dtype != 'O']
            categorical_columns = [col for col in data.columns if data[col].dtype == 'O']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                    ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Numerical Columns : {numerical_columns}")            
            logging.info(f"Categorical Columns : {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns),
                ]
            )

            logging.info("Pre processor object created !!")
            return preprocessor

        except Exception as e:
            try:
                raise CustomException(e,sys)
            except CustomException as ce:
                print(ce)
    
    def initiate_data_transformation(self,train_path: str,test_path: str):
        try:
            logging.info("Data transformation Started by instanciated its data transformation method !")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            target_column_name = "math_score"

            train_input_features_df = train_df.drop(columns=[target_column_name],axis=1)
            train_target_features_df = train_df[target_column_name]

            test_input_features_df = test_df.drop(columns=[target_column_name],axis=1)
            test_target_features_df = test_df[target_column_name]

            preprocessor_obj = self.get_preprocessor_obj(train_input_features_df)

            logging.info("Applying Preprocessor Object to input features of train and test data to create their arrays")
            
            train_input_features_arr = preprocessor_obj.fit_transform(train_input_features_df)
            test_input_features_arr = preprocessor_obj.transform(test_input_features_df)

            train_arr = np.c_[train_input_features_arr,np.array(train_target_features_df)]
            test_arr = np.c_[test_input_features_arr,np.array(test_target_features_df)]

            logging.info("Saving Preprocessor as a pickle object using save_object() method defined in utils")
            save_object(
                filepath=self.data_transformer_config.preprocessor_obj_filepath,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_filepath
            )
            

            
        except Exception as e:
            try:
                raise CustomException(e,sys)
            except CustomException as ce:
                print(ce)
    