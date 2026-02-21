import os,sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath: str = os.path.join("artifacts","preprocessor_obj.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_obj(self,data):
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
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            try:
                CustomException(e,sys)
            except CustomException as ce:
                print(ce)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column_name = "math_score"

            train_input_features_df = train_df.drop(columns=[target_column_name],axis=1)
            train_target_features_df = train_df[target_column_name]

            test_input_features_df = test_df.drop(columns=[target_column_name],axis=1)
            test_target_features_df = test_df[target_column_name]

            preprocessor_obj = self.get_preprocessor_obj(train_input_features_df)

            train_input_features_arr = preprocessor_obj.fit_transform(train_input_features_df)
            test_input_features_arr = preprocessor_obj.transform(test_input_features_df)
            
            train_arr = np.c_(train_input_features_arr,np.array(train_target_features_df))
            test_arr = np.c_(test_input_features_arr,np.array(test_target_features_df))

            save_object(
                filepath=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath
            )

        except Exception as e:
            try:
                CustomException(e,sys)
            except CustomException as ce:
                print(ce)