import pandas as pd
import os, sys
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact',"train.csv")
    test_data_path: str = os.path.join('artifact',"test.csv")
    raw_data_path: str = os.path.join('artifact',"raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        logging.info("Entered into DataIngestion Class and initiated ingestion method")

        try:
            logging.info("Trying to read a Red Wine Dataset Locally")
            filepath = r"data\red_wine_cleaned.csv"
            df = pd.read_csv(filepath)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            logging.info("Creating Train & Test Data")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)


            logging.info("Storing files in csv format to Artifact Repo")
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Finally Data Ingestion Completed")

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_ingestion()
            
    