import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.exception import CustomException 
from src.logger import logging
import sys


class DataIngestion:
    def __init__(self):
        self.train_data_path = "artifacts/train.csv"
        self.test_data_path = "artifacts/test.csv"
        self.raw_data_path = "artifacts/raw.csv"

    def initiate_data_ingestion(self):
            
            try: 
                 logging.info("Reading dataset")
                 df = pd.read_csv("notebook\data\work_from_home_burnout_dataset.csv")

                 # drop burn_risk 
                 if "burn_risk" in df.columns:
                      df.drop(columns=["burn_risk"], inplace=True)
                 
                 os.makedirs("artifacts",exist_ok=True)
                 df.to_csv(self.raw_data_path, index=False)
                 
                 train_set, test_set = train_test_split(df, test_size=0.2, random_state = 42)

                 train_set.to_csv(self.train_data_path, index=False)
                 test_set.to_csv(self.test_data_path, index =False)
                 
                 logging.info("Data Ingestion completed")
                 return self.train_data_path, self.test_data_path
            
            except Exception as e:
                 raise CustomException(e,sys)

 