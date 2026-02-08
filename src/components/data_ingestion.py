import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, data_path: str = "notebook/data/work_from_home_burnout_dataset.csv"):
        logging.info("Starting data ingestion")
        try:
            
            data_path = os.path.normpath(data_path)
            df = pd.read_csv(data_path)

            logging.info(f"Read dataset with shape: {df.shape}")

            # Safely drop non-predictive column 
            if "burnout_risk" in df.columns:
                df.drop(columns=["burnout_risk"], inplace=True)
                logging.info("Dropped 'burnout_risk' column")

            # Clip burnout_score if present
            if "burnout_score" in df.columns:
                original_min = df["burnout_score"].min()
                original_max = df["burnout_score"].max()
                df["burnout_score"] = df["burnout_score"].clip(0, 100)
                logging.info(f"Clipped burnout_score from [{original_min:.2f}, {original_max:.2f}] to [0, 100]")

            # ensure artifacts folder exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # save raw
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Saved raw data to {self.ingestion_config.raw_data_path}")

            # train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed")
            logging.info(f"Train shape: {train_set.shape}, Test shape: {test_set.shape}")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
