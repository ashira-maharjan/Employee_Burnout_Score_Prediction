import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.exception import CustomException 
from src.logger import logging
import sys
from dataclasses import dataclass 

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
          self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self,data_path: str = "notebook/data/work_from_home_burnout_dataset.csv"):
        logging.info("Reading dataset")
        
        try: 
            # FIXED: Use forward slashes or raw string for Windows paths
            df = pd.read_csv(r"notebook\data\work_from_home_burnout_dataset.csv")
            # Alternative: df = pd.read_csv("notebook/data/work_from_home_burnout_dataset.csv")

            # drop burnout_risk column (fixed column name)
            if "burnout_risk" in df.columns:
                 df.drop(columns=["burnout_risk"], inplace=True)
            logging.info("Dropped 'burnout_risk' column")

            # Clip burnout_score to valid range [0, 100]
            if "burnout_score" in df.columns:
                original_min = df["burnout_score"].min()
                original_max = df["burnout_score"].max()
                df["burnout_score"] = df["burnout_score"].clip(0, 100)
                logging.info(f"Clipped burnout_score from [{original_min:.2f}, {original_max:.2f}] to [0, 100]")
            
            # creating folders
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            # train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # save train and test set
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed")
            logging.info(f"Train data shape: {train_set.shape}")
            logging.info(f"Test data shape: {test_set.shape}")

            return (
                  self.ingestion_config.train_data_path,
                  self.ingestion_config.test_data_path
            )
        
        except Exception as e:
             raise CustomException(e, sys)

# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data, test_data = obj.initiate_data_ingestion()

#     data_transformation = DataTransformation()
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

#     modeltrainer = ModelTrainer()
#     result = modeltrainer.initiate_model_trainer(train_arr, test_arr)
#     print(f"Best Model: {result[1]} | R² Score: {result[0]:.4f}")


# Standalone execution for testing
# if __name__ == "__main__":
#     try:
#         obj = DataIngestion()
#         train_data, test_data = obj.initiate_data_ingestion()
#         print(f"Data ingestion successful!")
#         print(f"Train data path: {train_data}")
#         print(f"Test data path: {test_data}")
#     except Exception as e:
#         print(f"Error during data ingestion: {e}")