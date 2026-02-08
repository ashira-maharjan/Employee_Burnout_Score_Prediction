import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    logging.info("Starting Burnout Prediction Model Training Pipeline")
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(
            data_path="notebook/data/work_from_home_burnout_dataset.csv"
        )
        logging.info("Data ingestion completed successfully")

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )
        logging.info("Data transformation completed successfully")

        model_trainer = ModelTrainer()
        r2_value, best_model_name = model_trainer.initiate_model_trainer(
            train_array=train_arr,
            test_array=test_arr
        )
        logging.info(f"Model training completed successfully with R2 score: {r2_value:.4f}")
        logging.info(f"Best model name: {best_model_name}")

        print("\n" + "="*70)
        print("MODEL TRAINING SUCCESSFUL")
        print("="*70)
        print(f" Best Model   : {best_model_name}")
        print(f"R² Score     : {r2_value:.4f}")
        print("="*70)
        print("\n Artifacts saved in 'artifacts/' directory:")
        print("   - Model: artifacts/model.pkl")
        print("   - Preprocessor: artifacts/preprocessor.pkl")
        print("="*70)

    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise CustomException(e, sys)
