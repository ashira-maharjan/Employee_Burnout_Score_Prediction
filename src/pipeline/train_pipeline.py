from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    report = trainer.initiate_model_training(X_train, X_test, y_train, y_test)

    print("Model Training Report:", report)
