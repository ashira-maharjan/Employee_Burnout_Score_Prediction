import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    # def __init__(self):
    # self.trained_model_file_path = os.path.join("artifacts", "model.pkl")

    trained_model_file_path: str = os.path.join("artifacts","model.pkl")

    class ModelTrainer:
        def __init__(self):
            self.model_trainer_config = ModelTrainerConfig()

        def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            #models
             models = {
                "Linear Regression": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(objective="reg:squarederror"),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
            }

            params = {
               
               "Linear Regression": {},
                
                "KNN Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                },


                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },

                "Random Forest": {
                    'n_estimators': [64, 128, 256]
                },

                "Gradient Boosting": {
                    'learning_rate': [.1, .05, .01],
                    'subsample': [0.75, 0.8, 0.9],
                    'n_estimators': [64, 128, 256]
                },

                "AdaBoost Regressor": {
                    'learning_rate': [.1, .05, .01],
                    'n_estimators': [64, 128, 256]
                },
                

                "XGBRegressor": {
                    'learning_rate': [.1, .05, .01],
                    'n_estimators': [64, 128, 256]
                },

                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 200]
                },

                
            }

            logging.info("Evaluating Regression Models")

            # model_report = evaluate_models(


            # model_report is expected to be a dict

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            #extract best model score 
            best_model_score = max(sorted(model_report.values()))

            # best model name from dict
            # best_model_name = max(model_report, key=model_report.get)

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.60:
               raise CustomException("No suitable  best model found", sys)
            logging.info(f"Best model selected: {best_model_name} with R2 Score: {best_model_score}") 

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            #Final evaluation
            predictions = best_model.predict(X_test)
            r2 = r2_score(y_test, predictions)

            logging.info(f"Final R2 Score on test data: {r2_square}")
            
            print("\n================= Model Training Completed =================")
            print(f"Best Model: {best_model_name}")
            print(f"Test R2 Score: {r2_square}")
            print("===========================================================\n")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)




            










