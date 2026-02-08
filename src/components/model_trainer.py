import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(objective="reg:squarederror", verbosity=0),
                "CatBoosting Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05], 
                    'subsample': [0.7, 0.8, 0.9], 
                    'n_estimators': [16, 32, 64]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01], 
                    'n_estimators': [16, 32]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8], 
                    'learning_rate': [0.01, 0.05], 
                    'iterations': [50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01], 
                    'n_estimators': [16, 32]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7], 
                    'weights': ['uniform', 'distance']
                }
            }

            best_model = None
            best_score = -float("inf")
            model_report = {}

            for name, model in models.items():
                logging.info(f"Evaluating model: {name}")
                param_grid = params.get(name, {})
                if param_grid:
                    gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=1)
                    gs.fit(X_train, y_train)
                    fitted_model = gs.best_estimator_
                else:
                    # No params to tune; fit directly
                    fitted_model = model.fit(X_train, y_train)

                preds = fitted_model.predict(X_test)
                score = r2_score(y_test, preds)
                model_report[name] = score
                logging.info(f"{name} R2: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = fitted_model
                    best_model_name = name

            if best_model is None or best_score < 0.6:
                raise CustomException("No model met the performance threshold (R2 >= 0.6)", sys)

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info(f"Best model ({best_model_name}) saved to {self.model_trainer_config.trained_model_file_path}")

            return best_score, best_model_name

        except Exception as e:
            raise CustomException(e, sys)
