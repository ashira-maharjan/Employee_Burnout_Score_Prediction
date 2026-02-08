import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import sklearn 
from packaging import version


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_features, categorical_features):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # choose correct OneHotEncoder parameter depending on scikit-learn version 
            if version.parse(sklearn.__version__) >= version.parse("1.2"):
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
                   
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", ohe)
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_features),
                    ("cat", cat_pipeline, categorical_features)
                ],
                remainder="drop"
            )

            logging.info(f"Created preprocessor with {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        drop_cols = ["user_id", "burnout_risk"]
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                logging.info(f"Dropped column: {col}")

        if "burnout_score" in df.columns:
            original_min = df["burnout_score"].min()
            original_max = df["burnout_score"].max()
            df["burnout_score"] = df["burnout_score"].clip(0, 100)
            logging.info(f"Clipped burnout_score from [{original_min:.2f}, {original_max:.2f}] to [0, 100]")

        if "task_completion_rate" in df.columns:
            original_min = df["task_completion_rate"].min()
            original_max = df["task_completion_rate"].max()
            df["task_completion_rate"] = df["task_completion_rate"].clip(0, 100)
            logging.info(f"Clipped task_completion_rate from [{original_min:.2f}, {original_max:.2f}] to [0, 100]")

        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train and test data")

            train_df = self._basic_preprocessing(train_df)
            test_df = self._basic_preprocessing(test_df)

            target_column = "burnout_score"
            if target_column not in train_df.columns:
                raise CustomException(f"Target column '{target_column}' not found in training data", sys)

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Ensure columns match
            if set(X_train.columns) != set(X_test.columns):
                raise CustomException("Train and test feature columns mismatch after preprocessing", sys)

            # Determine feature types
            numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")

            preprocessor = self.get_data_transformer_object(numerical_features, categorical_features)

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor)
            logging.info(f"Saved preprocessor at {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
