import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts/preprocessor.pkl")

    def get_data_transformer(self,df):
        try:
            cat_cols = df.select_dtypes(include=["object"]).columns
            num_cols = df.select_dtypes(exclude = ["object"]).drop("burnout_score", axis=1).columns

            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])
            
            cat_pipeline = Pipeline({
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            })
            
            preprocessor = ColumnTransformer([
            # transformers = [
            #     ("cat", OneHotEncoder(handle_unknown="ignore"),cat_cols)
            #     ("num","passthrough",num_cols),
            # ]

            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
            ])
            
            return preprocessor
            
        except Exception as e: 
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")

            target = "burnout_score"

            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]

            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]
            
            logging.info("Obtaining preprocessing object ")

            preprocessor = self.get_preprocessor(train_df)

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            save_object(self.preprocessor_path, preprocessor)
            
            return(
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
            )


        except Exception as e:
            raise CustomException(e,sys)