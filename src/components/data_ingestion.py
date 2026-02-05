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