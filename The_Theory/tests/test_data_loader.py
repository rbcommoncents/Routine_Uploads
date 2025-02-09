from dotenv import load_dotenv
import os
import unittest
import pandas as pd
from utils.data_loader import load_dataset

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

BASE_DATA = os.getenv("FILE_PATH_BASE")
PREDICT_DATA = os.getenv("FILE_PATH_PREDICTIVE")

class TestDataLoader(unittest.TestCase):
    def test_load_first_dataset(self):
        data = load_dataset(BASE_DATA)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape[1], 4)
        self.assertListEqual(list(data.columns), ['temperature', 'humidity', 'pressure', 'label'])

    def test_load_second_dataset(self):
        data = load_dataset(PREDICT_DATA)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape[1], 3)

if __name__ == '__main__':
    unitest.main()