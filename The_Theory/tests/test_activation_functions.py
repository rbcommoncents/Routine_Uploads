import unittest
import pandas as pd 
from utils.data_loader impor load_dataset

class TestDataLoader(unittest.TestCase):
    def test_load_base_dataset(self):
        data = load_dataset('data/base_dataset.csv')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (4, 4))
        self.assertListEqual(list(data.columns), ['feature1', 'feature2', 'feature3', 'label'])

    def test_load_working_dataset(self):
        data = load_dataset('data/working_dataset.csv')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)

if __name__ == '__main__':
    unittest.main()