from dotenv import load_dotenv
import os
import numpy as np 
from utils.data_loader import load_dataset, split_dataset
from utils.data_preprocessor import preprocess_data
from models.neural_network import NeuralNetwork

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

BASE_DATA = os.getenv("FILE_PATH_BASE")
PREDICT_DATA = os.getenv("FILE_PATH_PREDICTIVE")

def main():
    data = load_dataset(BASE_DATA)
    if data is not None:
#        features, labels = preprocess_data(data)
        train_data, val_data, test_data = split_dataset(data)

        X_train, y_train = preprocess_data(train_data)
        X_val, y_val = preprocess_data(val_data)
        X_test, y_test = preprocess_data(test_data)

        nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=5, output_size=1)
        nn.train(X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy(), epochs=3000, learning_rate=0.01)

        data_2 = load_dataset(PREDICT_DATA)
        X_data_2 = preprocess_data(data_2, is_training=False)
        predictions = nn.predict(X_data_2.to_numpy())

        print("New Weather Predictions:")
        for i, pred in enumerate(predictions):
            print(f"Day {i+1}: {'Rain' if pred > 0.5 else 'No Rain'}")

        """predictions = nn.predict(X_test.to_numpy())
        accuracy = np.mean((predictions > 0.5) == y_test.to_numpy())
        print(f"Model accuracy: {accuracy * 100:.2f}%")"""

if __name__ == "__main__":
    main()