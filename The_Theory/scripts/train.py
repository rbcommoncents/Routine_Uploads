import numpy as np 
from utils.data_loader import load_dataset, split_dataset
from utils.data_preprocessor import preprocess_data
from models.neural_network import NeuralNetwork

def main():
    data = load_dataset('/home/host/Desktop/Git/Directory_1/Routine/The_Theory/data/base_dataset.csv')
    if data is not None:
        features, labels = preprocess_data(data)
        train_data, test_data = split_dataset(data)

        X_train, y_train = preprocess_data(train_data)
        X_test, y_test = preprocess_data(test_data)

        nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=5, output_size=1)
        nn.train(X_train.to_numpy(), y_train.to_numpy(), epochs=1000, learning_rate=0.01)

        predictions = nn.predict(X_test.to_numpy())
        accuracy = np.mean((predictions > 0.5) == y_test.to_numpy())
        print(f"Model accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()