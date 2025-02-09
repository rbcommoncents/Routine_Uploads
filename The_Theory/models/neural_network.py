import numpy as np 
from utils.activation_function import sigmoid, sigmoid_derivative

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output, learning_rate):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(self.final_input)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T )
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_input)

        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta)*learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True)*learning_rate

        self.weights_input_hidden += np.dot(X.T, hidden_delta)*learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)*learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y.reshape(-1, 1), output, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((y.reshape(-1, 1) - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, X):
        return self.forward(X)