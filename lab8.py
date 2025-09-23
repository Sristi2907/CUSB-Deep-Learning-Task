import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class Autoencoder:
    def __init__(self, input_size, hidden_size):
        # Initialize weights with Xavier/Glorot initialization
        self.weights_encode = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.bias_encode = np.zeros((1, hidden_size))
        self.weights_decode = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / hidden_size)
        self.bias_decode = np.zeros((1, input_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Encode
        self.hidden = self.sigmoid(np.dot(X, self.weights_encode) + self.bias_encode)
        # Decode
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_decode) + self.bias_decode)
        return self.output
    
    def backward(self, X, learning_rate):
        # Calculate gradients
        error = X - self.output
        d_output = error * self.sigmoid_derivative(self.output)
        
        # Update decoder parameters (fixed gradient descent)
        self.weights_decode -= learning_rate * np.dot(self.hidden.T, d_output)
        self.bias_decode -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
        
        # Update encoder parameters (fixed gradient descent)
        d_hidden = np.dot(d_output, self.weights_decode.T) * self.sigmoid_derivative(self.hidden)
        self.weights_encode -= learning_rate * np.dot(X.T, d_hidden)
        self.bias_encode -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)
    
    def train(self, X, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, learning_rate)
            loss = np.mean(np.square(X - output))
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Autoencoder Epoch {epoch}, Loss: {loss:.6f}")
        return losses

class DeepNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
    def initialize_with_autoencoder(self, X, hidden_sizes):
        current_input = X
        # Train autoencoders layer by layer
        for i, hidden_size in enumerate(hidden_sizes):
            print(f"\nTraining autoencoder for layer {i+1}")
            autoencoder = Autoencoder(current_input.shape[1], hidden_size)
            autoencoder.train(current_input, epochs=500, learning_rate=0.1)
            
            # Store the encoder weights and biases
            self.weights.append(autoencoder.weights_encode)
            self.biases.append(autoencoder.bias_encode)
            
            # Get encoded representation for next layer
            current_input = autoencoder.sigmoid(
                np.dot(current_input, autoencoder.weights_encode) + autoencoder.bias_encode
            )
        
        # Add final output layer weights
        self.weights.append(np.random.randn(hidden_sizes[-1], 3) * np.sqrt(2.0 / hidden_sizes[-1]))
        self.biases.append(np.zeros((1, 3)))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def softmax(self, x):
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shift)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights) - 1):
            X = self.sigmoid(np.dot(X, self.weights[i]) + self.biases[i])
            self.activations.append(X)
        # Output layer with softmax
        output = self.softmax(np.dot(X, self.weights[-1]) + self.biases[-1])
        self.activations.append(output)
        return output
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        deltas = []
        
        # Output layer error
        delta = self.activations[-1] - y
        deltas.append(delta)
        
        # Hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * (
                self.activations[i] * (1 - self.activations[i])
            )
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.activations[i].T, deltas[i]) / m
            self.biases[i] -= learning_rate * np.mean(deltas[i], axis=0, keepdims=True)
    
    def train(self, X, y, epochs, learning_rate):
        # Convert labels to one-hot encoding
        y_one_hot = np.zeros((y.shape[0], 3))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        
        losses = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y_one_hot, learning_rate)
            
            # Calculate loss
            loss = -np.mean(np.sum(y_one_hot * np.log(output + 1e-8), axis=1))
            losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"DNN Epoch {epoch}, Loss: {loss:.6f}")
        return losses
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Create and train the network
hidden_sizes = [8, 6]  # Two hidden layers
dnn = DeepNeuralNetwork([X_train.shape[1]] + hidden_sizes + [3])

# Initialize weights using autoencoder
print("Pre-training with autoencoders...")
dnn.initialize_with_autoencoder(X_train, hidden_sizes)

# Fine-tune the network
print("\nFine-tuning the network...")
losses = dnn.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Evaluate the model
y_pred = dnn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"\nTest accuracy: {accuracy:.4f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()