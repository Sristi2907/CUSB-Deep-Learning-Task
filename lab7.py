import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):  
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):   
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    x_shift = x - np.max(x, axis=1, keepdims=True)   
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        # Xavier Initialization for sigmoid
        self.weights = [np.random.randn(i, j) * np.sqrt(1 / i) 
                        for i, j in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((1, j)) for j in layer_sizes[1:]]

    def feedforward(self, X):
        self.activations = [X]
        self.zs = []
        # Hidden layers (sigmoid)
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.zs.append(z)
            self.activations.append(sigmoid(z))
        # Output layer (softmax)
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.zs.append(z)
        self.activations.append(softmax(z))
        return self.activations[-1]

    def backward(self, y, lr=0.05):
        n_layers = len(self.weights)
        deltas = [None] * n_layers
        # Output error
        deltas[-1] = self.activations[-1] - y
        # Hidden layers
        for i in reversed(range(n_layers - 1)):
            deltas[i] = (deltas[i+1] @ self.weights[i+1].T) * sigmoid_grad(self.zs[i])
        # Update weights & biases
        for i in range(n_layers):
            self.weights[i] -= lr * (self.activations[i].T @ deltas[i]) / y.shape[0]
            self.biases[i] -= lr * np.mean(deltas[i], axis=0, keepdims=True)

    def train(self, X, y, epochs=2000, lr=0.05, return_loss=False):
        losses = []
        for epoch in range(epochs):
            self.feedforward(X)
            self.backward(y, lr)
            if epoch % 200 == 0 or epoch == epochs-1:
                loss = -np.mean(np.sum(y * np.log(self.activations[-1] + 1e-8), axis=1))
                losses.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        if return_loss:
            return losses[-1]

    def predict(self, X):
        probs = self.feedforward(X)
        return np.argmax(probs, axis=1)

# Load and preprocess data
iris = load_iris()
X = iris.data                       
y = iris.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

enc = OneHotEncoder(sparse_output=False)  
y_encoded = enc.fit_transform(y)    

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Architecture exploration parameters
neurons_per_layer = [4, 8, 16, 32]  # x-axis values
num_hidden_layers = [1, 2, 3, 4]    # y-axis values
results = []

# Test different architectures
for n_neurons in neurons_per_layer:
    for n_layers in num_hidden_layers:
        print(f"\nTesting architecture with {n_neurons} neurons in {n_layers} hidden layers")
        
        # Construct layer sizes
        layer_sizes = [X_train.shape[1]]  # Input layer
        layer_sizes.extend([n_neurons] * n_layers)  # Hidden layers
        layer_sizes.append(y_train.shape[1])  # Output layer
        
        # Create and train network
        nn = NeuralNetwork(layer_sizes)
        final_loss = nn.train(X_train, y_train, epochs=1000, lr=0.05, return_loss=True)
        
        results.append((n_neurons, n_layers, final_loss))
        print(f"Final loss: {final_loss:.4f}")

# Create 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract coordinates
x_coords, y_coords, z_coords = zip(*results)

# Create scatter plot
scatter = ax.scatter(x_coords, y_coords, z_coords, c=z_coords, cmap='viridis')

# Customize plot
ax.set_xlabel('Neurons per Hidden Layer')
ax.set_ylabel('Number of Hidden Layers')
ax.set_zlabel('Final Loss')
plt.title('Neural Network Architecture Performance')

# Add colorbar
plt.colorbar(scatter, label='Loss Value')

# Find best architecture
best_result = min(results, key=lambda x: x[2])
print(f"\nBest architecture found:")
print(f"Neurons per layer: {best_result[0]}")
print(f"Number of hidden layers: {best_result[1]}")
print(f"Loss: {best_result[2]:.4f}")

# Train final model with best architecture
best_layer_sizes = [X_train.shape[1]] + [best_result[0]] * best_result[1] + [y_train.shape[1]]
best_nn = NeuralNetwork(best_layer_sizes)
best_nn.train(X_train, y_train, epochs=2000, lr=0.05)

# Evaluate final model
y_pred = best_nn.predict(X_test)
y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_true)
print(f"\nTest accuracy with best architecture: {accuracy:.4f}")

plt.show()