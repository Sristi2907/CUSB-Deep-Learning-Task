import numpy as np

# Activation Functions and Their Derivatives

def relu(x):
    return np.maximum(0, x)

def relu_grad(a):
    return (a > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(a):
    return a * (1 - a)

def tanh(x):
    return np.tanh(x)

def tanh_grad(a):
    return 1 - a**2

def softmax(logits):
    # Stabilize by subtracting max
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(logits)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

def cross_entropy(pred_probs, true_onehot):
    n = pred_probs.shape[0]
    return -np.sum(true_onehot * np.log(pred_probs + 1e-12)) / n

def one_hot_encode(y, num_classes):
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded

# Simple Feedforward Neural Network

class SimpleNN:
    def __init__(self, input_dim, hidden_layers, output_dim,
                 activation="relu", seed=42):
        np.random.seed(seed)

        # layer sizes
        self.layer_dims = [input_dim] + hidden_layers + [output_dim]

        # weights & biases initialization
        self.W = []
        self.b = []
        for in_dim, out_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
           
            self.W.append(np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim))
            self.b.append(np.zeros((1, out_dim)))

        self.activation = activation

    # Activation helpers
    def _activate(self, x):
        if self.activation == "relu":
            return relu(x)
        elif self.activation == "sigmoid":
            return sigmoid(x)
        elif self.activation == "tanh":
            return tanh(x)

    def _activate_grad(self, a):
        if self.activation == "relu":
            return relu_grad(a)
        elif self.activation == "sigmoid":
            return sigmoid_grad(a)
        elif self.activation == "tanh":
            return tanh_grad(a)

    # Forward propagation
    def forward(self, X):
        A, Z = [X], []
        for idx, (W, b) in enumerate(zip(self.W, self.b)):
            z = A[-1] @ W + b
            Z.append(z)
            if idx < len(self.W) - 1:
                A.append(self._activate(z))
            else:
                A.append(z) 
        return Z, A

    def predict_proba(self, X):
        _, activations = self.forward(X)
        return softmax(activations[-1])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # Backward propagation
    def compute_grads(self, X, y_onehot):
        m = X.shape[0]
        Z, A = self.forward(X)

        # initial delta for softmax + cross-entropy
        probs = softmax(A[-1])
        delta = (probs - y_onehot) / m

        dW, db = [None] * len(self.W), [None] * len(self.b)

        for i in reversed(range(len(self.W))):
            dW[i] = A[i].T @ delta
            db[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                delta = (delta @ self.W[i].T) * self._activate_grad(A[i])
        return dW, db

    # Training with optimizers

    def fit(self, X, y, epochs=100, lr=0.01, batch_size=32,
            optimizer="sgd", beta=0.9, beta2=0.999, eps=1e-8,
            decay=0.0, verbose=True):

        num_classes = len(np.unique(y))
        y_onehot = one_hot_encode(y, num_classes)
        n_samples = X.shape[0]

        # optimizer state initialization
        vW = [np.zeros_like(W) for W in self.W]
        vB = [np.zeros_like(b) for b in self.b]
        mW = [np.zeros_like(W) for W in self.W]
        mB = [np.zeros_like(b) for b in self.b]

        for epoch in range(1, epochs + 1):
            # apply learning rate decay
            lr_epoch = lr / (1 + decay * epoch)

            # shuffle data
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y_onehot[idx]

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuf[start:start + batch_size]
                y_batch = y_shuf[start:start + batch_size]

                dW, db = self.compute_grads(X_batch, y_batch)

                #Optimizer updates
                if optimizer == "sgd":
                    for j in range(len(self.W)):
                        self.W[j] -= lr_epoch * dW[j]
                        self.b[j] -= lr_epoch * db[j]

                elif optimizer == "momentum":
                    for j in range(len(self.W)):
                        vW[j] = beta * vW[j] + (1 - beta) * dW[j]
                        vB[j] = beta * vB[j] + (1 - beta) * db[j]
                        self.W[j] -= lr_epoch * vW[j]
                        self.b[j] -= lr_epoch * vB[j]

                elif optimizer == "nesterov":
                    for j in range(len(self.W)):
                        prev_vW, prev_vB = vW[j].copy(), vB[j].copy()
                        vW[j] = beta * vW[j] + lr_epoch * dW[j]
                        vB[j] = beta * vB[j] + lr_epoch * db[j]
                        self.W[j] -= -beta * prev_vW + (1 + beta) * vW[j]
                        self.b[j] -= -beta * prev_vB + (1 + beta) * vB[j]

                elif optimizer == "adagrad":
                    for j in range(len(self.W)):
                        vW[j] += dW[j] ** 2
                        vB[j] += db[j] ** 2
                        self.W[j] -= lr_epoch * dW[j] / (np.sqrt(vW[j]) + eps)
                        self.b[j] -= lr_epoch * db[j] / (np.sqrt(vB[j]) + eps)

                elif optimizer == "rmsprop":
                    for j in range(len(self.W)):
                        vW[j] = beta * vW[j] + (1 - beta) * (dW[j] ** 2)
                        vB[j] = beta * vB[j] + (1 - beta) * (db[j] ** 2)
                        self.W[j] -= lr_epoch * dW[j] / (np.sqrt(vW[j]) + eps)
                        self.b[j] -= lr_epoch * db[j] / (np.sqrt(vB[j]) + eps)

                elif optimizer == "adam":
                    for j in range(len(self.W)):
                        mW[j] = beta * mW[j] + (1 - beta) * dW[j]
                        mB[j] = beta * mB[j] + (1 - beta) * db[j]
                        vW[j] = beta2 * vW[j] + (1 - beta2) * (dW[j] ** 2)
                        vB[j] = beta2 * vB[j] + (1 - beta2) * (db[j] ** 2)

                        # bias correction
                        mW_hat = mW[j] / (1 - beta ** epoch)
                        mB_hat = mB[j] / (1 - beta ** epoch)
                        vW_hat = vW[j] / (1 - beta2 ** epoch)
                        vB_hat = vB[j] / (1 - beta2 ** epoch)

                        self.W[j] -= lr_epoch * mW_hat / (np.sqrt(vW_hat) + eps)
                        self.b[j] -= lr_epoch * mB_hat / (np.sqrt(vB_hat) + eps)

            # progress logging
            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                probs = self.predict_proba(X)
                loss = cross_entropy(probs, y_onehot)
                acc = np.mean(self.predict(X) == y)
                print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")


# Example Run (Iris dataset)

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X = (iris.data - iris.data.mean(0)) / iris.data.std(0)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SimpleNN(input_dim=4, hidden_layers=[16, 12], output_dim=3)

    # Ensure verbose is explicitly set to True to see training progress
    model.fit(X_train, y_train, epochs=100, lr=0.05, optimizer="adam", verbose=True)

    preds = model.predict(X_test)
    test_accuracy = np.mean(preds == y_test)
    print("Test Accuracy:", test_accuracy)
    print("This is a verification that print statements are working")
