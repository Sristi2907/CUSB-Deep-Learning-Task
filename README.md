# 🧠 Deep Learning Tasks Collection

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

Welcome to our comprehensive Deep Learning repository! This collection offers hands-on implementations of various neural network architectures and deep learning concepts. Perfect for both beginners and intermediate learners! 🚀

## 📚 Repository Structure

### 🔮 Foundational Concepts
| Project | Description |
|---------|-------------|
| [`mp_neuron_logic_gates.py`](mp_neuron_logic_gates.py) | 🧮 McCulloch-Pitts neuron implementing basic logic gates |
| [`MLP_XOR`](MLP_XOR) | 🔄 XOR problem solution using Multi-Layer Perceptron |

### 🏗️ Neural Network Implementations
| Project | Description |
|---------|-------------|
| [`Lab4`](Lab4) & [`Lab4.1`](Lab4.1) | 🌺 Iris dataset classification using neural networks |
| [`lab5.py`](lab5.py) | ⚡ Activation functions & optimizer implementations |
| [`lab6.py`](lab6.py) | 📊 MNIST visualization using PCA |
| [`lab7.py`](lab7.py) | 🎨 Neural architecture exploration & visualization |
| [`lab8.py`](lab8.py) | 🎯 Autoencoder implementation with visualizations |

### 🤖 Applications
| Project | Description |
|---------|-------------|
| [`MNIST`](MNIST) | ✍️ Digit classification using TensorFlow/Keras |

## 🛠️ Technical Features

### Core Components
- 🧠 **Neural Networks**: Progressive implementations from basic to advanced
- 🔥 **Activation Functions**: 
  - ReLU (Rectified Linear Unit)
  - Sigmoid
  - Tanh
  - Softmax
- 📉 **Loss Functions**:
  - Cross-entropy
  - Mean Squared Error (MSE)

### Advanced Features
- 🚀 **Optimization Techniques**:
  ```python
  optimizers = {
      'SGD': 'Stochastic Gradient Descent',
      'Momentum': 'SGD with Momentum',
      'Adam': 'Adaptive Moment Estimation',
      'RMSprop': 'Root Mean Square Propagation'
  }
  ```
- 📊 **Visualization Tools**:
  - PCA dimensionality reduction
  - Loss/accuracy curves
  - Network architecture diagrams
  - Training progress monitors

## ⚙️ Setup & Requirements

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv deeplearning-env
.\deeplearning-env\Scripts\activate  # Windows
```

### Dependencies
```python
# requirements.txt
numpy>=1.19.2
torch>=1.7.0
tensorflow>=2.4.0
scikit-learn>=0.24.0
matplotlib>=3.3.2
pandas>=1.2.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start Guide

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/deep-learning-tasks.git
   cd deep-learning-tasks
   ```

2. **Run individual projects**:
   ```bash
   # Basic concepts
   python mp_neuron_logic_gates.py

   # Neural architectures
   python lab7.py

   # Autoencoders
   python lab8.py
   ```

## 📈 Learning Path

1. Start with `mp_neuron_logic_gates.py` for basic neural concepts
2. Move to `MLP_XOR` for fundamental neural network understanding
3. Explore `Lab4` series for practical implementation
4. Progress through `lab5.py` to `lab8.py` for advanced concepts
5. Complete with MNIST implementation for real-world application

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed for educational purposes. Feel free to use and modify for learning!

## ⭐ Support

If you find this repository helpful, please consider giving it a star!

---

*Happy Deep Learning! May your gradients be stable and your losses low! 🚀*
