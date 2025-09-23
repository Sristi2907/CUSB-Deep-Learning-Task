# CUSB Deep Learning Task — Repository

Small collection of lab exercises and example implementations for introductory deep learning / neural network experiments.

## 📂 Contents

### Core Lab Implementations

- **`lab4/`** *(Coming Soon)*  
  Basic neural network implementations and experiments.

- **`lab5.py`**  
  Neural network implementation with multiple optimizers:
  - Supported optimizers: SGD, Momentum, Adagrad, RMSProp, Adam
  - Core utilities: `softmax`, `cross_entropy`, `one_hot_encode`
  - Class: `SimpleNN` for forward/backward pass demonstrations

- **`lab6.py`**  
  PCA visualization for dimensionality reduction:
  - Dataset: `sklearn.datasets.load_digits`
  - Output: `pca_digits_visualization_lab6.1.png`
  - Visualizes two principal components
  - Reports explained variance ratios

- **`lab7.py`**  
  Iris dataset classification using neural networks:
  - Data preprocessing (standardization)
  - One-hot encoding implementation
  - 2D/3D visualization of results
  - Training and evaluation metrics

- **`lab8.py`**  
  Advanced neural network with autoencoder pre-training:
  - `Autoencoder` class for unsupervised pre-training
  - `DeepNeuralNetwork` class for supervised learning
  - Layer-wise pre-training implementation
  - Xavier/Glorot weight initialization
  - Cross-entropy loss and softmax output
  - Training visualization

### Additional Examples

- **`mp_neuron_logic_gates.py`**  
  McCulloch-Pitts neuron demonstrations:
  - Logic gate implementations (AND/OR/NOT)
  - XOR problem demonstration
  - Uses PyTorch tensors for computations

### Work in Progress

- `Lab4.1/`
- `MLP_XOR/`
- `MNIST/`

## ⚙️ Requirements

- Python 3.8+
- Key Dependencies:
  ```
  numpy
  matplotlib
  scikit-learn
  pandas
  torch  # Optional: only for mp_neuron_logic_gates.py
  ```

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/macOS

# Install dependencies
pip install numpy matplotlib scikit-learn pandas
pip install torch  # Optional
```

## 🚀 Running the Examples

### Basic Usage

```bash
# From repository root:
python lab6.py  # Generates PCA visualization
python lab7.py  # Iris classification
python lab8.py  # Autoencoder + Deep NN
```

### Lab 8 Specific Instructions

```bash
python lab8.py
```

This will:
1. Pre-train the network using autoencoders
2. Fine-tune using supervised learning
3. Display training progress
4. Show final accuracy
5. Plot loss curve

## 📊 Output Files

- `pca_digits_visualization_lab6.1.png`: PCA visualization from lab6
- `dnn_training_loss.png`: Training curves from lab8

## 🛠️ Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **Plot Display Issues**
   - Use `plt.savefig()` for headless environments
   - Set matplotlib backend: `matplotlib.use('Agg')`

3. **CUDA/GPU Issues**
   - Code runs on CPU by default
   - No GPU required

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📌 License

This repository is provided for educational purposes.
Feel free to use and modify the code for academic projects.

## 🔄 Updates

- Added comprehensive documentation for lab8.py
- Improved installation instructions
- Added troubleshooting section
