import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#  Step 1: Fetch and Clean Wikipedia Text 
def get_wikipedia_text(url: str) -> str:
    """Fetch and clean Wikipedia text."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        text = re.sub(r"\[.*?\]", "", text)  # remove references
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation/numbers
        text = re.sub(r"\s+", " ", text).lower().strip()
        return text
    except Exception as e:
        logging.error(f"Error fetching Wikipedia text: {e}")
        raise

# Step 2: Preprocess Data 
def preprocess_text(text, window_size=2):
    """Tokenize text and create CBOW training samples."""
    words = text.split()
    vocab = sorted(set(words))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    data = []
    for i in range(window_size, len(words) - window_size):
        context = words[i - window_size:i] + words[i + 1:i + window_size + 1]
        target = words[i]
        data.append((context, target))

    return data, word2idx, idx2word, vocab

# Step 3: One-Hot Encoding 
def one_hot_vector(word, word2idx):
    """Return one-hot vector for a given word."""
    vec = np.zeros(len(word2idx))
    vec[word2idx[word]] = 1
    return vec

#  Step 4: CBOW Model 
class CBOW:
    def __init__(self, vocab_size, embedding_dim=10, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # Initialize weights
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, context_vectors):
        # Average context embeddings
        h = np.mean(np.dot(context_vectors, self.W1), axis=0)
        u = np.dot(h, self.W2)
        y_pred = self.softmax(u)
        return y_pred, h

    def train(self, data, word2idx, epochs=500):
        for epoch in range(epochs):
            total_loss = 0
            for context, target in data:
                # Prepare input
                x = np.array([one_hot_vector(w, word2idx) for w in context])
                y_true = one_hot_vector(target, word2idx)

                # Forward pass
                y_pred, h = self.forward(x)

                # Compute loss
                total_loss += -np.sum(y_true * np.log(y_pred + 1e-10))

                # Backpropagation
                error = y_pred - y_true
                self.W2 -= self.lr * np.outer(h, error)

                # Update each context word in W1
                for w in context:
                    self.W1[word2idx[w]] -= self.lr * np.dot(self.W2, error)

            if (epoch + 1) % 100 == 0:
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    def get_word_vector(self, word, word2idx):
        return self.W1[word2idx[word]]

#  Step 5: Run the Model 
def main():
    url = "https://en.wikipedia.org/wiki/Deep_learning"
    text = get_wikipedia_text(url)

    # Limit text for faster training
    words = text.split()[:1000]
    text = " ".join(words)

    data, word2idx, idx2word, vocab = preprocess_text(text, window_size=2)
    logging.info(f"Vocabulary size: {len(vocab)}, Training samples: {len(data)}")

    cbow = CBOW(vocab_size=len(vocab), embedding_dim=10, learning_rate=0.05)
    cbow.train(data, word2idx, epochs=500)

    # Show embeddings for a few words
    for w in ["deep", "learning", "neural", "network"]:
        if w in word2idx:
            print(f"\nWord vector for '{w}':\n{cbow.get_word_vector(w, word2idx)}")
        else:
            print(f"'{w}' not found in vocabulary")

if __name__ == "__main__":
    main()
