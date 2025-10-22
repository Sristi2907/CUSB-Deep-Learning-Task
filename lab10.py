import numpy as np
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import re
from typing import Tuple, List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_wikipedia_text() -> str:
    """Fetch clean text from Wikipedia's Deep Learning page."""
    try:
        url = "https://en.wikipedia.org/wiki/Deep_learning"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        if not paragraphs:
            raise ValueError("No paragraphs found in the Wikipedia page")

        text = " ".join(p.get_text() for p in paragraphs)
        text = re.sub(r"\[.*?\]", "", text)  # Remove references
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special chars
        text = re.sub(r"\s+", " ", text).lower().strip()
        return text

    except requests.RequestException as e:
        logging.error(f"Error fetching Wikipedia page: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

def create_cooccurrence_matrix(text: str, window_size: int = 2, max_vocab_size: int = 3000
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """Create co-occurrence matrix with reduced vocab for stability."""
    if not text or not isinstance(text, str):
        raise ValueError("Invalid input text")

    words = text.split()
    if len(words) < 2:
        raise ValueError("Text too short for co-occurrence analysis")

    # Limit vocabulary to most frequent words to prevent memory errors
    freq = defaultdict(int)
    for w in words:
        freq[w] += 1
    vocab = sorted(freq, key=freq.get, reverse=True)[:max_vocab_size]
    word2idx = {word: i for i, word in enumerate(vocab)}

    n_words = len(vocab)
    cooc_matrix = np.zeros((n_words, n_words), dtype=np.float32)

    for i, word in enumerate(words):
        if word not in word2idx:
            continue
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        for j in range(start, end):
            if i != j and words[j] in word2idx:
                cooc_matrix[word2idx[word]][word2idx[words[j]]] += 1

    return cooc_matrix, word2idx, vocab

def get_k_rank_approximation(matrix: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute low-rank SVD approximation."""
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if k <= 0 or k > min(matrix.shape):
        raise ValueError(f"Invalid k value. Must be between 1 and {min(matrix.shape)}")

    try:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        S_k = np.diag(S[:k])
        U_k = U[:, :k]
        Vt_k = Vt[:k, :]

        W_word = np.dot(U_k, np.sqrt(S_k))
        W_context = np.dot(Vt_k.T, np.sqrt(S_k))  # FIXED SHAPE: transpose to align
        return W_word, W_context

    except np.linalg.LinAlgError as e:
        logging.error(f"SVD failed: {e}")
        raise

def predict_next_word(
    input_word: str,
    word2idx: Dict[str, int],
    vocab: List[str],
    W_word: np.ndarray,
    W_context: np.ndarray,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """Predict most likely next words based on cosine similarity."""
    if input_word not in word2idx:
        logging.warning(f"Word '{input_word}' not in vocabulary")
        return []

    word_idx = word2idx[input_word]
    word_vector = W_word[word_idx]

    # Compute cosine similarities
    similarities = np.dot(W_context, word_vector) / (
        np.linalg.norm(W_context, axis=1) * np.linalg.norm(word_vector) + 1e-10
    )

    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [(vocab[idx], float(similarities[idx])) for idx in top_indices]

def main():
    try:
        logging.info("Fetching Wikipedia text...")
        text = get_wikipedia_text()

        logging.info("Creating co-occurrence matrix...")
        cooc_matrix, word2idx, vocab = create_cooccurrence_matrix(text)
        logging.info(f"Vocabulary size: {len(vocab)}")

        k = 50
        logging.info(f"Computing {k}-rank approximation...")
        W_word, W_context = get_k_rank_approximation(cooc_matrix, k)

        input_word = "deep"
        logging.info(f"Predicting next words for '{input_word}'...")
        predictions = predict_next_word(input_word, word2idx, vocab, W_word, W_context)

        if predictions:
            print(f"\nTop predicted words for '{input_word}':")
            for word, score in predictions:
                print(f"{word}: {score:.4f}")
        else:
            print(f"No predictions available for word '{input_word}'")

    except Exception as e:
        logging.error(f"Program failed: {e}")

if __name__ == "__main__":
    main()
