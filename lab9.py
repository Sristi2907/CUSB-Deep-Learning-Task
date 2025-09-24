import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict

def preprocess_text(text: str) -> List[str]:
    """
    Clean and tokenize text.
    
    Args:
        text (str): Input text string
    Returns:
        List[str]: List of cleaned and tokenized words
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return [word for word in text.split() if word]

def build_vocabulary(corpus: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Build vocabulary from tokenized corpus.
    
    Args:
        corpus (List[str]): List of sentences
    Returns:
        Tuple[List[str], Dict[str, int]]: Vocabulary list and word-to-index mapping
    """
    if not corpus:
        raise ValueError("Corpus cannot be empty")
    vocab = sorted(list(set(
        word for sentence in corpus
        for word in preprocess_text(sentence)
    )))
    if not vocab:
        raise ValueError("No valid words found in corpus")
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2idx

def create_cooccurrence_matrix(corpus: List[str], window_size: int = 2) -> Tuple[np.ndarray, List[str]]:
    """
    Create co-occurrence matrix from corpus.
    
    Args:
        corpus (List[str]): List of sentences
        window_size (int): Size of context window
    Returns:
        Tuple[np.ndarray, List[str]]: Co-occurrence matrix and vocabulary
    """
    if window_size < 1:
        raise ValueError("Window size must be positive")
    try:
        tokenized_corpus = [preprocess_text(sentence) for sentence in corpus]
        vocab, word2idx = build_vocabulary(corpus)
        matrix_size = len(vocab)
        cooccurrence_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)

        for sentence in tokenized_corpus:
            for i, target_word in enumerate(sentence):
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                for j in range(start, end):
                    if i != j and j < len(sentence):  # Added bounds check
                        context_word = sentence[j]
                        target_idx = word2idx[target_word]
                        context_idx = word2idx[context_word]
                        cooccurrence_matrix[target_idx][context_idx] += 1.0

        return cooccurrence_matrix, vocab

    except Exception as e:
        raise RuntimeError(f"Error creating co-occurrence matrix: {str(e)}")

def plot_heatmap(matrix: np.ndarray, vocab: List[str], save_path: str = None) -> None:
    """
    Plot the co-occurrence matrix as a heatmap.
    
    Args:
        matrix (np.ndarray): Co-occurrence matrix
        vocab (List[str]): Vocabulary list
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, 
                xticklabels=vocab, 
                yticklabels=vocab, 
                cmap="YlGnBu", 
                annot=True, 
                fmt=".0f", 
                linewidths=0.5)
    plt.title("Word Co-occurrence Matrix Heatmap", fontsize=16, pad=20)
    plt.xlabel("Context Words", fontsize=14)
    plt.ylabel("Target Words", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics(matrix: np.ndarray, vocab: List[str]) -> None:
    """
    Print statistical information about the co-occurrence matrix.
    
    Args:
        matrix (np.ndarray): Co-occurrence matrix
        vocab (List[str]): Vocabulary list
    """
    print("\nMatrix Statistics:")
    print(f"Total word pairs: {np.sum(matrix):.0f}")
    print(f"Unique words: {len(vocab)}")
    print(f"Most frequent co-occurring words:")
    
    # Get top 5 co-occurrences
    flat_indices = np.argsort(matrix.flatten())[-5:]
    row_indices, col_indices = np.unravel_index(flat_indices, matrix.shape)
    
    for i, j in zip(row_indices, col_indices):
        print(f"  {vocab[i]} - {vocab[j]}: {matrix[i, j]:.0f}")

def main():
    """Main function to demonstrate co-occurrence matrix creation and visualization."""
    try:
        corpus = [
            "I love machine learning",
            "I love deep learning",
            "machine learning is fascinating",
            "deep learning is amazing",
            "artificial intelligence and machine learning"
        ]

        print("Processing corpus...")
        cooccurrence_matrix, vocabulary = create_cooccurrence_matrix(corpus, window_size=2)

        print("\nVocabulary:")
        print(vocabulary)

        # Print statistical information
        print_statistics(cooccurrence_matrix, vocabulary)

        # Plot and save the heatmap
        plot_heatmap(cooccurrence_matrix, vocabulary, save_path="cooccurrence_heatmap.png")

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 
