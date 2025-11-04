import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Step 1: Safe NLTK Downloads
def safe_nltk_download(resource):
    """Download NLTK resource safely if not already available."""
    try:
        nltk.data.find(resource)
    except LookupError:
        print(f" Downloading missing resource: {resource}")
        nltk.download(resource.split('/')[-1], quiet=True)

print(" Checking and downloading required NLTK data...")
safe_nltk_download('corpora/gutenberg')
safe_nltk_download('tokenizers/punkt')
safe_nltk_download('tokenizers/punkt_tab')

# Step 2: Load dataset (Gutenberg Corpus)
print("\n Loading dataset...")
sample_text = gutenberg.raw('austen-emma.txt')  # Jane Austen's "Emma"

print("\n Sample text preview:")
print(sample_text[:500])

# Step 3: Text Preprocessing
print("\n Preprocessing text...")
tokens = word_tokenize(sample_text.lower())              # Tokenize text
words = [word for word in tokens if word.isalpha()]      # Keep only alphabetic words

print(f" Total words after cleaning: {len(words)}")
print(f" Sample tokens: {words[:20]}")

# Step 4: Prepare data for Word2Vec
data = []
sentence_length = 20
for i in range(0, len(words) - sentence_length, sentence_length):
    data.append(words[i:i + sentence_length])

print(f" Total sentences prepared: {len(data)}")
print(f" Example sentence: {data[0]}")

# Step 5: Train Word2Vec model
print("\n Training Word2Vec model...")
model = Word2Vec(
    sentences=data,
    vector_size=100,   # Dimensionality of word embeddings
    window=5,          # Context window size
    min_count=3,       # Ignore words with freq < 3
    workers=4,         # Use multiple CPU cores
    sg=0               # 0 = CBOW, 1 = Skip-Gram
)

# Save the model
model.save("word2vec_model.model")
print(" Model trained and saved successfully as 'word2vec_model.model'!")

# Step 6: Explore the model
print("\n Top 10 words similar to 'love':")
try:
    similar_words = model.wv.most_similar('love', topn=10)
    for word, score in similar_words:
        print(f"{word}: {score:.4f}")
except KeyError:
    print(" The word 'love' is not in the vocabulary!")

print("\n Analogy Test: King - Man + Woman = ?")
try:
    result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
    print(result)
except KeyError:
    print(" One of the analogy words is not in vocabulary!")

print("\n Similarity Test between 'good' and 'bad':")
try:
    similarity = model.wv.similarity('good', 'bad')
    print(f"Similarity score: {similarity:.4f}")
except KeyError:
    print(" One of the words not in vocabulary!")

# Step 7: Visualize Word Embeddings
print("\n Visualizing word embeddings using PCA...")

words_to_visualize = ['love', 'friend', 'marriage', 'woman', 'man',
                      'happy', 'sad', 'rich', 'poor', 'beautiful']

# Filter out words not in vocabulary
words_filtered = [w for w in words_to_visualize if w in model.wv.key_to_index]

if words_filtered:
    X = model.wv[words_filtered]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words_filtered):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.title("Word2Vec Embeddings (PCA Projection)")
    plt.show()
else:
    print(" None of the words for visualization were found in the vocabulary.")

print("\n Word2Vec training and visualization completed successfully!")
