import pandas as pd
import numpy as np
import gensim.models.keyedvectors as Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from scipy.stats import spearmanr

# Load the data
data = pd.read_csv('combined.csv')

# Extract word pairs and human scores
word_pairs = data[['Word 1', 'Word 2']].values.tolist()
human_scores = data['Human (mean)'].tolist()

# Get unique words
unique_words = set(word for pair in word_pairs for word in pair)

# Load the GloVe model
glove = {}
with open('glove.6B.50d.txt', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype=float)
        glove[word] = vector

# Load the models
word2vec = Word2Vec.KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
fasttext = KeyedVectors.load_word2vec_format('fasttext.vec')

# Create a dictionary of word embeddings
word_embeddings_word2vec = {word: word2vec[word] for word in unique_words if word in word2vec}
word_embeddings_fasttext = {word: fasttext[word] for word in unique_words if word in fasttext}
word_embeddings_glove = {word: glove[word] for word in unique_words if word in glove}


# word_embeddings_glove
def cosine_sim_w2v(word1, word2):
    if word1 in word_embeddings_word2vec and word2 in word_embeddings_word2vec:
        return np.dot(word_embeddings_word2vec[word1], word_embeddings_word2vec[word2]) / (np.linalg.norm(word_embeddings_word2vec[word1]) * np.linalg.norm(word_embeddings_word2vec[word2]))
    else:
        return None
def cosine_sim_ft(word1, word2):
    if word1 in word_embeddings_fasttext and word2 in word_embeddings_fasttext:
        return np.dot(word_embeddings_fasttext[word1], word_embeddings_fasttext[word2]) / (np.linalg.norm(word_embeddings_fasttext[word1]) * np.linalg.norm(word_embeddings_fasttext[word2]))
    else:
        return None
def cosine_sim_glove(word1, word2):
    if word1 in word_embeddings_glove and word2 in word_embeddings_glove:
        return np.dot(word_embeddings_glove[word1], word_embeddings_glove[word2]) / (np.linalg.norm(word_embeddings_glove[word1]) * np.linalg.norm(word_embeddings_glove[word2]))
    else:
        return None

# lowercase the word pairs and pass into the cosine similarity functions
cosine_similarities_w2v = [cosine_sim_w2v(word1.lower(), word2.lower()) for word1, word2 in word_pairs]
cosine_similarities_ft = [cosine_sim_ft(word1.lower(), word2.lower()) for word1, word2 in word_pairs]
cosine_similarities_glove = [cosine_sim_glove(word1.lower(), word2.lower()) for word1, word2 in word_pairs]

# Spearman rank correlation
def rank(cosine_similarities):
    cosine_similarities_filtered = [similarity for similarity in cosine_similarities if similarity is not None]
    return np.argsort(np.argsort(cosine_similarities_filtered))

def spearman(human_scores, cosine_sim):
    cosine_sim_filtered = [similarity for similarity in cosine_sim if similarity is not None]
    human_scores_filtered = [score for i, score in enumerate(human_scores) if cosine_sim[i] is not None]

    rank_human = rank(human_scores_filtered)
    rank_cosine = rank(cosine_sim_filtered)
    length = len(human_scores_filtered)
    sum_d = 0  # Initialize sum_d
    for i in range(length):
        sum_d += (rank_human[i] - rank_cosine[i]) ** 2

    return 1 - (6 * sum_d) / (length * (length ** 2 - 1)) if length > 1 else 0

print("Spearman rank correlation for word2vec: ", spearman(human_scores, cosine_similarities_w2v))
print("Spearman rank correlation for fasttext: ", spearman(human_scores, cosine_similarities_ft))
print("Spearman rank correlation for glove: ", spearman(human_scores, cosine_similarities_glove))