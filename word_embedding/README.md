# Word Embedding Comparison Project

This project explores and compares different word embedding techniques, including Word2Vec, GloVe, and FastText, by calculating the cosine similarity between word pairs and evaluating these similarities against human judgment using Spearman's rank correlation. This provides insight into how well each embedding captures semantic relationships according to human perception.

## Overview

Word embeddings play a crucial role in NLP by providing a dense, vector-based representation of words that capture their meanings and semantic relationships. This project focuses on:

- **Word2Vec**: A neural network-based approach to learn word associations from large text corpora.
- **GloVe (Global Vectors for Word Representation)**: Utilizes global word-word co-occurrence statistics from a corpus to generate word vectors.
- **FastText**: Enhances Word2Vec by representing each word as an n-gram of characters, aiding in understanding morphologically rich languages and handling out-of-vocabulary words.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system to run the project files. The project requires the following Python libraries:

- pandas
- numpy
- gensim
- scipy

You can install these dependencies via pip:

`pip install pandas numpy gensim scipy`


### Download Required Models

Download the pre-trained models necessary for this project:

- **Word2Vec**: [GoogleNews-vectors-negative300](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors)
- **GloVe**: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) (Download the 6B tokens, 50d vectors version)
- **FastText**: [fasttext.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)

### Setup

1. **Clone the Repository**: Clone this repository to obtain the project files.
2. **Download and Prepare the Embedding Models**: Follow the links provided above to download the word embedding models.
3. **Prepare the Data**: The project uses a dataset (`combined.csv`) of word pairs and their similarity scores. Ensure this file is placed in the root directory of the project.
