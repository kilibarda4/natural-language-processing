# Neural Network for Text Classification

This project demonstrates the implementation of a neural network from scratch to perform text classification based on Myers-Briggs Type Indicator (MBTI) personality types. Utilizing word embeddings for text representation, this neural network employs a three-layer architecture with forward and backward propagation, trained to predict MBTI types from posts.

## Overview

The neural network is designed with the following components:

- **Preprocessing**: Tokenization and lemmatization of text data.
- **Word Embeddings**: Use of pre-trained Word2Vec embeddings to represent words.
- **Neural Network Architecture**: A three-layer model including input, hidden, and output layers, with tanh activation functions (tanh) and a softmax output layer.
- **Training Process**: Implements forward propagation, loss computation, backward propagation, and parameter update steps.
- **Evaluation**: Uses K-Fold cross-validation to evaluate model performance based on accuracy, precision, recall, and F1 score metrics.

## Dependencies

- Python 3.x
- pandas
- numpy
- nltk
- gensim
- scikit-learn

*These libraries were used for preprocessing and validation, the algorithm to train and update the network was built using only numpy!*

If you are missing the above Python libraries, you can install them using pip:

`pip install pandas numpy nltk gensim scikit-learn`

## Setup

1. **Clone or download this repository** to your local machine.
2. **Download the pre-trained Word2Vec model** and place it in the project directory.
3. Ensure the dataset ([MBTI500.csv](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset/data)) is placed in the root directory of your project.

## Running the Script

To run the script, navigate to the project directory and execute:

`python neural.py`

## How It Works

The script performs the following steps:

1. **Preprocesses the text data** from the `MBTI500.csv` file, applying tokenization and lemmatization.
2. **Generates embeddings** for each post by averaging the Word2Vec vectors of the words in the post.
3. **Initializes the neural network parameters** using Xavier's initialization method to reduce the problem of vanishing/exploding gradients.
4. **Trains the neural network** using the training subset of the data, performing forward propagation to make predictions, computing the cross-entropy loss, and applying backward propagation to update the model parameters.
5. **Evaluates the model** on the test subset using K-Fold cross-validation and computes the accuracy, precision, recall, and F1 score.

## Customizing the Model

You can customize the learning rate, number of epochs, and the architecture of the neural network (e.g., number of neurons in the hidden layers) by modifying the relevant parts of the script.

## Note

This project is for educational purposes and demonstrates basic principles of building and training a neural network from scratch. For more advanced applications, consider using deep learning frameworks like TensorFlow or PyTorch.
