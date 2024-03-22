import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

file = pd.read_csv('MBTI500.csv')

label_encoder = LabelEncoder()
file['type_encoded'] = label_encoder.fit_transform(file['type'])
types_encoded = file['type_encoded'].values

data = np.array(file)

## UNCOMMENT THESE IF NECESSARY
# nltk.download('wordnet')
# nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
word2vec = KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
# word2vec.save("word2vec.model")

## UNCOMMENT IF YOU WANT, .model FILE HAS BEEN SUBMITTTED ALONG WITH THE CODE
# word2vec = KeyedVectors.load("word2vec.model", mmap='r')

def preprocess_text(text):
    text = text.lower() #Convert to lowercase
    word_list = nltk.word_tokenize(text) #Tokenize the text input
    word_list = [lemmatizer.lemmatize(word) for word in word_list] #Lemmatize the words
    return word_list

#pass only the column containing the posts
def get_embeddings(posts):
    posts_embeddings = []
    for post in posts:
        preprocess = preprocess_text(post)
        post_embeddings = np.array([word2vec[word] for word in preprocess if word in word2vec])
        if post_embeddings.size > 0:
            post_embeddings = np.mean(post_embeddings, axis=0)
            posts_embeddings.append(post_embeddings)
        else:
            posts_embeddings.append(np.zeros(300)) #fill with zeros if post has no embeddings present in w2v
    return posts_embeddings

def initialize_parameters():
    #Xavier's initialization to reduce vanishing/exploding gradients
    w1 = np.random.randn(128, 300) * np.sqrt(1/300) # 128 neurons in the first hidden layer
    w2 = np.random.randn(64, 128) * np.sqrt(1/128) # 64 neurons in the second hidden layer
    w = np.random.randn(16, 64) * np.sqrt(1/64) # 16 neurons in the output layer, one for each class
    
    b1 = np.zeros((128, 1))
    b2 = np.zeros((64, 1))
    b = np.zeros((16, 1))

    return w1, b1, w2, b2, w, b

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def forward_propagation(X, w1, b1, w2, b2, w3, b3):
    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = np.tanh(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = softmax(z3)
    
    return z1, a1, z2, a2, z3, a3

def backward_propagation(X, Y, z1, a1, z2, a2, z3, a3, w1, w2, w3):
    m = X.shape[1]
    dz3 = a3 - Y
    dw3 = (1 / m) * np.dot(dz3, a2.T)
    db3 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)
    dz2 = np.multiply(np.dot(w3.T, dz3), 1 - np.power(a2, 2))
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2, dw3, db3

def update_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3

    return w1, b1, w2, b2, w3, b3

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def train(learning_rate, num_epochs, X, Y, w1, b1, w2, b2, w3, b3):
    for i in range(num_epochs):
        z1, a1, z2, a2, z3, a3 = forward_propagation(X, w1, b1, w2, b2, w3, b3)
        dw1, db1, dw2, db2, dw3, db3 = backward_propagation(X, Y, z1, a1, z2, a2, z3, a3, w1, w2, w3)
        w1, b1, w2, b2, w3, b3 = update_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate)
        if i % 100 == 0:
            print(f'Iteration: {i}, Loss: {cross_entropy_loss(Y, a3)}')
    return w1, b1, w2, b2, w3, b3

embedded_data = get_embeddings(data[:, 0])
embedded_data = np.array(embedded_data)
Y = np.eye(16)[types_encoded]

def cross_validation(X, Y, num_folds = 5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        w1, b1, w2, b2, w3, b3 = initialize_parameters()
        w1, b1, w2, b2, w3, b3 = train(0.05, 5, X_train.T, Y_train.T, w1, b1, w2, b2, w3, b3)
        _, _, _, _, _, A3_test = forward_propagation(X_test.T, w1, b1, w2, b2, w3, b3)
        predictions = np.argmax(A3_test, axis=0)
        true_labels = np.argmax(Y_test.T, axis=0)
        accuracy.append(np.mean(predictions == true_labels))
        precision.append(precision_score(true_labels, predictions, average='weighted', zero_division=0))
        recall.append(recall_score(true_labels, predictions, average='weighted', zero_division=0))
        f1.append(f1_score(true_labels, predictions, average='weighted', zero_division=0))

    
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = cross_validation(embedded_data, Y)

print(f'Accuracy: {np.mean(accuracy) * 100:.2f}%')
print(f'Precision: {np.mean(precision) * 100:.2f}%')
print(f'Recall: {np.mean(recall) * 100:.2f}%')
print(f'F1: {np.mean(f1) * 100:.2f}%')

### Forward propagation
    # input: X
    # weights: W
    # bias: b
    # linear function: z = W*X + b
    # activation function: a = g(z)
    # output: softmax(a)
    # prediction: argmax(softmax(a))

### Compute loss function

### Backward propagation

### Compute gradients
