import os
import numpy as np

training_set = 'WSJ_02-21.pos'
testing_set = 'WSJ_24.pos'

with open(training_set, 'r') as file:
    lines = [next(file) for _ in range(10)]

# Read the file and parse it
def read_data(file_path):
    
    sentences = []
    current_sentence = [] 
    
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split the line into word and tag
            stripped_line = line.strip()
            if stripped_line:
                word, tag = stripped_line.split('\t')
                current_sentence.append((word, tag))
            else:  
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def initial_counts(sentences):
    transition_counts = {}
    emission_counts = {}
    tag_counts = {}
    start_tag_counts = {}

    for sentence in sentences:
        previous_tag = None
        for word, tag in sentence:
            # Increment the tag counts
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Increment the emission counts
            if tag not in emission_counts:
                emission_counts[tag] = {}
            emission_counts[tag][word] = emission_counts[tag].get(word, 0) + 1

            # Increment the transition counts
            if previous_tag is not None:
                if previous_tag not in transition_counts:
                    transition_counts[previous_tag] = {}
                transition_counts[previous_tag][tag] = transition_counts[previous_tag].get(tag, 0) + 1

            previous_tag = tag

        # If this is the first word in a sentence, increment the start tag counts
        start_tag = sentence[0][1] if sentence else None
        if start_tag:
            start_tag_counts[start_tag] = start_tag_counts.get(start_tag, 0) + 1

    return transition_counts, emission_counts, tag_counts, start_tag_counts

def counts_to_probs(transition_counts, emission_counts, start_tag_counts, len_sentences, vocab_size, tag_set_size):
    transition_probabilities = {}
    emission_probabilities = {}
    start_tag_probabilities = {}

    # Calculate transition probabilities with smoothing
    for prev_tag, tag_transitions in transition_counts.items():
        total_transitions = sum(tag_transitions.values()) + (tag_set_size * 1)  # Add-one smoothing for each possible tag
        transition_probabilities[prev_tag] = {tag: (count + 1) / total_transitions for tag, count in tag_transitions.items()}
        # Ensure all possible tags are in the dictionary, even those not seen after prev_tag
        for possible_tag in transition_counts:
            if possible_tag not in transition_probabilities[prev_tag]:
                transition_probabilities[prev_tag][possible_tag] = 1 / total_transitions

    # Calculate emission probabilities with smoothing
    for tag, tag_emissions in emission_counts.items():
        total_emissions = sum(tag_emissions.values()) + (vocab_size * 1)  # Add-one smoothing for each word in the vocabulary
        emission_probabilities[tag] = {word: (count + 1) / total_emissions for word, count in tag_emissions.items()}

    # Calculate start tag probabilities with smoothing
    total_starts = sum(start_tag_counts.values()) + (tag_set_size * 1)  # Add-one smoothing for each tag
    start_tag_probabilities = {tag: (count + 1) / total_starts for tag, count in start_tag_counts.items()}
    # Ensure all possible tags are represented in the start probabilities
    for possible_tag in transition_counts:
        if possible_tag not in start_tag_probabilities:
            start_tag_probabilities[possible_tag] = 1 / total_starts

    return transition_probabilities, emission_probabilities, start_tag_probabilities

def forward_algorithm(sentences, transition_probabilities, emission_probabilities, start_tag_probabilities):
    forward = []
    for sentence in sentences:
        # Initialize forward probabilities matrix with dimensions [T x N]
        # T = length of the sentence, N = number of tags
        forward_sentence = []
        for i, (word, _) in enumerate(sentence):
            forward_t = {}
            for tag in emission_probabilities:
                if i == 0:  # If this is the first word, use the start probabilities
                    forward_t[tag] = start_tag_probabilities.get(tag, 0) * emission_probabilities[tag].get(word, 0)
                else:
                    # Sum over all possible states that could have transitioned to the current state
                    forward_t[tag] = sum(forward_sentence[i-1][prev_tag] * transition_probabilities[prev_tag].get(tag, 0)
                                          for prev_tag in transition_probabilities) * emission_probabilities[tag].get(word, 0)
            # Normalize to prevent underflow
            normalization_factor = sum(forward_t.values())
            for tag in forward_t:
                forward_t[tag] /= normalization_factor
            forward_sentence.append(forward_t)
        forward.append(forward_sentence)
    return forward

def backward_algorithm(sentences, transition_probabilities, emission_probabilities):
    backward = []
    for sentence in sentences:
        # Initialize backward probabilities matrix with dimensions [T x N]
        T = len(sentence)
        N = len(transition_probabilities)
        backward_sentence = [{} for _ in range(T)]
        # Initialize the probabilities at the last time step to 1
        for tag in transition_probabilities:
            backward_sentence[T-1][tag] = 1.0

        # Go backwards in time
        for i in range(T-2, -1, -1):
            for tag in transition_probabilities:
                backward_sentence[i][tag] = sum(transition_probabilities[tag].get(next_tag, 0) *
                                                emission_probabilities[next_tag].get(sentence[i+1][0], 0) *
                                                backward_sentence[i+1][next_tag] for next_tag in transition_probabilities)
                # Normalize to prevent underflow
                normalization_factor = sum(backward_sentence[i].values())
                for tag in backward_sentence[i]:
                    backward_sentence[i][tag] /= normalization_factor
        backward.append(backward_sentence)
    return backward

# Function to combine forward and backward probabilities to calculate the posteriors
def compute_posteriors(forward, backward):
    posteriors = []
    for i in range(len(forward)):
        sentence_posteriors = []
        for t in range(len(forward[i])):
            total_probability = sum(forward[i][t][tag] * backward[i][t][tag] for tag in forward[i][t])
            posteriors_t = {tag: (forward[i][t][tag] * backward[i][t][tag]) / total_probability for tag in forward[i][t]}
            sentence_posteriors.append(posteriors_t)
        posteriors.append(sentence_posteriors)
    return posteriors

from collections import defaultdict

def maximization_step(sentences, posteriors, len_vocab, len_tag_set):
    updated_transition_counts = defaultdict(lambda: defaultdict(int))
    updated_emission_counts = defaultdict(lambda: defaultdict(int))
    updated_start_tag_counts = defaultdict(int)

    # Calculate the updated counts for emissions and transitions based on posteriors
    for sentence, sentence_posteriors in zip(sentences, posteriors):
        for t, (word, tag) in enumerate(sentence):
            for possible_tag in sentence_posteriors[t]:
                # Update emission counts
                updated_emission_counts[possible_tag][word] += sentence_posteriors[t][possible_tag]
                
                # Update start tag counts if it's the first word in the sentence
                if t == 0:
                    updated_start_tag_counts[possible_tag] += sentence_posteriors[t][possible_tag]
                
                # Update transition counts
                if t > 0:
                    prev_tag = sentence[t-1][1]
                    updated_transition_counts[prev_tag][possible_tag] += sentence_posteriors[t-1][prev_tag]

    # Convert the updated counts to probabilities
    updated_transition_probabilities, updated_emission_probabilities, updated_start_tag_probabilities = counts_to_probs(updated_transition_counts,
                                                                                                                            updated_emission_counts,
                                                                                                                            updated_start_tag_counts,
                                                                                                                            len(sentences), len_vocab, len_tag_set)
    return updated_transition_probabilities, updated_emission_probabilities, updated_start_tag_probabilities

def compute_likelihood(sentences, forward):
    likelihood = 0.0
    for i, sentence in enumerate(sentences):

        sentence_likelihood = sum(forward[i][-1].values())
        
        if sentence_likelihood == 0:
            sentence_likelihood = 1e-10
        
        likelihood += np.log(sentence_likelihood)
    return likelihood

def viterbi_algorithm(sentence, transition_probs, emission_probs, start_probs):
    states = list(transition_probs.keys())
    V = [{}]
    path = {}

    for state in states:
        V[0][state] = start_probs.get(state, 0) * emission_probs[state].get(sentence[0], 0)
        path[state] = [state]
    
    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}
        
        for y in states:
            (prob, state) = max(
                (V[t-1][y0] * transition_probs[y0].get(y, 0) * emission_probs[y].get(sentence[t], 0), y0) for y0 in states
            )
            V[t][y] = prob
            new_path[y] = path[state] + [y]

        path = new_path
    
    n = 0
    if len(sentence) != 1:
        n = t
    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])

def predict_tags(sentences, transition_probabilities, emission_probabilities, start_tag_probabilities):
    predicted_tags = []
    for sentence in sentences:
        sentence_words = [word for word, _ in sentence]
        _, predicted_tags_sentence = viterbi_algorithm(sentence_words, transition_probabilities, emission_probabilities, start_tag_probabilities)
        predicted_tags.append(predicted_tags_sentence)
    return predicted_tags

def evaluate_accuracy(true_tags, predicted_tags):
    correct = sum(t1 == t2 for sentence_true, sentence_pred in zip(true_tags, predicted_tags) for t1, t2 in zip(sentence_true, sentence_pred))
    total = sum(len(sentence) for sentence in true_tags)
    return correct / total

# Read the training and testing data
train_sentences = read_data(training_set)
test_sentences = read_data(testing_set)

# Initialize the transition, emission, and start probabilities
transition_counts, emission_counts, tag_counts, start_tag_counts = initial_counts(train_sentences)
vocab = set()
tag_set = set()

for sentence in train_sentences:
    for word, tag in sentence:
        vocab.add(word)
        tag_set.add(tag)

transition_probs, emission_probs, start_probs = counts_to_probs(transition_counts, emission_counts, start_tag_counts, len(train_sentences), len(vocab), len(tag_set))

max_epochs = 3
convergence_threshold = 1e-4
previous_likelihood = float('-inf')
converged = False

for epoch in range(max_epochs):
    # E-step
    forward = forward_algorithm(train_sentences, transition_probs, emission_probs, start_probs)
    backward = backward_algorithm(train_sentences, transition_probs, emission_probs)
    posteriors = compute_posteriors(forward, backward)
    
    # M-step
    transition_probs, emission_probs, start_probs = maximization_step(train_sentences, posteriors, len(vocab), len(tag_set))
    
    # Compute likelihood for convergence check
    current_likelihood = compute_likelihood(train_sentences, forward)
    likelihood_change = current_likelihood - previous_likelihood
    if abs(likelihood_change) < convergence_threshold:
        converged = True
        break
    previous_likelihood = current_likelihood
    print(f"Epoch {epoch}: Likelihood = {current_likelihood}")

if converged:
    print(f"Convergence reached after {epoch} epochs.")
else:
    print(f"Maximum number of epochs reached without convergence.")

# Use the trained HMM to predict tags on the test set
test_sentences_words = [[word for word, _ in sentence] for sentence in test_sentences]
predicted_tags = predict_tags(test_sentences_words, transition_probs, emission_probs, start_probs)

# Extract the actual tags from the test set
actual_tags = [[tag for _, tag in sentence] for sentence in test_sentences]

# Evaluate the accuracy of the tag predictions
accuracy = evaluate_accuracy(actual_tags, predicted_tags)
print(f"Prediction accuracy on the test set: {accuracy}")