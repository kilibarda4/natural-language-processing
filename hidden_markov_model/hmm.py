import os

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

    # In case there is no newline after the last sentence
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def create_counts(sentences):
    
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
            else:
                # If this is the first word in a sentence, increment the start tag counts
                start_tag_counts[tag] = start_tag_counts.get(tag, 0) + 1
            # Set the previous tag for the next iteration
            previous_tag = tag
    return transition_counts, emission_counts, tag_counts, start_tag_counts

# Now we convert counts to probabilities
def create_probabilities(transition_counts, emission_counts, start_tag_counts, len_sentences):
    
    transition_probabilities = {}
    emission_probabilities = {}
    start_tag_probabilities = {}

    for prev_tag, tag_transitions in transition_counts.items():
        total_transitions = sum(tag_transitions.values())
        transition_probabilities[prev_tag] = {tag: count / total_transitions
                                            for tag, count in tag_transitions.items()}

    for tag, tag_emissions in emission_counts.items():
        total_emissions = sum(tag_emissions.values())
        emission_probabilities[tag] = {word: count / total_emissions
                                    for word, count in tag_emissions.items()}
    for tag, count in start_tag_counts.items():
        start_tag_probabilities[tag] = count / len_sentences

    return transition_probabilities, emission_probabilities, start_tag_probabilities

def viterbi_algorithm(words, states, start_p, trans_p, emit_p):
    err = 1e-15
    V = [{}]
    path = {}

    for state in states:
        V[0][state] = start_p.get(state, 0) * emit_p[state].get(words[0], err)
        path[state] = [state]

    for t in range(1, len(words)):
        V.append({})
        new_path = {}

        for y in states:
            (prob, state) = max(
                (V[t-1][y0] * trans_p.get(y0, {}).get(y, 0) * emit_p[y].get(words[t], err), y0)
                for y0 in states
            )
            V[t][y] = prob
            new_path[y] = path[state] + [y]

        path = new_path

    # Choose the best path at the end
    n = len(words) - 1
    (prob, state) = max((V[n][y], y) for y in states)

    return (prob, path[state])

def pos_tagger(train_path, test_path):
    # Read and prepare the training data
    training_sentences = read_data(train_path)
    transition_counts, emission_counts, tag_counts, start_tag_counts = create_counts(training_sentences)
    transition_probs, emission_probs, start_probs = create_probabilities(transition_counts, emission_counts, start_tag_counts, len(training_sentences))

    # Read the test data
    test_sentences = read_data(test_path)

    # Initialize the tag state space
    states = list(tag_counts.keys())

    # Variables to keep track of the correct tags and the total number of tags
    correct_tags = 0
    total_tags = 0

    # Predict tags for each sentence in the test data and calculate accuracy
    for test_sentence in test_sentences:
        words, actual_tags = zip(*test_sentence)  # Separate the words and the actual tags
        _, predicted_tags = viterbi_algorithm(words, states, start_probs, transition_probs, emission_probs)

        total_tags += len(actual_tags)

        correct_tags += sum(1 for actual, predicted in zip(actual_tags, predicted_tags) if actual == predicted)
    # Calculate and return the accuracy
    accuracy = correct_tags / total_tags
    return accuracy

# Run the POS tagger and print the accuracy
accuracy = pos_tagger(training_set, testing_set)
print("Accuracy:", accuracy)