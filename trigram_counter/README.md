# Trigram Perplexity Calculator

This project features a Python script that calculates the perplexity of text corpora using trigram models with linear interpolation. The perplexity measure is crucial for evaluating language models, providing insights into how well a model predicts a sample of text. This implementation uses a linear interpolation of unigram, bigram, and trigram probabilities to estimate the likelihood of sequences of words in a given corpus.

## Overview

The Trigram Perplexity Calculator is designed to:

- Count the occurrences of unigrams, bigrams, and trigrams in a training corpus.
- Calculate the probabilities of trigrams in a test dataset using linear interpolation.
- Compute the perplexity of the test dataset, offering a quantifiable measure of the model's predictive power.

## Features

- **Linear Interpolation**: Combines unigram, bigram, and trigram probabilities with predefined lambda weights.
- **Perplexity Calculation**: Evaluates the model's performance on unseen text, useful for comparing different language models.
- **Customizable Lambda Parameters**: Allows adjustment of the lambda weights for linear interpolation, enabling fine-tuning of the model's reliance on unigram, bigram, and trigram probabilities.

## Dependencies

This script requires Python 3.x and the following Python libraries:

- `collections`
- `math`

These libraries are part of the standard Python library and do not require external installation.

## Getting Started

### Setup

1. Ensure you have Python 3.x installed on your system.
2. Clone this repository or download the script to your local machine.

### Running the Script

1. Open your terminal or command prompt.
2. Navigate to the directory containing the script.
3. Run the script:

`python trigram_perplexity_calculator.py`


## Example Usage

The script is set up to train on a predefined corpus and then calculate the probabilities of trigrams in a test dataset, followed by computing the overall perplexity. You can modify the `training_corpus` and `test_data` variables in the script to experiment with different texts.

## Customization

You can adjust the lambda weights used for linear interpolation by changing the values in the `lambdas` list passed to the `TrigramPerplexityCalculator` class instantiation. The default values are `[0.5, 0.4, 0.1]`, but you can experiment with different weights to see how they affect the perplexity outcome.