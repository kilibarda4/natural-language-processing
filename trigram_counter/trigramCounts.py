import collections
import math


class TrigramPerplexityCalculator:
    def __init__(self, lambdas):
        self.lambdas = lambdas
        #keep track of the counts of each ngram
        self.unigram_counts = collections.defaultdict(int)
        self.bigram_counts = collections.defaultdict(int)
        self.trigram_counts = collections.defaultdict(int)

    def fit(self, corpus):
        for sentence in corpus:
            words = sentence.split()
            for i in range(len(words)):
                self.unigram_counts[words[i]] += 1
                if i < len(words) - 1:
                    self.bigram_counts[tuple(words[i:i+2])] += 1
                if i < len(words) - 2:
                    self.trigram_counts[tuple(words[i:i+3])] += 1

    def linear_interpolation(self, ngram):
        unigram_prob = self.unigram_counts[ngram[2]] / sum(self.unigram_counts.values())
        bigram_prob = self.bigram_counts[ngram[1:]] / self.unigram_counts[ngram[1]] if self.unigram_counts[ngram[1]] != 0 else 0
        trigram_prob = self.trigram_counts[ngram] / self.bigram_counts[ngram[:2]] if self.bigram_counts[ngram[:2]] != 0 else 0
        return self.lambdas[0] * unigram_prob + self.lambdas[1] * bigram_prob + self.lambdas[2] * trigram_prob
    
    def perplexity(self, corpus):
        N = 0
        log_prob_sum = 0
        for sentence in corpus:
            words = sentence.split()
            for i in range(len(words) - 2):
                N += 1
                trigram = tuple(words[i:i+3])
                prob = self.linear_interpolation(trigram)
                if prob > 0:
                    log_prob_sum += math.log2(prob)
        return math.pow(2, -log_prob_sum / N) if N > 0 else float('inf')

def main():
    training_corpus = [
        "the cat watched children play in the park",
        "their laughter echoed near the fragrant garden",
        "the breeze spread the garden’s scent into the city",
        "it wafted past the cafe, famous for apple pie",
        "the cafe’s aroma reminded people of the nearby library",
        "the library held tales of the ancient clock tower",
        "the tower tolled, echoing in the quiet morning streets",
        "these streets, bustling by day, were peaceful at dawn",
        "at night, they lay under a star-filled sky",
        "the moonlight shone on the lake where a fisherman waited"
    ]

    test_data = [
        "the sunset painted the city sky with colors",
        "an old train’s whistle echoed past the lake",
        "soft music played in the cozy cafe"
    ]

    calculator = TrigramPerplexityCalculator([0.5, 0.4, 0.1])
    calculator.fit(training_corpus)

    for sentence in test_data:
        words = sentence.split()
        for i in range(len(words) - 2):
            trigram = tuple(words[i:i+3])
            print(f"Trigram: {trigram}, Probability: {calculator.linear_interpolation(trigram)}")
    
    print(f"Perplexity: {calculator.perplexity(test_data)}")

if __name__ == "__main__":
    main()