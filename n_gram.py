"""
n-gram language model for Assignment 2: Starter code.
"""

from collections import defaultdict
import csv
import math
import os

# import sys
import argparse
import random
from typing import List, Any

# from tqdm import tqdm
# from collections import Counter
import numpy as np
import pandas as pd

from load_data import load_data
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer


def get_args():
    """
    You may freely add new command line arguments to this function.
    """
    parser = argparse.ArgumentParser(description="n-gram model")
    parser.add_argument(
        "-t",
        "--tokenization_level",
        type=str,
        default="character",
        help="At what level to tokenize the input data",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=4,
        help="The value of n to use for the n-gram model",
    )
    parser.add_argument(
        "-e",
        "--experiment_name",
        type=str,
        default="testing",
        help="What should we name our experiment?",
    )
    parser.add_argument(
        "-s",
        "--num_samples",
        type=int,
        default=10,
        help="How many samples should we get from our model??",
    )
    parser.add_argument(
        "-x",
        "--max_steps",
        type=int,
        default=40,
        help="What should the maximum output length of our samples be?",
    )
    args = parser.parse_args()
    return args


class NGramLM:
    """
    N-gram language model
    """

    def __init__(self, n: int):
        """
        Initializes the n-gram model. You may add keyword arguments to this function
        to modify the behavior of the n-gram model. The default behavior for unit tests should
        be that of an n-gram model without any label smoothing.

        Important for unit tests: If you add <bos> or <eos> tokens to model inputs, this should
        be done in data processing, outside of the NGramLM class.

        Inputs:
            n (int): The value of n to use in the n-gram model
        """  # noqa: E501
        self.n = n
        self.n_gram_probs = {}
        self.context_counts = defaultdict(int)
        self.n_gram_counts = defaultdict(int)
        self.vocab_size = 0

    def log_probability(self, model_input: List[Any], base=np.e, lambdas=None, k=None):
        """
        Returns the log-probability of the provided model input.

        Inputs:
            model_input (List[Any]): The list of tokens associated with the input text.
            base (float): The base with which to compute the log-probability
        """  # noqa: E501
        if lambdas is None:
            lambdas = np.zeros(self.n)
            lambdas[-1] = 1

        model_input = ["<none>"] * (self.n - 1) + model_input

        log_prob = 0.0
        for i in range(len(model_input) - self.n + 1):
            interpolated_prob = 0.0
            for j, lambd in enumerate(lambdas, 1):
                if lambd == 0:
                    continue

                ngram = tuple(model_input[i + self.n - j : i + self.n])
                if k is None:
                    prob = self.n_gram_probs.get(ngram, 0.5)  # 1e-2)
                else:
                    ngram_count = self.n_gram_counts.get(ngram, 0)
                    context = ngram[:-1]

                    if len(context) == 0:
                        total_tokens = sum(
                            self.n_gram_counts[gram]
                            for gram in self.n_gram_counts
                            if len(gram) == 1
                        )
                    else:
                        total_tokens = self.context_counts.get(context, 0)

                    prob = (ngram_count + k) / (total_tokens + k * self.vocab_size)

                interpolated_prob += lambd * prob
            log_prob += math.log(interpolated_prob, base)

        return log_prob

    def learn(self, training_data: List[List[Any]]):
        """
        Function for learning n-grams from the provided training data. You may
        add keywords to this function as needed, provided that the default behavior
        is that of an n-gram model without any label smoothing.
        Inputs:
            training_data (List[List[Any]]): A list of model inputs, which should each be lists
                                             of input tokens
        """  # noqa: E501

        for input in training_data:
            input = ["<none>"] * (self.n - 1) + input
            for i in range(len(input) - self.n + 1):

                for j in range(1, self.n + 1):

                    n_gram = tuple(input[i : i + j])
                    self.n_gram_counts[n_gram] += 1

                    if j > 1:
                        self.context_counts[n_gram[:-1]] += 1

        total_tokens = sum(
            self.n_gram_counts[n_gram]
            for n_gram in self.n_gram_counts
            if len(n_gram) == 1
        )

        # After updating all counts, we compute the probabilities
        for n_gram, count in self.n_gram_counts.items():
            if len(n_gram) == 1:
                self.n_gram_probs[n_gram] = count / total_tokens
            else:
                context_count = self.context_counts[n_gram[:-1]]
                if context_count > 0:
                    self.n_gram_probs[n_gram] = count / context_count

        self.vocab_size = len(
            set([token for n_gram in self.n_gram_probs.keys() for token in n_gram])
        )


def main():
    # Get key arguments
    args = get_args()

    # Get the data for language-modeling and WER computation
    tokenization_level = args.tokenization_level
    model_type = "n_gram"

    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data, _ = (
        load_data(tokenization_level, model_type)
    )

    # Initialize and "train" the n-gram model
    n = args.n
    model = NGramLM(n)
    model.learn(train_data)

    # Evaluate model perplexity
    val_perplexity = evaluate_perplexity(model, val_data)
    print(f"Model perplexity on the val set: {val_perplexity}")
    dev_perplexity = evaluate_perplexity(model, dev_data)
    print(f"Model perplexity on the dev set: {dev_perplexity}")
    test_perplexity = evaluate_perplexity(model, test_data)
    print(f"Model perplexity on the test set: {test_perplexity}")

    # Evaluate model WER
    experiment_name = args.experiment_name

    dev_wer_savepath = os.path.join(
        "results", f"{experiment_name}_n_gram_dev_wer_predictions.csv"
    )
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
    dev_wer = compute_wer("data/wer_data/dev_ground_truths.csv", dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join(
        "results", f"{experiment_name}_n_gram_test_wer_predictions.csv"
    )
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)


if __name__ == "__main__":
    main()
