import numpy as np
import pandas as pd
from typing import List, Any

from evaluate import load

from load_data import tokenize_n_gram


def create_times_x_lambdas(n, x=3):
    lambdas = np.zeros(n)
    lambdas[-1] = 1.0

    # Makes it so they increase times x each n
    for i in range(n - 2, -1, -1):
        lambdas[i] = lambdas[i + 1] / x

    # Normalize so they sum to 1
    lambdas /= np.sum(lambdas)

    return lambdas


def rerank_sentences_for_wer(model: Any, wer_data: List[Any], savepath: str):
    """
    Function to rerank candidate sentences in the HUB dataset. For each set of sentences, you must
    assign each sentence a score in the form of the sentence's acoustic score plus the sentence's
    log probability. You should then save the top scoring sentences in a .csv file similar to those
    found in the results directory.

    Inputs:
        model (Any): An n-gram or Transformer model.
        wer_data (List[Any]): Processed data from the HUB dataset.
        savepath (str): The path to save the csv file pairing sentence set ids and the top ranked sentences.
    """  # noqa: E501

    reranked_sentences_list = []

    for id, data in wer_data.items():
        tokenizations = data["tensors"]
        sentences = data["sentences"]
        acoustic_scores = data["acoustic_scores"]

        # Compute log probabilities for each sentence
        if isinstance(data["tensors"], list):
            log_probs = []
            for sentence in tokenizations:
                log_probs.append(
                    model.log_probability(
                        sentence,
                        base=np.e,
                        lambdas=create_times_x_lambdas(model.n, 2.5),
                        k=0.1,
                    )
                )

        else:
            log_probs = []
            for tensor in tokenizations:
                log_probs.append(model.log_probability(tensor[0], tensor[1]))

        # Combine scores and rank sentences
        combined_scores = [
            log_prob + acoustic_score
            for log_prob, acoustic_score in zip(log_probs, acoustic_scores)
        ]
        best_sentence_index = combined_scores.index(max(combined_scores))

        reranked_sentences_list.append(
            {"id": id, "sentences": sentences[best_sentence_index]}
        )

    # Save reranked sentences to CSV
    pd.DataFrame(reranked_sentences_list).to_csv(savepath, index=False)


def compute_wer(gt_path, model_path):
    # Load the sentences
    ground_truths = pd.read_csv(gt_path)["sentences"].tolist()
    guesses = pd.read_csv(model_path)["sentences"].tolist()

    # Compute wer
    wer = load("wer")
    wer_value = wer.compute(predictions=guesses, references=ground_truths)
    return wer_value
