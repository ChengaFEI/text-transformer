import json
import random
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from typing import List
from tokenizers import CharBPETokenizer
from typing import Union
from tokenizers import ByteLevelBPETokenizer
from collections import Counter

BATCH_SIZE = 128

def create_tokenizer(tokenization_level: str) -> Union[Tokenizer, ByteLevelBPETokenizer]:
    """Create a tokenizer based on the specified level of tokenization for n-gram."""
    if tokenization_level == "character": # character-level tokenization
        tokenizer = Tokenizer(models.Model())
        # tokenizer = Tokenizer.from_pretrained("gpt2")

        # Manually add tokens for each ASCII character (adjust range if needed)
        for i in range(32, 127):
            tokenizer.add_tokens([chr(i)])

        # Set up a template processor to not add any special tokens
        tokenizer.post_processor = TemplateProcessing(
            single="$A",
            special_tokens=[],
        )
        
        # check if tokenization is working ok
        encoded_output = tokenizer.encode("This is a test sentence.")
        print(encoded_output.tokens)

    elif tokenization_level == "word": # word-level tokenization
        # Initialize a WordLevel tokenizer model
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        # Add a Whitespace pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # train the tokenizer with our training data
        train_path = './data/lm_data/treebank-sentences-train_vocab.txt'
        trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
        print('training the word-level tokenizer...')
        tokenizer.train([train_path], trainer)

        # Ensure that the tokenizer uses the [UNK] token correctly
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

        # check if tokenization is working ok
        encoded_output = tokenizer.encode("This is a test sentence.")
        print(encoded_output.tokens)
        
    elif tokenization_level == "subword": # subword-level tokenization
        # Initialize a BPE tokenizer model
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        # Add a Bttelevel pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        # train the tokenizer with our training data
        train_path = './data/lm_data/treebank-sentences-train_vocab.txt'
        trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
        print('training the subword-level tokenizer...')
        tokenizer.train([train_path], trainer)

        # check if tokenization is working ok
        encoded_output = tokenizer.encode("This is a test sentence.")
        print(encoded_output.tokens)
    
    else:
        raise ValueError(f"Unsupported tokenization level: {tokenization_level}")

    return tokenizer

def load_data(
    tokenization_level: str,
    model_type: str = "transformer",
):
    """
    Function for loading data for language modeling and WER computation. You
    may modify the function header and outputs as necessary.

    Inputs:
        tokenization_level (str): The level at which to tokenize the input
        model_type (str): n_gram or transformer
    """

    print(f"Loading data for {model_type} model...")

    # Step 1: Load the treebank data
    print("Loading treebank data...")

    treebank_file_paths = [
        "./data/lm_data/treebank-sentences-train.txt",
        "./data/lm_data/treebank-sentences-test.txt",
        "./data/lm_data/treebank-sentences-dev.txt",
    ]

    wer_file_paths = [
        "./data/wer_data/dev_sentences.json",
        "./data/wer_data/test_sentences.json",
    ]

    if model_type == "n_gram": # n-gram case

        tokenizer = create_tokenizer(tokenization_level)

        def tokenize_n_gram(data, tokenization_level=tokenization_level):
            # now compatible for all tokenization levels (char-, word-, and subword- level)
            tokenization_level == tokenization_level
            tokenized_data = []

            for line in data:
                if tokenization_level == 'character' or tokenization_level == 'subword' or tokenization_level == 'word':
                    encoded = tokenizer.encode("<bos>" + line + "<eos>") 
                    tokenized_data.append(encoded.ids)
                else:
                    raise ValueError("Unsupported tokenizer type")

            return tokenized_data
        
        treebank_datasets = []
        for file_path in treebank_file_paths:
            data = load_from_file(file_path)
            treebank_datasets.append(tokenize_n_gram(data, tokenization_level))

        train_size = int(0.8 * len(treebank_datasets[0]))
        val_size = len(treebank_datasets[0]) - train_size

        train_data, val_data = split_random(treebank_datasets[0])

        treebank_datasets = [train_data, val_data] + treebank_datasets[1:]

        wer_datasets = []
        for file_path in wer_file_paths:
            with open(file_path, "r") as f:
                wer_data = json.load(f)

                for key, value in wer_data.items():
                    wer_data[key]["tensors"] = tokenize_n_gram(value["sentences"])

                wer_datasets.append(wer_data)

        return tuple(treebank_datasets + wer_datasets + [0])

    else: # transformer case
        treebank_datasets = [
            TextDataset(file_path, tokenization_level)
            for file_path in treebank_file_paths
        ]

        vocab_size = (
            len(treebank_datasets[0].tokenizer.get_vocab()) + 1
        )  # Add 1 for padding token  # noqa: E501

        train_size = int(0.8 * len(treebank_datasets[0]))
        val_size = len(treebank_datasets[0]) - train_size
        train_data, val_data = random_split(
            treebank_datasets[0], [train_size, val_size]
        )
        treebank_datasets = [train_data, val_data] + treebank_datasets[1:]

        loaders = [
            DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            for dataset in treebank_datasets
        ]

        print("Treebank data loaded.")

        # Step 2: Load the WER data
        print("Loading WER data...")

        wer_datasets = []
        for file_path in wer_file_paths:
            with open(file_path, "r") as f:
                wer_data = json.load(f)

                for key, value in wer_data.items():
                    wer_data[key]["tensors"] = DataLoader(
                        treebank_datasets[2].tokenize_data(value["sentences"]),
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                    )

                wer_datasets.append(wer_data)

        print("WER data loaded.")

        return tuple(loaders + wer_datasets + [vocab_size])


def split_random(data, train_ratio=0.8):
    random.seed(42)

    data_shuffled = data.copy()
    random.shuffle(data_shuffled)

    split_point = int(len(data_shuffled) * train_ratio)

    return data_shuffled[:split_point], data_shuffled[split_point:]

def tokenize_n_gram(): 

    tokenized_data = []
    tokenizer = create_character_tokenizer()

    for line in tokenized_data:
        encoded = tokenizer.encode(line)
        tensor = torch.tensor(encoded.ids, dtype=torch.long)
        tokenized_data.append(tensor)

def create_character_tokenizer() -> Tokenizer: 
    """Create a character-level tokenizer for transformer."""

    tokenizer = Tokenizer(models.Model())
    # tokenizer = Tokenizer.from_pretrained("gpt2")

    # Manually add tokens for each ASCII character (adjust range if needed)
    for i in range(32, 127):
        tokenizer.add_tokens([chr(i)])

    # Set up a template processor to not add any special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[],
    )

    return tokenizer

def load_from_file(file_path: str) -> List[str]:
    """Load data from a file."""

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenization_level: str = "character"):
        """
        Custom Dataset for loading and tokenizing text data.

        Args:
        - file_path (str): Path to the text file containing data.
        - tokenization_level (str): Level of tokenization ('character').
        """

        # Step 0: Make sure the tokenization level is valid
        assert (
            tokenization_level == "character"
        ), "Only character-level tokenization is supported."

        # Step 1: Load the data
        self.data = load_from_file(file_path)

        # Step 2: Tokenize the data
        self.tokenization_level = tokenization_level
        self.tokenizer = create_character_tokenizer()
        self.tokenized_data = self.tokenize_data(self.data)

    def tokenize_data(self, data: List[str]) -> List[List[int]]:
        """Tokenize data at the character level using the Tokenizer class."""

        tokenized_data = []

        for line in data:
            encoded = self.tokenizer.encode(line)
            tensor = torch.tensor(encoded.ids, dtype=torch.long)
            tokenized_data.append(tensor)

        tokenized_data = pad_sequence(
            tokenized_data,
            batch_first=True,
            padding_value=0,
        )
        tokenized_data = [(tensor[:-1], tensor[1:]) for tensor in tokenized_data]

        return tokenized_data

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        """Return a single tokenized item from the dataset."""
        return self.tokenized_data[idx]
        # self.features = pad_sequence(self.features, batch_first=True, padding_value=0)  # noqa: E501


def custom_collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length for each batch.

    Args:
        batch: A list of tensors from the dataset.

    Returns:
        A batch of sequences padded to the same length.
    """

    # Separate the input and target sequences (if necessary)
    sequences = batch

    # Pad sequences to the longest sequence in the batch
    # padding_value is the value used for padding
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0,
    )

    return padded_sequences