import argparse
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from load_data import load_data
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer

DEVICE = (
    torch.device('cuda')
    if torch.cuda.is_available()
    else torch.device('cpu')
)


class CharacterLevelTransformer(nn.Module):
    """
    For this part of the assignment, we provide you with a skeleton for the Transformer decoder. However, we've introduced numerous errors to the code! The model currently compiles, but performs the incorrect computations. You must fix them to pass the unit tests.

    You may introduce additional keyword arguments after fixing the transformer, as long as the default behavior does not stray from the skeleton provided to you.
    """  # noqa: E501

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        vocab_size: int
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=vocab_size-1
        )
        self.pos_embed = PositionalEncoding(hidden_dim, dropout)
        self.decoder = Decoder(
            num_layers,
            hidden_dim,
            num_heads,
            ff_dim,
            dropout
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.padding_idx = vocab_size - 1

    def log_probability(
        self,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        base=np.e
    ):
        """
        Computes the log-probabilities for the inputs in the given minibatch.

        Input:
            input_tokens (torch.Tensor): A tensor of shape (B, T), where B is the batch-size and T is the input length.  # noqa: E501
            target_tokens (torch.Tensor): A tensor of shape (B, T). For a given (i, j), target_tokens[i, j] should be the token following input_tokens[i, j]  # noqa: E501
        Output (torch.Tensor): A tensor of shape (B,) containing the log-probability for each example in the minibatch  # noqa: E501
        """  # noqa: E501

        with torch.no_grad():  # Move tensors to the correct device
            input_tokens = input_tokens.to(DEVICE)
            target_tokens = target_tokens.to(DEVICE)
            # Predict the next tokens
            output = self.forward(input_tokens)
            # Calculate log probabilities
            log_probs = F.log_softmax(output, dim=-1)
            # Gather log probabilities for target tokens
            target_log_probs = torch.gather(log_probs, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)  # noqa: E501
            # Mask padding tokens
            non_pad_mask = target_tokens != self.padding_idx
            masked_log_probs = target_log_probs * non_pad_mask.float()
            # Sum log probabilities over tokens
            log_probabilities = masked_log_probs.sum(dim=-1)
            # Convert to specified base
            log_probabilities = log_probabilities / np.log(base)
            return log_probabilities

    def forward(self, model_input):
        # Perform the embedding
        embeds = self.embed(model_input) * math.sqrt(self.hidden_dim)
        embeds = self.pos_embed(embeds)

        # Pass through the decoder
        mask = construct_self_attn_mask(model_input)
        decoder_output = self.decoder(embeds, mask)
        output = self.lm_head(decoder_output)
        return output


def construct_self_attn_mask(x: torch.Tensor):
    """
    The output to this function should be a mask of shape
    (1, T, T). Indices that a token can attend to should be
    set to true.

    There are two errors in this function.
    """

    T = x.size(1)

    all_ones = torch.ones(T, T)
    mask = torch.triu(all_ones, diagonal=1)
    mask = mask == 0
    mask = mask.unsqueeze(0)

    return mask.to(x.device)


class Decoder(nn.Module):

    def __init__(self, num_layers, hidden_dim, num_heads, ff_dim, dropout):
        """
        There is a single error in this function that will prevent the model from learning.
        """  # noqa: E501

        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                num_heads,
                hidden_dim,
                ff_dim,
                dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, num_heads, hidden_dim, ff_dim, dropout):
        super().__init__()

        # Attention block
        self.attn_block = MultiHeadAttention(num_heads, hidden_dim, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Feedforward block
        self.mlp_block = TransformerMLP(hidden_dim, ff_dim, dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        """
        There are two types of errors in this function.
        """

        # Apply attention to the input
        attn_output = self.attn_block(x, mask)
        # Apply dropout to the attention output
        attn_output = self.attn_dropout(attn_output)
        # Add the input (residual connection) and apply layer normalization
        x = self.attn_norm(x + attn_output)

        # Apply the feedforward network
        mlp_output = self.mlp_block(x)
        # Apply dropout to the MLP output
        mlp_output = self.mlp_dropout(mlp_output)
        # Add the output of the attention layer (residual connection) and apply layer normalization  # noqa: E501
        x = self.mlp_norm(x + mlp_output)

        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        self.h = num_heads
        self.qkv_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, query, key, value, mask):
        """
        There are three errors in this function to fix.
        """

        # Compute scaled dot-product attention
        dot_products = torch.matmul(
            query,
            key.transpose(-2, -1)
        ) / math.sqrt(self.qkv_dim)

        # Apply the mask to the dot products
        # Here, mask should indicate positions to ignore
        # if mask is not None:
        dot_products = dot_products.masked_fill(mask == 0, -1e9)

        # Softmax to obtain attention weights
        attn = F.softmax(dot_products, dim=-1)
        attn = self.dropout(attn)

        # Apply attention weights to the values
        return torch.matmul(attn, value)

    def forward(self, x, mask):
        """
        There are two errors in this function to fix
        """

        # Ensure mask has the correct shape
        mask = mask.unsqueeze(1)
        B = x.size(0)

        # Compute the query, key and value vectors
        query = self.q_proj(x).view(
            B, -1, self.h, self.qkv_dim
        ).transpose(1, 2)
        key = self.k_proj(x).view(
            B, -1, self.h, self.qkv_dim
        ).transpose(1, 2)
        value = self.v_proj(x).view(
            B, -1, self.h, self.qkv_dim
        ).transpose(1, 2)

        # Perform self-attention
        x = self.attention(query, key, value, mask)

        # Concatenate the outputs for each attention head
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.qkv_dim)
        return self.out_proj(x)


class TransformerMLP(nn.Module):

    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        There is a single error in this function to fix.
        """

        # Apply the first linear layer
        x = self.fc1(x)
        # Apply the ReLU activation function
        x = F.gelu(x)
        # Apply dropout
        x = self.dropout(x)
        # Apply the second linear layer
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encodings = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) *
            (- math.log(10000) / hidden_dim)
        )
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)
        positional_encodings = positional_encodings.unsqueeze(0)

        self.register_buffer(
            'positional_encodings',
            positional_encodings,
            persistent=False
        )

    def forward(self, x):
        x = x + self.positional_encodings[:, :x.size(1)]
        return self.dropout(x)


def train(
    model,
    train_data,
    val_data,
    dev_wer_data,
    loss_fct,
    optimizer,
    max_epochs
):
    """
    Training loop for the transformer model. You may change the header as you see fit.  # noqa: E501
    """  # noqa: E501

    for epoch in range(max_epochs):
        # print(f"Epoch {epoch}")

        model.train()
        total_loss = 0

        for batch in tqdm(train_data):
            # Prepare input and target tensors
            inputs, targets = batch[0], batch[1]
            # print("batch: ", batch, len(batch))
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            # targets = targets.reshape(-1)

            # print("inputs: ", inputs)
            # print(len(inputs[0]))
            # print("targets: ", targets)
            # print(len(targets[0]))

            # Forward pass
            outputs = model(inputs)
            # outputs = outputs.reshape(-1, outputs.size(-1))

            # Reshape outputs and targets
            outputs = outputs.reshape(-1, outputs.size(-1))  # Reshape to [N*T, C]  # noqa: E501
            targets = targets.reshape(-1)  # Reshape to [N*T]

            # print("outputs: ", outputs)
            # print(len(outputs[0][0]))

            loss = loss_fct(outputs, targets)
            # print("loss: ", loss.item())

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(train_data)}")


def get_args():
    """
    You may freely add new command line arguments to this function.
    """

    parser = argparse.ArgumentParser(description='Transformer model')
    parser.add_argument(
        '--num_layers',
        type=int, default=6,
        help="How many transformer blocks to use"
    )
    parser.add_argument(
        '--hidden_dim',
        type=int, default=512,
        help="What is the transformer hidden dimension"
    )
    parser.add_argument(
        '--num_heads',
        type=int, default=8,
        help="How many heads to use for Multihead Attention"
    )
    parser.add_argument(
        '--ff_dim',
        type=int, default=2048,
        help="What is the intermediate dimension for the feedforward layer"
    )
    parser.add_argument(
        '--dropout_p',
        type=int, default=0.1,
        help="The dropout probability to use"
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='testing_'
    )
    parser.add_argument(
        '--num_samples',
        type=int, default=10,
        help="How many samples should we get from our model??"
    )
    parser.add_argument(
        '--max_steps',
        type=int, default=40,
        help="What should the maximum output length be?"
    )
    args = parser.parse_args()
    return args


def main():
    # Get key arguments
    args = get_args()

    # Get the data
    tokenization_level = "character"
    model_type = "transformer"

    train_data, val_data, \
        dev_data, test_data, \
        dev_wer_data, test_wer_data, \
        vocab_size = \
        load_data(tokenization_level, model_type)

    # print("vocab_size: ", vocab_size)
    # print("train_data: ", train_data.dataset[0])
    # print("train_data_size: ", len(train_data.dataset[0]))

    # Initialize the transformer and train
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    num_heads = args.num_heads
    ff_dim = args.ff_dim
    dropout_p = args.dropout_p
    model = CharacterLevelTransformer(
        num_layers,
        hidden_dim,
        num_heads,
        ff_dim,
        dropout_p,
        vocab_size
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fct = nn.CrossEntropyLoss(ignore_index=model.padding_idx)
    max_epochs = 20
    train(
        model,
        train_data,
        val_data,
        dev_wer_data,
        loss_fct,
        optimizer,
        max_epochs
    )

    # Original:

    # max_epochs = 5
    # trial 1
    # Loss: 1.451767419928496
    # Model perplexity on the val set: 3.597335447949386
    # Model perplexity on the dev set: 4.332574601369229
    # Model perplexity on the test set: 3.9976707957505284
    # Dev set WER was:  0.04899777282850779

    # max_epochs = 5
    # trial 2
    # Loss: 0.712652538578941
    # Model perplexity on the val set: 1.895923686905772
    # Model perplexity on the dev set: 2.0579484059950213
    # Model perplexity on the test set: 1.9810173907145865
    # Dev set WER was:  0.04899777282850779

    # max_epochs = 10
    # trial 1
    # Epoch 9, Loss: 0.5920060059047481
    # Model perplexity on the val set: 1.7306316424113468
    # Model perplexity on the dev set: 1.8483452101483597
    # Model perplexity on the test set: 1.7952066252321244
    # Dev set WER was:  0.04899777282850779

    # max_epochs = 20
    # trial 1
    # Loss: 0.5159640904302639
    # Model perplexity on the val set: 1.6049556783231056
    # Model perplexity on the dev set: 1.7173430129291878
    # Model perplexity on the test set: 1.6758748031200852
    # Dev set WER was:  0.04899777282850779

    # max_epochs = 3
    # trial 1
    # Loss: 0.8711119653895038
    # Model perplexity on the val set: 2.1999607552084144
    # Model perplexity on the dev set: 2.4346135488073126
    # Model perplexity on the test set: 2.315613746973106
    # Dev set WER was:  0.04899777282850779

    # max_epochs = 20
    # --num_layers=1 --hidden_dim=128
    # trial 1
    # Epoch 19, Loss: 0.9746034037699258
    # Model perplexity on the val set: 3.5205123414123616
    # Model perplexity on the dev set: 3.9741733980493232
    # Model perplexity on the test set: 3.755511244793742
    # Dev set WER was:  0.04899777282850779

    # max_epochs = 20
    # --num_layers=5 --hidden_dim=512
    # Epoch 19, Loss: 1.3683571421627432
    # Model perplexity on the val set: 11.401895414027784
    # Model perplexity on the dev set: 12.108516704607405
    # Model perplexity on the test set: 11.761557030033346
    # Dev set WER was:  0.04899777282850779

    # Updated:

    # max_epochs = 20
    # --num_layers=1 --hidden_dim=128
    # Epoch 19, Loss: 0.9786826697740261
    # Model perplexity on the val set: 7.88599229871518
    # Model perplexity on the dev set: 8.421809810696038
    # Model perplexity on the test set: 8.225075720435752
    # Downloading builder script: 100% 4.49k/4.49k [00:00<00:00, 15.4MB/s]
    # Dev set WER was:  0.04899777282850779

    # max_epochs = 5
    # --num_layers=5 --hidden_dim=256
    # trial 1
    # Epoch 4, Loss: 0.6681989331602525
    # Model perplexity on the val set: 1.8260817673475853
    # Model perplexity on the dev set: 1.9704936642519493
    # Model perplexity on the test set: 1.9045458438683744
    # Dev set WER was:  0.04899777282850779

    # Evaluate model perplexity
    model.eval()
    val_perplexity = evaluate_perplexity(model, val_data)
    print(f'Model perplexity on the val set: {val_perplexity}')
    dev_perplexity = evaluate_perplexity(model, dev_data)
    print(f'Model perplexity on the dev set: {dev_perplexity}')
    test_perplexity = evaluate_perplexity(model, test_data)
    print(f'Model perplexity on the test set: {test_perplexity}')

    # Evaluate model WER
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join(
        'results',
        f'{experiment_name}transformer_dev_wer_predictions.csv'
    )
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
    dev_wer = compute_wer(
        'data/wer_data/dev_ground_truths.csv',
        dev_wer_savepath
    )
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join(
        'results',
        f'{experiment_name}transformer_test_wer_predictions.csv'
    )
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)

    # Generate text from the model
    # generation_path = os.path.join(
    #     'generations',
    #     f'{experiment_name}transformer_generation_examples.pkl'
    # )
    # num_samples = args.num_samples
    # max_steps = args.max_steps
    # model.generate(num_samples, max_steps, generation_path)


if __name__ == "__main__":
    main()
