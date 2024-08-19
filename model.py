import torch 
import torch.nn as nn
import math 


class InputEmbeddings(nn.Module):
    """
    Class for input embeddings.
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: int, size of the vocabulary.
            d_model: int, dimension of the model.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
        Args:
            d_model: int, dimension of the model.
            seq_len: int, length of the sequence.
            dropout: float, dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions in the array
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe) # (Tensor will be saved in state_dict when saving the model)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

