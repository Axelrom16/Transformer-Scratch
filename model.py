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
            vocab_size: int, size of the vocabulary
            d_model: int, dimension of the model
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)