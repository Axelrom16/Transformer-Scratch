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
    """
    Class for positional encoding.
    """
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


class LayerNormalization(nn.Module):
    """
    Class for layer normalization.
    """
    def __init__(self, eps: float=1e-6):
        """
        Args:
            eps: float, epsilon value for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1)) # Multiplied 
        self.beta = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    

class FeedForwardBlock(nn.Module):
    """
    Class for feed forward block.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Args:
            d_model: int, dimension of the model.
            d_ff: int, dimension of the feed forward layer.
            dropout: float, dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and b2
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
        """
        # x: (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)

        # x: (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        x = self.linear2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    """
    Class for multi-head attention.
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        """
        Args:
            d_model: int, dimension of the model.
            h: int, number of heads.
            dropout: float, dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h

        # Linear layers for queries, keys, and values
        self.w_q = nn.Linear(d_model, d_model) # W_q
        self.w_k = nn.Linear(d_model, d_model) # W_k
        self.w_v = nn.Linear(d_model, d_model) # W_v

        # Linear layer for the output
        self.w_out = nn.Linear(d_model, d_model) # W_o
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Attention mechanism.
        Args:
            query: Tensor, shape (batch_size, h, seq_len, d_k)
            key: Tensor, shape (batch_size, h, seq_len, d_k)
            value: Tensor, shape (batch_size, h, seq_len, d_k)
            mask: Tensor, shape (batch_size, h, seq_len, seq_len)
            dropout: nn.Dropout
        """
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) x (batch_size, h, d_k, seq_len) --> (batch_size, h, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1) # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores, value), attention_scores # (batch_size, h, seq_len, d_k)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Tensor, shape (batch_size, seq_len, d_model)
            key: Tensor, shape (batch_size, seq_len, d_model)
            value: Tensor, shape (batch_size, seq_len, d_model)
            mask: Tensor, shape (batch_size, seq_len)
        """
        batch_size = query.shape[0]

        # Linear layers for queries, keys, and values
        Q = self.w_q(query) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        K = self.w_k(key) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        V = self.w_v(value) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)

        # Split the d_model into num_heads
        Q = Q.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3) # (batch_size, h, seq_len, d_k)
        K = K.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3) # (batch_size, h, seq_len, d_k)
        V = V.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3) # (batch_size, h, seq_len, d_k)

        x, self.attention_scores = self.attention(Q, K, V, mask, self.dropout) # (batch_size, h, seq_len, d_k)

        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.h * self.d_k)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.w_out(x) 


class ResidualConnection(nn.Module):
    """
    Class for residual skip connection.
    """
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
            sublayer: nn.Module
        """
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    """
    Class for the encoder block.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
            src_mask: Tensor, shape (batch_size, seq_len)
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Class for encoder.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
            mask: Tensor, shape (batch_size, seq_len)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    Class for the decoder block.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
            encoder_output: Tensor, shape (batch_size, seq_len, d_model)
            src_mask: Tensor, shape (batch_size, seq_len)
            tgt_mask: Tensor, shape (batch_size, seq_len)
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """
    Class for decoder.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
            encoder_output: Tensor, shape (batch_size, seq_len, d_model)
            src_mask: Tensor, shape (batch_size, seq_len)
            tgt_mask: Tensor, shape (batch_size, seq_len)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Class for projection layer.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
        """
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """
    Class for the transformer model.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder  
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection

    def encode(self, src, src_mask):
        """
        Args:
            src: Tensor, shape (batch_size, src_len)
            src_mask: Tensor, shape (batch_size, src_len)
        """
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Args:
            encoder_output: Tensor, shape (batch_size, src_len, d_model)
            src_mask: Tensor, shape (batch_size, src_len)
            tgt: Tensor, shape (batch_size, tgt_len)
            tgt_mask: Tensor, shape (batch_size, tgt_len)
        """
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, tgt_len, d_model)
        """
        return self.projection(x)


def build_transformer(
    src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048
):
    """
    Function to build the transformer model.
    Args:
        src_vocab_size: int, size of the source vocabulary.
        tgt_vocab_size: int, size of the target vocabulary.
        d_model: int, dimension of the model.
        d_ff: int, dimension of the feed forward layer.
        h: int, number of heads.
        num_layers: int, number of layers.
        dropout: float, dropout rate.
    """
    # Create embedding layers
    src_embed = InputEmbeddings(src_vocab_size, d_model)
    tgt_embed = InputEmbeddings(tgt_vocab_size, d_model)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks 
    encoder = Encoder(
        nn.ModuleList(
            [
                EncoderBlock(
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout
                ) for _ in range(N)
            ]
        )
    )

    # Create decoder blocks
    decoder = Decoder(
        nn.ModuleList(
            [
                DecoderBlock(
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout
                ) for _ in range(N)
            ]
        )
    )

    # Create projection layer
    projection = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the Transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer