import torch
import math
import torch.nn as nn

class InputEmbedding(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len,d_model)
        # Create a vector of shape (seq_len, 1)
        row_position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        col_position = torch.arange(0, d_model,2, dtype = torch.float)
        term = -math.log(10000.0) / d_model
        div_term = torch.exp(col_position * term)

        # Apply the sin to even positions and cos to the odd
        pe[:,0::2] = torch.sin(row_position* div_term)
        pe[:,1::2] = torch.cos(row_position* div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float= 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim =True)
        std = x.std(dim=-1, keepdim =True)

        return self.alpha * (x - mean)  / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W1 and B1

    def forward(self, x):
        # (batch_size,seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)

        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

