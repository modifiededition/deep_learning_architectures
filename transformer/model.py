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

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.h = h
        assert d_model %h ==0, "d_model is not divisible by h"

        self.d_k = d_model //h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o =  nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9) # matrix filled with -1e9 whereever mask ==0 is True

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value) , attention_scores

    def forward(self, q,k,v, mask):
        query = self.w_q(q) # (batch_size,seq_len,d_model) -> (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size,seq_len,d_model) -> (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size,seq_len,d_model) -> (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, dk) -> (batch_size, h, seq_len dk)
        # now, for each head we have complete seq_len and their d_k vector
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        # (batch_size, seq_len, d_model) -> (batch_size,seq_len,d_model)

        return self.w_o(x)





