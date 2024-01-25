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

class ResidualConnection(nn.Module):

    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x, sublayer):
        return x+ self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.resididual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x, src_mask):
        # In the encoder we used mask to apply masking in padding tokens
        x = self.resididual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.resididual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)

        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention,
                  cross_attention_bloc: MultiHeadAttention,
                  feed_forward_block: FeedForwardBlock,
                  dropout:float):

        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_bloc = cross_attention_bloc
        self.feed_forward_block = feed_forward_block
        self.resididual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.resididual_connections[0](x, lambda x : self.self_attention_block(x,x,x,tgt_mask))
        x = self.resididual_connections[1](x, lambda x: self.cross_attention_bloc(x, encoder_output,encoder_output, src_mask))
        x = self.resididual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask, tgt_mask)

        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbedding,
                 tgt_embed: InputEmbedding,
                 src_pos_embed: PositionalEmbedding,
                 tgt_pos_embed: PositionalEmbedding,
                 projection_layer:ProjectionLayer
                 ):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos_embed = src_pos_embed
        self.tgt_pos_embed = tgt_pos_embed
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos_embed(src)
        return self.encoder(src,src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos_embed(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size:int,
                      tgt_vocab_size:int,
                      src_seq_len:int,
                      tgt_seq_len:int,
                      d_model: int = 512,
                      N:int = 6,
                      h: int= 8,
                      dropout:float = 0.1,
                      d_ff = 2048):

    # create the input embedding
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model,tgt_vocab_size)

    # create the positional ecoding layer
    src_pos_embed = PositionalEmbedding(d_model,src_seq_len, dropout)
    tgt_pos_embed = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos_embed, tgt_pos_embed, projection_layer)

    # Initialize the parameters

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


