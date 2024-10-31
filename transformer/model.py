import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositonalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros((seq_len, d_model))

        positions = torch.arange(0, seq_len).float().unsqueeze(1)
        denominator = torch.exp((torch.arange(0, d_model, 2).float()/512)* math.log(10000))

        pe[:, ::2] = torch.sin(positions*denominator)
        pe[:, 1::2] = torch.cos(positions*denominator)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x+self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.feedforward = nn.Linear(d_model, d_ff)
        self.feedforward2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.feedforward2(self.relu(self.feedforward(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, h):
        super().__init__()
        assert (d_model%h==0), "d_model should be divisible by h"

        self.d_model = d_model
        self.h =h
        self.d_k = d_model // h

        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)


    def forward(self, key, value, query, mask):
        batch_size = key.shape[0]

        key = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        query = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)


        score = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            score = score.masked_fill(mask==0, float('-inf'))

        score = torch.softmax(score, dim=-1)
        attention = torch.matmul(score, value)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(attention)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10**-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha*(x-mean) / (std+self.eps) + self.bias

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, dropout, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, h)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x : self.attention(x,x,x, mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, dropout, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, h)
        self.cross_attention = MultiHeadAttention(d_model, h)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x, mask, encoder_output):
        x = self.residual_connections[0](x, lambda x : self.attention(x,x,x, mask))
        x = self.residual_connections[1](x, lambda x : self.cross_attention(encoder_output, encoder_output, x, mask=None))
        x = self.residual_connections[2](x, self.feed_forward)
        return x

class Transformer(nn.Module):
    def __init__(self, num_of_layers, d_model, h, dropout, d_ff, src_vocab_size, tgt_vocab_size, seq_len):
        super().__init__()

        self.src_embedding = Embedding(d_model, src_vocab_size)
        self.tgt_embedding = Embedding(d_model, tgt_vocab_size)

        self.src_pos_encoding = PositonalEncoding(d_model, seq_len, dropout)
        self.tgt_pos_encoding = PositonalEncoding(d_model, seq_len, dropout)

        self.encoders= nn.ModuleList([EncoderBlock(d_model, h, dropout, d_ff) for _ in range(num_of_layers)])
        self.decoders = nn.ModuleList([DecoderBlock(d_model, h, dropout, d_ff) for _ in range(num_of_layers)])

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, mask):
        x = self.src_pos_encoding(self.src_embedding(src))

        for layer in self.encoders:
            x = layer(x, mask)
        return x
    
    def decode(self, tgt, encoder_output, mask):
        x = self.tgt_pos_encoding(self.tgt_embedding(tgt))

        for layer in self.decoders:
            x = layer(x, mask, encoder_output)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask)

        logits =  self.output_linear(decoder_output)

        return torch.softmax(logits, dim=-1)
        