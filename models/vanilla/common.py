import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
  def __init__(self, d_model: int, seq_length: int, dropout: float):
    super(PositionalEncoder, self).__init__()
    self.dropout = nn.Dropout(dropout)

    position = torch.arange(seq_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         (-math.log(10000) / d_model))
    pe = torch.zeros(seq_length, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)


class LayerNormalization(nn.Module):
  def __init__(self, shape: torch.Size, eps=1e-5):
    super(LayerNormalization, self).__init__()
    self.shape = shape
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(shape))
    self.beta = nn.Parameter(torch.zeros(shape))

  def forward(self, x: torch.Tensor):
    dims = (-(i + 1) for i in range(len(self.shape)))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
    std = (var + self.eps).sqrt()
    x = (x - mean) / std
    return self.gamma * x + self.beta


class PositionwiseFeedForward(nn.Module):
  def __init__(self, d_model: int, d_hidden: int, dropout: float):
    super(PositionwiseFeedForward, self).__init__()
    self.linear1 = nn.Linear(d_model, d_hidden)
    self.linear2 = nn.Linear(d_hidden, d_model)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x: torch.Tensor):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear2(x)
    return x
