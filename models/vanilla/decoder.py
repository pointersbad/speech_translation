
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .common import LayerNormalization, PositionalEncoder, PositionwiseFeedForward
import re


class Tokenizer:
  SEP = ' '
  PAD_IDX = 0
  START_IDX = 1
  END_IDX = 2

  def __init__(self, corpus: str, seq_length: int):
    self.seq_length = seq_length
    special_tokens = ['<PAD>', '<SOS>', '<EOS>']
    words = sorted(list(set(corpus.split())))
    self.itos = [*special_tokens, *words]
    self.stoi = {v: k for k, v in enumerate(self.itos)}

  def __call__(self, x: str | torch.Tensor):
    raw = isinstance(x, str)
    dict_name = 'stoi' if raw else 'itos'
    pad = self.PAD_IDX if raw else self.itos[self.PAD_IDX]
    if raw:
      x = self.SEP.join(re.findall(r'[А-яЁё]+', x)).lower().split(' ')
    x = [getattr(self, dict_name)[token] for token in (_x for _x in x)]
    for _ in range(len(x), self.seq_length):
      x.append(pad)
    return torch.tensor(x).detach() if raw else x

  def __len__(self):
    return len(self.itos)


class Embedding(nn.Module):
  def __init__(
      self,
      d_model: int,
      seq_length: int,
      dropout: float,
  ):
    super(Embedding, self).__init__()
    with open('vocab.txt', 'r') as file:
      self.tokenizer = Tokenizer(file.read(), seq_length)
    self.embedding = nn.Embedding(len(self.tokenizer), d_model)
    self.pe = PositionalEncoder(d_model, seq_length, dropout)
    self.dropout = nn.Dropout(dropout)
    self.seq_length = seq_length

  def forward(self, x: str | torch.Tensor):
    if isinstance(x, str):
      x = self.tokenizer(x)
      x = self.embedding(x)
    x = self.pe(x.reshape(1, self.seq_length, -1))
    return self.dropout(x)


class DecoderBlock(nn.Module):
  def __init__(
      self,
      d_model: int,
      d_hidden: int,
      n_heads: int,
      dropout: float
  ):
    super(DecoderBlock, self).__init__()
    self.attention = MultiHeadAttention(d_model, n_heads)
    self.norm1 = LayerNormalization([d_model])
    self.dropout1 = nn.Dropout(dropout)
    self.cross_attention = MultiHeadAttention(d_model, n_heads)
    self.norm2 = LayerNormalization([d_model])
    self.dropout2 = nn.Dropout(dropout)
    self.ffnn = PositionwiseFeedForward(d_model, d_hidden, dropout)
    self.norm3 = LayerNormalization([d_model])
    self.dropout3 = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    _y = y
    y = self.attention(y, mask=mask)
    y = self.dropout1(y)
    y = self.norm1(y + _y)
    _y = y
    y = self.cross_attention(x, y)
    y = self.dropout2(y)
    y = self.norm2(y + _y)
    _y = y
    y = self.ffnn(y)
    y = self.dropout3(y)
    y = self.norm3(y + _y)
    return y


class SequentialDecoder(nn.Sequential):
  def forward(self, *args):
    for module in self._modules.values():
      y = module(*args)
    return y


class Decoder(nn.Module):
  def __init__(
      self,
      d_model: int,
      d_hidden: int,
      n_heads: int,
      n_layers: int,
      seq_length: int,
      dropout: float
  ):
    super(Decoder, self).__init__()
    self.mask = torch.full([seq_length, seq_length], float('-inf')).triu(1)
    self.embedding = Embedding(d_model, seq_length, dropout)
    self.layers = SequentialDecoder(*(
        DecoderBlock(d_model, d_hidden, n_heads, dropout)
        for _ in range(n_layers)
    ))

  def forward(self, x: torch.Tensor, y: str | torch.Tensor):
    y = self.embedding(y)
    return self.layers(x, y, self.mask)
