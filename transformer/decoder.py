
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .common import LayerNormalization, PositionalEncoder, PositionwiseFeedForward
import re


class HypothesisBuffer:
  def __init__(self):
    self.offset = 0
    self.new = []
    self.buffer = []
    self.trunk = []

  def __call__(self, new: list):
    new = [(w, t + self.offset,) for w, t in new]
    self.new = list(new)[len(self.trunk or (None,)) - 1:]
    if len(self.new) == 0:
      return torch.tensor([])
    cn = len(self.trunk)
    nn = len(self.new)
    for i in range(1, min(cn, nn, 5) + 1):
      c = " ".join([str(self.trunk[-j][0]) for j in range(1, i + 1)][::-1])
      tail = " ".join(str(self.new[j - 1][0]) for j in range(1, i + 1))
      if c == tail:
        for _ in range(i):
          self.new.pop(0)
        break
    return self.__write()

  def __write(self):
    commit = []
    while self.new and len(self.buffer) != 0:
      w, t = self.new[0]
      if w != self.buffer[0][0]:
        break
      commit.append((w, t))
      self.offset = t
      self.buffer.pop(0)
      self.new.pop(0)
    self.buffer = self.new
    self.new = []
    self.trunk.extend(commit)
    return torch.tensor([x[0] for x in commit])


class Tokenizer:
  def __init__(self, corpus: str, seq_length: int):
    self.seq_length = seq_length
    self.START_IDX = 1
    self.END_IDX = 2
    self.PAD_IDX = 0

    special_tokens = ['<PAD>', '<SOS>', '<EOS>']
    words = sorted(list(set(corpus.split())))
    self.itos = [*special_tokens, *words]
    self.stoi = {v: k for k, v in enumerate(self.itos)}

  def __call__(self, x: str | torch.Tensor):
    raw = isinstance(x, str)
    dict_name = 'stoi' if raw else 'itos'
    pad = self.PAD_IDX if raw else self.itos[self.PAD_IDX]
    x = ' '.join(re.findall(r'[А-яЁё]+', x)).lower().split(' ') if raw else x
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
