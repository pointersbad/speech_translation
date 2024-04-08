
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
  def __init__(
      self,
      d_model=512,
      d_hidden=1024,
      n_heads=8,
      n_layers=6,
      seq_length=5000,
      dropout=0.1
  ):
    super(Transformer, self).__init__()
    args = d_model, d_hidden, n_heads, n_layers, seq_length, dropout
    self.encoder = Encoder(*args)
    self.decoder = Decoder(*args)
    self.linear = nn.Linear(d_model, len(self.decoder.embedding.tokenizer))
    self.softmax = nn.Softmax(2)

  def forward(self, x, y):
    x = self.encoder(x)
    x = self.decoder(x, y)
    return self.linear(x), x.detach()
