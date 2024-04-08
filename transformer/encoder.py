import librosa
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .common import LayerNormalization, PositionalEncoder, PositionwiseFeedForward


class OnlineSpeechFeatureExtractor(nn.Module):
  def __init__(
      self,
      sample_rate: int,
      segment_length: int,
      n_mfcc: int,
      d_out: int,
      hop_length=160
  ):
    super(OnlineSpeechFeatureExtractor, self).__init__()

    d_hidden = n_mfcc * 2
    conv_d_out = segment_length // hop_length // 8 - 2
    self.conv1 = nn.Conv1d(n_mfcc, d_hidden * 2, 3)
    self.conv2 = nn.Conv1d(d_hidden, d_hidden, 3)
    self.conv3 = nn.Conv1d(n_mfcc, d_hidden, 3)
    self.norm = LayerNormalization([conv_d_out])
    self.pool = nn.MaxPool2d(2, 2)
    self.relu = nn.ReLU()
    self.linear = nn.Linear(n_mfcc * conv_d_out, d_out)

    self.sample_rate = sample_rate
    self.hop_length = hop_length
    self.n_mfcc = n_mfcc

  def forward(self, y: torch.Tensor):
    mfcc = librosa.feature.mfcc(
        y=y.numpy(),
        sr=self.sample_rate,
        n_mfcc=self.n_mfcc,
        hop_length=self.hop_length)
    y = torch.tensor(mfcc).unsqueeze(0).detach()
    y = self.pool(self.relu(self.conv1(y)))
    y = self.pool(self.relu(self.conv2(y)))
    y = self.pool(self.relu(self.conv3(y)))
    y = self.norm(y).flatten()
    return self.linear(y)


class EncoderBlock(nn.Module):
  def __init__(
      self,
      d_model: int,
      d_hidden: int,
      n_heads: int,
      dropout: float
  ):
    super(EncoderBlock, self).__init__()
    self.attention = MultiHeadAttention(d_model, n_heads)
    self.norm1 = LayerNormalization([d_model])
    self.dropout1 = nn.Dropout(dropout)
    self.ffnn = PositionwiseFeedForward(d_model, d_hidden, dropout)
    self.norm2 = LayerNormalization([d_model])
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
    _x = x
    x = self.attention(x, mask)
    x = self.dropout1(x)
    x = self.norm1(x + _x)
    _x = x
    x = self.ffnn(x)
    x = self.dropout2(x)
    x = self.norm2(x + _x)
    return x


class Encoder(nn.Module):
  def __init__(
      self,
      d_model: int,
      d_hidden: int,
      n_heads: int,
      n_layers: int,
      seq_length: int,
      dropout: float,
  ):
    super(Encoder, self).__init__()
    self.pe = PositionalEncoder(d_model, seq_length, dropout)
    self.seq_length = seq_length
    self.layers = nn.Sequential(*(
        EncoderBlock(d_model, d_hidden, n_heads, dropout)
        for _ in range(n_layers)
    ))

  def forward(self, x: torch.Tensor):
    x = self.pe(x)
    return self.layers(x)
