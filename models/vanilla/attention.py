
import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
  def __init__(self, temperature: float):
    super(ScaledDotProductAttention, self).__init__()
    self.temperature = temperature
    self.softmax = nn.Softmax(-1)

  def forward(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      mask: torch.Tensor = None
  ):
    attention = torch.matmul(q, k.transpose(-1, -2)) / self.temperature
    if mask is not None:
      attention += mask
    attention = self.softmax(attention)
    return torch.matmul(attention, v), attention


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, n_heads: int):
    super(MultiHeadAttention, self).__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.d_head = d_model // n_heads
    self.attention = ScaledDotProductAttention(math.sqrt(d_model))
    self.q_w = nn.Linear(d_model, d_model)
    self.k_w = nn.Linear(d_model, d_model, bias=False)
    self.v_w = nn.Linear(d_model, d_model)
    self.linear = nn.Linear(d_model, d_model)

  def forward(
      self,
      x: torch.Tensor,
      y: torch.Tensor = None,
      mask: torch.Tensor = None
  ):
    if y is None:
      y = x
    batch_size, seq_length, d_model = x.size()
    mat = (t.reshape(batch_size, seq_length, self.n_heads, self.d_head)
           for t in (self.q_w(y), self.k_w(x), self.v_w(x)))
    q, k, v = (t.permute(0, 2, 1, 3) for t in mat)
    values, _ = self.attention(q, k, v, mask)
    values = values.reshape(batch_size, seq_length, d_model)
    return self.linear(values)
