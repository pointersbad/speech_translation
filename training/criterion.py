import torch
import torch.nn.functional as F
from torch.nn import Module
from geomloss import SamplesLoss


class OTConfig:
  def __init__(self, loss: str, p: int, blur: float, scaling: float):
    self.loss = loss
    self.p = p
    self.blur = blur
    self.scaling = scaling


class CTCWassersteinLoss(Module):
  def __init__(
      self,
      ctc_weight: float,
      ot_weight: float,
      ot_sampling_config: OTConfig,
      pad_idx: int,
      eos_idx: int,
      ot_position_weight=0.0,
      attn_weight_text=0.0,
      attn_weight_speech=0.0,
      gamma=0.0,  # label smoothing

  ):
    super().__init__()
    self.ctc_weight = ctc_weight
    self.ot_weight = ot_weight
    self.ot_sampling_config = vars(ot_sampling_config)
    self.pad_idx = pad_idx
    self.eos_idx = eos_idx
    if ot_position_weight > 0.0:
      self.ot_position_weight = torch.nn.Parameter(
          torch.Temsor([ot_position_weight]))
    self.attn_weight_text = attn_weight_text
    self.attn_weight_speech = attn_weight_speech
    self.gamma = gamma

    if self.ot_weight > 0.0:
      self.ot_loss = SamplesLoss(**self.ot_sampling_config)

  def forward(
      self,
      pred: torch.Tensor,
      target: torch.Tensor,
      x: torch.Tensor,
      # encoder outputs
      speech_out: torch.Tensor,
      text_out: torch.Tensor
  ):
    loss = self.ctc_weight * self.ctc(pred, x)
    loss += self.ot_weight * self.ot(speech_out, text_out)
    if self.attn_weight_text > 0.0:
      loss += self.attn_weight_text * self.ce(pred, target)
    if self.attn_weight_speech > 0.0:
      loss += self.attn_weight_speech * self.ce(speech_out, text_out)
    return loss

  def lprobs(self, pred: torch.Tensor, target: torch.Tensor):
    lprobs = F.log_softmax(pred, dim=-1).contiguous()
    return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

  def ce(self, pred: torch.Tensor, target: torch.Tensor):
    lprobs = F.log_softmax(pred, dim=-1).contiguous()
    return F.nll_loss(lprobs, target, ignore_index=self.pad_idx)

  def ctc(self, pred: torch.Tensor, target: torch.Tensor):
    lprobs = F.log_softmax(pred, dim=-1).contiguous()
    input_lengths = lprobs.new_full(
        (lprobs.size(1),),
        lprobs.size(0),
        dtype=torch.long
    )

    pad_mask = target not in (self.pad_idx, self.eos_idx)
    targets_flat = target.masked_select(pad_mask)
    target_lengths = pad_mask.sum(-1)

    with torch.backends.cudnn.flags(enabled=False):
      ctc_loss = F.ctc_loss(
          lprobs,
          targets_flat,
          input_lengths,
          target_lengths,
          reduction="sum",
          zero_infinity=True,
      )
    kl_loss = 0.0
    if self.gamma > 0:
      kl_loss = F.kl_div(
          lprobs.transpose(0, 1),
          torch.full_like(lprobs.transpose(0, 1),
                          1 / (lprobs.size(-1) - 1)),
          reduction="batchmean",
      )
    return (1. - self.gamma) * ctc_loss + self.gamma * kl_loss

  def ot(self, speech_out: torch.Tensor, text_out: torch.Tensor):
    speech_out, text_out = (out / torch.linalg.norm(
        out, dim=- 1, keepdim=True
    ) for out in (speech_out, text_out))
    text_out = text_out / \
        torch.linalg.norm(text_out, dim=-1, keepdim=True)
    if self.ot_position_weight[0] > 0.0:
      S, B, _ = speech_out.size()
      T = text_out.size()[0]
      speech_lens, text_lens = (lens.new_full(
          (lens.size(1),),
          lens.size(0),
          dtype=torch.long
      ) for lens in (speech_lens, text_lens))
      speech_lens[speech_lens <= 1] = 2
      text_lens[text_lens <= 1] = 2
      speech_pos, text_pos = (torch.matmul(
          torch.Tensor(
              range(dim),
              dtype=torch.float,
              device=speech_out.device
          ).unsqueeze(-1),
          torch.ones((1, B), device=speech_out.device)
      ) for dim in (S, T))
      speech_pos, text_pos = (
          pos / (lens - 1).unsqueeze(0)
          for pos, lens in zip(
              (speech_pos, text_pos), (speech_lens, text_lens)))
      speech_pos[speech_pos > 1] = 1e9
      text_pos[text_pos > 1] = 1e9
      speech_out, text_out = (torch.cat(
          (out, pos.unsqueeze(-1)), dim=-1)
          for out, pos in zip((speech_out, text_out), (speech_pos, text_pos)))
    with torch.cuda.amp.autocast(enabled=False):
      loss: torch.Tensor = self.ot_loss(
          speech_out.float().transpose(0, 1).contiguous(),
          text_out.float().transpose(0, 1).contiguous()
      ).sum()
    return loss
